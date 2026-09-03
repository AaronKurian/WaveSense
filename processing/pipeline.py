from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from statistics import mean, median, pstdev


@dataclass(frozen=True)
class NodePosition:
    x: float
    y: float


@dataclass
class ProcessingConfig:
    expected_rate_hz: float = 25.0
    min_motion_rate_hz: float = 5.0
    feature_bins: int = 64
    active_length_min_samples: int = 20
    subcarrier_filter_width: int = 3
    amplitude_clip_mad: float = 8.0
    lowpass_time_s: float = 0.14
    baseline_stationary_time_s: float = 12.0
    baseline_motion_time_s: float = 60.0
    motion_ema_time_s: float = 0.35
    presence_attack_time_s: float = 1.0
    presence_release_time_s: float = 3.5
    noise_floor: float = 0.35
    amplitude_noise_fraction: float = 0.035
    baseline_motion_gate: float = 0.16
    temporal_noise_window: int = 120
    temporal_noise_margin: float = 1.35
    delta_scale_alpha: float = 0.05
    delta_scale_floor: float = 0.08
    delta_response_gain: float = 1.6
    presence_deviation_threshold: float = 1.35
    presence_motion_weight: float = 0.25
    motion_threshold_low: float = 0.04
    motion_threshold_medium: float = 0.10
    motion_threshold_high: float = 0.12
    presence_threshold: float = 0.30
    imbalance_threshold: float = 0.18
    localization_min_response: float = 0.18
    localization_min_confidence: float = 0.35
    spatial_calibrated: bool = False
    hypothesis_band_count: int = 5
    hypothesis_min_activity: float = 0.22
    hypothesis_min_confidence: float = 0.34
    hypothesis_separation_ratio: float = 0.45
    max_hypotheses: int = 6
    candidate_create_threshold: float = 0.20
    candidate_release_threshold: float = 0.12
    candidate_min_persistence_frames: int = 3
    candidate_max_missed_frames: int = 8
    candidate_position_alpha: float = 0.30
    candidate_score_alpha: float = 0.35
    candidate_max_step: float = 0.16
    history_size: int = 150
    node_positions: dict[int, NodePosition] = field(
        default_factory=lambda: {
            1: NodePosition(-1.0, 0.0),
            2: NodePosition(1.0, 0.0),
        }
    )


@dataclass
class NodeFeatures:
    node_id: int
    rssi_dbm: int
    packet_rate_hz: float
    amplitude: list[float]
    filtered_amplitude: list[float]
    baseline: list[float]
    corrected: list[float]
    normalized: list[float]
    smoothed: list[float]
    median_amplitude: float
    mean_abs_deviation: float
    temporal_variance: float
    temporal_diff_energy: float
    robust_deviation: float
    variance: float
    baseline_distance: float
    motion_energy: float
    presence_score: float
    presence: bool
    motion_level: str


@dataclass
class ActivityTrack:
    track_id: str
    region: str
    display_x: float
    display_y: float
    confidence: float = 0.0
    activity: float = 0.0
    age_frames: int = 0
    evidence_frames: int = 0
    missed_frames: int = 0


@dataclass
class NodeProcessor:
    node_id: int
    config: ProcessingConfig = field(default_factory=ProcessingConfig)
    baseline: list[float] | None = None
    filtered: list[float] | None = None
    prev_filtered: list[float] | None = None
    smoothed_deviation: list[float] | None = None
    length_counts: dict[int, int] = field(default_factory=dict)
    active_length: int | None = None
    motion_score: float = 0.0
    presence_score: float = 0.0
    adaptive_delta_scale: float = 1.0
    raw_diff_history: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    diff_history: deque[float] = field(default_factory=lambda: deque(maxlen=150))
    deviation_history: deque[float] = field(default_factory=lambda: deque(maxlen=150))

    def __post_init__(self) -> None:
        self.raw_diff_history = deque(maxlen=self.config.temporal_noise_window)
        self.diff_history = deque(maxlen=self.config.history_size)
        self.deviation_history = deque(maxlen=self.config.history_size)

    def process(self, amplitudes: list[float], rssi_dbm: int, packet_rate_hz: float = 0.0) -> NodeFeatures:
        raw = sanitize_amplitudes(amplitudes)
        if not raw:
            return self._empty_features(rssi_dbm, packet_rate_hz)

        rate_hz = valid_rate(packet_rate_hz, self.config.expected_rate_hz)
        lowpass_alpha = alpha_for_rate(self.config.lowpass_time_s, rate_hz)
        motion_alpha = alpha_for_rate(self.config.motion_ema_time_s, rate_hz)
        if packet_rate_hz > 0 and packet_rate_hz < self.config.min_motion_rate_hz:
            self.motion_score = ema_scalar(self.motion_score, 0.0, motion_alpha)
            return self._held_features(raw, rssi_dbm, packet_rate_hz)
        if not self._accept_length(len(raw)):
            self.motion_score = ema_scalar(self.motion_score, 0.0, motion_alpha)
            return self._held_features(raw, rssi_dbm, packet_rate_hz)

        clipped = robust_clip(raw, self.config.amplitude_clip_mad)
        spatial_filtered = median_filter(clipped, self.config.subcarrier_filter_width)
        binned = bin_amplitudes(spatial_filtered, self.config.feature_bins)
        self.filtered = ema_vector(self.filtered, binned, lowpass_alpha)

        if self.baseline is None or len(self.baseline) != len(self.filtered):
            self.baseline = list(self.filtered)

        corrected_before = [x - b for x, b in zip(self.filtered, self.baseline)]
        signal_scale = temporal_signal_scale(self.baseline, self.config)
        normalized_before = [x / signal_scale for x in corrected_before]
        robust_deviation = robust_mean_abs(normalized_before)

        raw_diff = common_mode_removed_delta(self.filtered, self.prev_filtered) / signal_scale
        self._update_adaptive_delta_scale(raw_diff)
        self.prev_filtered = list(self.filtered)
        motion_excess = self._motion_excess(raw_diff)
        self.motion_score = ema_scalar(self.motion_score, motion_excess, motion_alpha)
        self.diff_history.append(self.motion_score)
        self.deviation_history.append(robust_deviation)

        baseline_alpha = self._baseline_alpha(rate_hz)
        self.baseline = ema_vector(self.baseline, self.filtered, baseline_alpha)

        corrected = [x - b for x, b in zip(self.filtered, self.baseline)]
        normalized = [x / signal_scale for x in corrected]
        smoothing_alpha = alpha_for_rate(self.config.lowpass_time_s * 1.8, rate_hz)
        self.smoothed_deviation = ema_vector(self.smoothed_deviation, normalized, smoothing_alpha)

        temporal_variance = pstdev(self.diff_history) if len(self.diff_history) > 1 else 0.0
        deviation_score = smoothstep(self.config.presence_deviation_threshold, self.config.presence_deviation_threshold * 2.8, robust_deviation)
        motion_presence = smoothstep(self.config.motion_threshold_low, self.config.motion_threshold_high, self.motion_score)
        instantaneous_presence = clamp01(
            deviation_score * (1.0 - self.config.presence_motion_weight)
            + motion_presence * self.config.presence_motion_weight
        )
        presence_alpha = alpha_for_rate(
            self.config.presence_attack_time_s if instantaneous_presence >= self.presence_score else self.config.presence_release_time_s,
            rate_hz,
        )
        self.presence_score = ema_scalar(self.presence_score, instantaneous_presence, presence_alpha)

        baseline_distance = mean(abs(x) for x in corrected) if corrected else 0.0
        variance = pstdev(self.smoothed_deviation) if len(self.smoothed_deviation) > 1 else 0.0
        median_amplitude = median(raw)
        mean_abs_deviation = robust_mean_abs([x - median_amplitude for x in raw])
        motion_level = classify_motion(self.motion_score, self.config)

        return NodeFeatures(
            node_id=self.node_id,
            rssi_dbm=rssi_dbm,
            packet_rate_hz=packet_rate_hz,
            amplitude=raw,
            filtered_amplitude=list(self.filtered),
            baseline=list(self.baseline),
            corrected=corrected,
            normalized=normalized,
            smoothed=list(self.smoothed_deviation),
            median_amplitude=median_amplitude,
            mean_abs_deviation=mean_abs_deviation,
            temporal_variance=temporal_variance,
            temporal_diff_energy=motion_excess,
            robust_deviation=robust_deviation,
            variance=variance,
            baseline_distance=baseline_distance,
            motion_energy=self.motion_score,
            presence_score=self.presence_score,
            presence=self.presence_score >= self.config.presence_threshold,
            motion_level=motion_level,
        )

    def _baseline_alpha(self, rate_hz: float) -> float:
        time_s = (
            self.config.baseline_stationary_time_s
            if self.motion_score < self.config.baseline_motion_gate
            else self.config.baseline_motion_time_s
        )
        return alpha_for_rate(time_s, rate_hz)

    def _update_adaptive_delta_scale(self, raw_diff: float) -> None:
        if not math.isfinite(raw_diff) or raw_diff <= 0.0:
            return
        self.adaptive_delta_scale = max(
            self.config.delta_scale_floor,
            (1.0 - self.config.delta_scale_alpha) * self.adaptive_delta_scale
            + self.config.delta_scale_alpha * raw_diff,
        )

    def _motion_excess(self, raw_diff: float) -> float:
        if len(self.raw_diff_history) < 12:
            self.raw_diff_history.append(raw_diff)
            return 0.0
        noise = median(self.raw_diff_history)
        self.raw_diff_history.append(raw_diff)
        excess = max(0.0, raw_diff - noise * self.config.temporal_noise_margin)
        return clamp(
            (excess / max(self.adaptive_delta_scale, self.config.delta_scale_floor)) * self.config.delta_response_gain,
            0.0,
            3.0,
        )

    def _accept_length(self, length: int) -> bool:
        self.length_counts[length] = self.length_counts.get(length, 0) + 1
        if self.active_length is None:
            self.active_length = length
            return True
        active_count = self.length_counts.get(self.active_length, 0)
        if self.length_counts[length] > active_count:
            self.active_length = length
            active_count = self.length_counts[length]
        return length == self.active_length or active_count < self.config.active_length_min_samples

    def _held_features(self, raw: list[float], rssi_dbm: int, packet_rate_hz: float) -> NodeFeatures:
        self.presence_score = ema_scalar(self.presence_score, 0.0, 0.01)
        median_amplitude = median(raw)
        mean_abs_deviation = robust_mean_abs([x - median_amplitude for x in raw])
        return NodeFeatures(
            node_id=self.node_id,
            rssi_dbm=rssi_dbm,
            packet_rate_hz=packet_rate_hz,
            amplitude=raw,
            filtered_amplitude=list(self.filtered or []),
            baseline=list(self.baseline or []),
            corrected=list(self.smoothed_deviation or []),
            normalized=list(self.smoothed_deviation or []),
            smoothed=list(self.smoothed_deviation or []),
            median_amplitude=median_amplitude,
            mean_abs_deviation=mean_abs_deviation,
            temporal_variance=pstdev(self.diff_history) if len(self.diff_history) > 1 else 0.0,
            temporal_diff_energy=0.0,
            robust_deviation=0.0,
            variance=pstdev(self.smoothed_deviation) if self.smoothed_deviation and len(self.smoothed_deviation) > 1 else 0.0,
            baseline_distance=0.0,
            motion_energy=self.motion_score,
            presence_score=self.presence_score,
            presence=self.presence_score >= self.config.presence_threshold,
            motion_level=classify_motion(self.motion_score, self.config),
        )

    def _empty_features(self, rssi_dbm: int, packet_rate_hz: float) -> NodeFeatures:
        self.motion_score = ema_scalar(self.motion_score, 0.0, 0.2)
        self.presence_score = ema_scalar(self.presence_score, 0.0, 0.1)
        return NodeFeatures(
            node_id=self.node_id,
            rssi_dbm=rssi_dbm,
            packet_rate_hz=packet_rate_hz,
            amplitude=[],
            filtered_amplitude=[],
            baseline=[],
            corrected=[],
            normalized=[],
            smoothed=[],
            median_amplitude=0.0,
            mean_abs_deviation=0.0,
            temporal_variance=0.0,
            temporal_diff_energy=0.0,
            robust_deviation=0.0,
            variance=0.0,
            baseline_distance=0.0,
            motion_energy=self.motion_score,
            presence_score=self.presence_score,
            presence=False,
            motion_level=classify_motion(self.motion_score, self.config),
        )


def sanitize_amplitudes(values: list[float]) -> list[float]:
    clean = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            clean.append(max(0.0, numeric))
    return clean


def robust_clip(values: list[float], clip_mad: float) -> list[float]:
    if len(values) < 4:
        return list(values)
    center = median(values)
    scale = median_absolute_deviation(values, center) or 1.0
    limit = clip_mad * scale
    return [clamp(v, center - limit, center + limit) for v in values]


def median_filter(values: list[float], width: int) -> list[float]:
    if width <= 1 or len(values) < 3:
        return list(values)
    half = max(1, width // 2)
    filtered = []
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        filtered.append(median(values[start:end]))
    return filtered


def bin_amplitudes(values: list[float], bins: int) -> list[float]:
    if bins <= 0 or not values:
        return []
    if len(values) == bins:
        return list(values)
    binned = []
    for index in range(bins):
        start = int(index * len(values) / bins)
        end = int((index + 1) * len(values) / bins)
        if end <= start:
            end = min(len(values), start + 1)
        binned.append(median(values[start:end]))
    return binned


def ema_vector(prev: list[float] | None, values: list[float], alpha: float) -> list[float]:
    alpha = clamp01(alpha)
    if prev is None or len(prev) != len(values):
        return list(values)
    return [alpha * x + (1.0 - alpha) * p for x, p in zip(values, prev)]


def ema_scalar(prev: float, value: float, alpha: float) -> float:
    return alpha * value + (1.0 - alpha) * prev


def alpha_for_rate(time_s: float, rate_hz: float) -> float:
    if time_s <= 0:
        return 1.0
    rate = max(1.0, min(100.0, rate_hz))
    return clamp01(1.0 - math.exp(-1.0 / (time_s * rate)))


def valid_rate(packet_rate_hz: float, fallback: float) -> float:
    if math.isfinite(packet_rate_hz) and packet_rate_hz > 0:
        return packet_rate_hz
    return fallback


def median_absolute_deviation(values: list[float], center: float | None = None) -> float:
    if not values:
        return 0.0
    midpoint = median(values) if center is None else center
    return median(abs(x - midpoint) for x in values)


def temporal_signal_scale(baseline: list[float], config: ProcessingConfig) -> float:
    if not baseline:
        return config.noise_floor
    return max(config.noise_floor, median(baseline) * config.amplitude_noise_fraction)


def robust_mean_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_abs = sorted(abs(x) for x in values)
    trim = len(sorted_abs) // 10
    trimmed = sorted_abs[trim:len(sorted_abs) - trim] if trim and len(sorted_abs) > 2 * trim else sorted_abs
    return mean(trimmed) if trimmed else 0.0


def vector_mean_abs_delta(current: list[float] | None, previous: list[float] | None) -> float:
    if not current or not previous or len(current) != len(previous):
        return 0.0
    return mean(abs(a - b) for a, b in zip(current, previous))


def common_mode_removed_delta(current: list[float] | None, previous: list[float] | None) -> float:
    if not current or not previous or len(current) != len(previous):
        return 0.0
    delta = [a - b for a, b in zip(current, previous)]
    center = median(delta)
    return robust_mean_abs([value - center for value in delta])


def classify_motion(motion_energy: float, config: ProcessingConfig) -> str:
    if motion_energy >= config.motion_threshold_high:
        return "HIGH"
    if motion_energy >= config.motion_threshold_medium:
        return "MEDIUM"
    if motion_energy >= config.motion_threshold_low:
        return "LOW"
    return "STATIONARY"


def fuse_two_nodes(features: dict[int, NodeFeatures], config: ProcessingConfig) -> dict[str, object]:
    ordered = [features[node_id] for node_id in sorted(features)]
    if not ordered:
        return {
            "presence": False,
            "presence_score": 0.0,
            "motion_energy": 0.0,
            "motion": "STATIONARY",
            "direction": "UNKNOWN",
            "imbalance": 0.0,
            "localization": localization_result(False, None, None, 0.0, "UNKNOWN", "no fresh node features"),
            "hypotheses": [],
        }

    presence_score = max(f.presence_score for f in ordered)
    motion_energy = max(f.motion_energy for f in ordered)
    direction = "CENTER"
    imbalance = 0.0
    localization = localization_result(False, None, None, 0.0, "UNLOCALIZED", "single node cannot estimate XY position")

    has_presence = presence_score >= config.presence_threshold

    if len(ordered) >= 2 and has_presence:
        node1, node2 = ordered[0], ordered[1]
        e1 = node_response(node1)
        e2 = node_response(node2)
        imbalance = (e2 - e1) / (e1 + e2 + 1e-9)
        if abs(imbalance) >= config.imbalance_threshold:
            direction = "TOWARD NODE 2" if imbalance > 0 else "TOWARD NODE 1"
        if config.spatial_calibrated:
            localization = estimate_two_node_location(node1, node2, e1, e2, presence_score, imbalance, config)
        else:
            localization = localization_result(
                False,
                None,
                None,
                0.0,
                "UNCALIBRATED",
                "phone-to-node link geometry is not calibrated; not emitting spatial coordinates",
            )
    elif len(ordered) >= 2:
        localization = localization_result(False, None, None, 0.0, "UNKNOWN", "no CSI presence detected")

    hypotheses = activity_hypotheses(ordered, localization, config)

    return {
        "presence": has_presence,
        "presence_score": presence_score,
        "motion_energy": motion_energy,
        "motion": classify_motion(motion_energy, config),
        "direction": direction,
        "imbalance": imbalance,
        "localization": localization,
        "hypotheses": hypotheses,
        "hypothesis_debug": hypothesis_debug(features, config),
    }


@dataclass
class ActivityTracker:
    tracks: dict[str, ActivityTrack] = field(default_factory=dict)

    def update(self, features: dict[int, NodeFeatures], config: ProcessingConfig) -> dict[str, object]:
        candidates, debug = region_candidates(features, config)
        by_key = {str(candidate["id"]): candidate for candidate in candidates}
        active_keys = set(by_key)

        for key, track in list(self.tracks.items()):
            candidate = by_key.get(key)
            if candidate is None:
                track.missed_frames += 1
                track.age_frames += 1
                track.confidence = ema_scalar(track.confidence, 0.0, config.candidate_score_alpha)
                track.activity = ema_scalar(track.activity, 0.0, config.candidate_score_alpha)
                if track.missed_frames > config.candidate_max_missed_frames:
                    del self.tracks[key]
                continue

            target_x = float(candidate["display_x"])
            target_y = float(candidate["display_y"])
            track.display_x = limited_ema(track.display_x, target_x, config.candidate_position_alpha, config.candidate_max_step)
            track.display_y = limited_ema(track.display_y, target_y, config.candidate_position_alpha, config.candidate_max_step)
            track.confidence = ema_scalar(track.confidence, float(candidate["confidence"]), config.candidate_score_alpha)
            track.activity = ema_scalar(track.activity, float(candidate["activity"]), config.candidate_score_alpha)
            track.age_frames += 1
            track.evidence_frames += 1
            track.missed_frames = 0

        for candidate in candidates:
            key = str(candidate["id"])
            if key in self.tracks:
                continue
            self.tracks[key] = ActivityTrack(
                track_id=key,
                region=str(candidate["region"]),
                display_x=float(candidate["display_x"]),
                display_y=float(candidate["display_y"]),
                confidence=float(candidate["confidence"]) * config.candidate_score_alpha,
                activity=float(candidate["activity"]) * config.candidate_score_alpha,
                age_frames=1,
                evidence_frames=1,
                missed_frames=0,
            )

        hypotheses = [
            tracked_hypothesis(track, by_key.get(track.track_id), config)
            for track in sorted(self.tracks.values(), key=lambda item: item.track_id)
            if track.evidence_frames >= config.candidate_min_persistence_frames
            and track.track_id in active_keys
            and track.missed_frames == 0
            and track.confidence >= config.candidate_release_threshold
        ][: config.max_hypotheses]

        debug["sustained_frames"] = {
            region: track.evidence_frames for region, track in sorted(self.tracks.items())
        }
        debug["missed_frames"] = {
            region: track.missed_frames for region, track in sorted(self.tracks.items())
        }
        debug["emitted"] = [hypothesis["id"] for hypothesis in hypotheses]
        debug["active_regions"] = sorted({str(candidate["region"]) for candidate in candidates})
        debug["active_keys"] = sorted(active_keys)
        return {"hypotheses": hypotheses, "hypothesis_debug": debug}


def tracked_hypothesis(
    track: ActivityTrack,
    candidate: dict[str, object] | None,
    config: ProcessingConfig,
) -> dict[str, object]:
    source = candidate.get("source", {}) if candidate else {"mode": "track_decay"}
    return activity_hypothesis(
        track.track_id,
        None,
        None,
        track.display_x,
        track.display_y,
        track.region,
        track.confidence,
        track.activity,
        "sustained tracked CSI activity candidate",
        {
            **source,
            "age_frames": track.age_frames,
            "evidence_frames": track.evidence_frames,
            "missed_frames": track.missed_frames,
            "min_persistence_frames": config.candidate_min_persistence_frames,
            "spatial_evidence": "coarse_region_only",
        },
    )


def region_candidates(features: dict[int, NodeFeatures], config: ProcessingConfig) -> tuple[list[dict[str, object]], dict[str, object]]:
    ordered = [features[node_id] for node_id in sorted(features)]
    debug = {
        "candidate_threshold": config.candidate_create_threshold,
        "release_threshold": config.candidate_release_threshold,
        "min_persistence_frames": config.candidate_min_persistence_frames,
        "unlocalized_score": 0.0,
        "left_score": 0.0,
        "center_score": 0.0,
        "right_score": 0.0,
        "active_bands": {"UNLOCALIZED": 0, "LEFT": 0, "CENTER": 0, "RIGHT": 0},
        "candidates_rejected": [],
    }
    if len(ordered) == 1:
        feature = ordered[0]
        score = clamp01(max(feature.presence_score, feature.motion_energy * 1.6))
        debug["unlocalized_score"] = score
        if not feature.presence:
            debug["candidates_rejected"].append({"reason": "single fresh node has no sustained CSI presence"})
            return [], debug
        if feature.packet_rate_hz < config.min_motion_rate_hz:
            debug["candidates_rejected"].append({"reason": "single-node packet rate below motion threshold"})
            return [], debug
        if score < config.candidate_create_threshold:
            debug["candidates_rejected"].append({"reason": "single-node score below create threshold", "score": score})
            return [], debug
        debug["active_bands"]["UNLOCALIZED"] = 1
        return [
            activity_hypothesis(
                "candidate-unlocalized",
                None,
                None,
                0.0,
                0.0,
                "UNLOCALIZED",
                score,
                clamp01(max(feature.motion_energy, feature.presence_score * 0.5)),
                "single-node sustained CSI activity; location unknown",
                {
                    "nodes": [feature.node_id],
                    "mode": "single_node_presence_candidate",
                    "spatial_evidence": "none",
                },
            )
        ], debug
    if len(ordered) < 2:
        debug["candidates_rejected"].append({"reason": "requires at least one fresh node stream"})
        return [], debug

    node1, node2 = ordered[0], ordered[1]
    if not node1.presence and not node2.presence:
        debug["candidates_rejected"].append({"reason": "no sustained CSI presence"})
        return [], debug
    if node1.packet_rate_hz < config.min_motion_rate_hz or node2.packet_rate_hz < config.min_motion_rate_hz:
        debug["candidates_rejected"].append({"reason": "packet rate below motion threshold"})
        return [], debug

    bands1 = band_responses(node1, config)
    bands2 = band_responses(node2, config)
    band_entries = []

    for index in range(min(len(bands1), len(bands2))):
        response1 = bands1[index]
        response2 = bands2[index]
        total = response1 + response2
        if total < config.hypothesis_min_activity:
            debug["candidates_rejected"].append({"band": index, "reason": "band activity below threshold", "activity": total})
            continue
        imbalance = (response2 - response1) / (total + 1e-9)
        region = coarse_region(imbalance, config.imbalance_threshold)
        agreement = 1.0 - min(1.0, abs(imbalance))
        if region == "CENTER":
            score = clamp01((total / 2.0) * (0.45 + 0.55 * agreement))
        else:
            score = clamp01((total / 2.0) * (0.55 + 0.45 * abs(imbalance)))
        debug["active_bands"][region] += 1
        debug[f"{region.lower()}_score"] = max(debug[f"{region.lower()}_score"], score)
        if score < config.candidate_create_threshold:
            if score > 0:
                debug["candidates_rejected"].append({"band": index, "region": region, "reason": "band-region score below create threshold", "score": score})
            continue
        band_entries.append({
            "index": index,
            "region": region,
            "score": score,
            "activity": clamp01(total / 2.0),
            "node1_response": response1,
            "node2_response": response2,
            "imbalance": imbalance,
        })

    band_candidates = clustered_band_candidates(band_entries, node1.node_id, node2.node_id, config)
    band_candidates.sort(key=lambda item: (item["activity"], item["confidence"]), reverse=True)
    candidates = distinct_band_region_hypotheses(band_candidates, config)
    return candidates, debug


def clustered_band_candidates(
    band_entries: list[dict[str, object]],
    node1_id: int,
    node2_id: int,
    config: ProcessingConfig,
) -> list[dict[str, object]]:
    clusters = []
    current = []
    for entry in band_entries:
        if current and (
            entry["region"] != current[-1]["region"]
            or int(entry["index"]) != int(current[-1]["index"]) + 1
        ):
            clusters.append(current)
            current = []
        current.append(entry)
    if current:
        clusters.append(current)

    candidates = []
    for cluster in clusters:
        region = str(cluster[0]["region"])
        start_index = int(cluster[0]["index"])
        end_index = int(cluster[-1]["index"])
        center_index = (start_index + end_index) / 2.0
        confidence = max(float(entry["score"]) for entry in cluster)
        activity = max(float(entry["activity"]) for entry in cluster)
        node1_response = mean(float(entry["node1_response"]) for entry in cluster)
        node2_response = mean(float(entry["node2_response"]) for entry in cluster)
        imbalance = mean(float(entry["imbalance"]) for entry in cluster)
        display_x, display_y = hypothesis_region_position(region, center_index, config.hypothesis_band_count)
        band_label = str(start_index + 1) if start_index == end_index else f"{start_index + 1}-{end_index + 1}"
        candidates.append(
            activity_hypothesis(
                f"candidate-{region.lower()}-band-{band_label}",
                None,
                None,
                display_x,
                display_y,
                region,
                confidence,
                activity,
                "sustained two-node CSI region evidence candidate",
                {
                    "nodes": [node1_id, node2_id],
                    "mode": "two_node_region_candidate",
                    "band_index": center_index,
                    "band_start": start_index,
                    "band_end": end_index,
                    "band_count": config.hypothesis_band_count,
                    "node1_response": node1_response,
                    "node2_response": node2_response,
                    "imbalance": imbalance,
                    "spatial_evidence": "coarse_region_only",
                },
            )
        )
    return candidates


def node_response(feature: NodeFeatures) -> float:
    if not feature.presence:
        return 0.0
    deviation = max(0.0, feature.robust_deviation) / 6.0
    motion = max(0.0, feature.motion_energy)
    return clamp01(feature.presence_score * 0.45 + deviation * 0.35 + motion * 0.20)


def activity_hypotheses(
    features: list[NodeFeatures],
    localization: dict[str, object],
    config: ProcessingConfig,
) -> list[dict[str, object]]:
    active = [feature for feature in features if feature.presence]
    if not active:
        return []
    if not config.spatial_calibrated:
        return []
    if len(active) == 1:
        feature = active[0]
        return [
            activity_hypothesis(
                "single-node-presence",
                None,
                None,
                0.0,
                0.0,
                "UNLOCALIZED",
                clamp01(feature.presence_score * 0.7),
                clamp01(max(feature.motion_energy, feature.presence_score * 0.5)),
                "single fresh node: presence/activity only, no XY localization",
                {
                    "nodes": [feature.node_id],
                    "mode": "single_node",
                    "spatial_evidence": "none",
                },
            )
        ]

    node1, node2 = active[0], active[1]
    bands1 = band_responses(node1, config)
    bands2 = band_responses(node2, config)
    candidates = []
    for index in range(min(len(bands1), len(bands2))):
        response1 = bands1[index]
        response2 = bands2[index]
        total = response1 + response2
        if total < config.hypothesis_min_activity:
            continue
        imbalance = (response2 - response1) / (total + 1e-9)
        region = coarse_region(imbalance, config.imbalance_threshold)
        confidence = clamp01(max(node1.presence_score, node2.presence_score) * min(1.0, total))
        if confidence < config.hypothesis_min_confidence:
            continue
        x, y = hypothesis_region_position(region, index, config.hypothesis_band_count)
        candidates.append(
            activity_hypothesis(
                f"band-{index + 1}-{region.lower()}",
                None,
                None,
                x,
                y,
                region,
                confidence,
                clamp01(total),
                "subcarrier-band CSI activity signature",
                {
                    "nodes": [node1.node_id, node2.node_id],
                    "mode": "two_node_band",
                    "band_index": index,
                    "band_count": config.hypothesis_band_count,
                    "node1_response": response1,
                    "node2_response": response2,
                    "imbalance": imbalance,
                    "spatial_evidence": "coarse_region_only",
                },
            )
        )

    candidates.sort(key=lambda item: (item["activity"], item["confidence"]), reverse=True)
    distinct = distinct_hypotheses(candidates, config)
    if distinct:
        return distinct[:config.max_hypotheses]

    if localization.get("localized") is True:
        return [
            activity_hypothesis(
                "two-node-fused",
                localization.get("x"),
                localization.get("y"),
                localization.get("x"),
                localization.get("y"),
                str(localization.get("region", "UNKNOWN")),
                float(localization.get("confidence", 0.0)),
                clamp01(max(node1.motion_energy, node2.motion_energy, max(node1.presence_score, node2.presence_score) * 0.5)),
                "single fused two-node CSI response; no separable multi-source evidence",
                {
                    "nodes": [node1.node_id, node2.node_id],
                    "mode": "two_node_fused",
                    "spatial_evidence": "coarse_region_only",
                },
            )
        ]
    return []


def band_responses(feature: NodeFeatures, config: ProcessingConfig) -> list[float]:
    values = feature.smoothed or feature.normalized or feature.corrected
    if not values or config.hypothesis_band_count <= 0:
        return []
    centered = [abs(value - median(values)) for value in values]
    bands = []
    for index in range(config.hypothesis_band_count):
        start = int(index * len(centered) / config.hypothesis_band_count)
        end = int((index + 1) * len(centered) / config.hypothesis_band_count)
        band = centered[start:max(start + 1, end)]
        deviation = robust_mean_abs(band) / max(config.presence_deviation_threshold * 2.4, 1e-9)
        motion = max(0.0, feature.motion_energy)
        presence = max(0.0, feature.presence_score)
        bands.append(clamp01(deviation * 0.58 + motion * 0.22 + presence * 0.20))
    return bands


def hypothesis_debug(features: dict[int, NodeFeatures], config: ProcessingConfig) -> dict[str, object]:
    _candidates, debug = region_candidates(features, config)
    return debug


def distinct_hypotheses(candidates: list[dict[str, object]], config: ProcessingConfig) -> list[dict[str, object]]:
    if not candidates:
        return []
    strongest = float(candidates[0]["activity"])
    strongest_by_region = {}
    for candidate in candidates:
        region = candidate["region"]
        if region not in strongest_by_region:
            strongest_by_region[region] = candidate
    candidates = list(strongest_by_region.values())
    distinct = []
    for candidate in candidates:
        activity = float(candidate["activity"])
        if activity < strongest * config.hypothesis_separation_ratio and distinct:
            continue
        distinct.append(candidate)
    return distinct


def distinct_band_region_hypotheses(candidates: list[dict[str, object]], config: ProcessingConfig) -> list[dict[str, object]]:
    if not candidates:
        return []
    strongest = float(candidates[0]["activity"])
    distinct = []
    occupied: set[tuple[str, int]] = set()
    for candidate in candidates:
        activity = float(candidate["activity"])
        if activity < strongest * config.hypothesis_separation_ratio and distinct:
            continue
        source = candidate.get("source", {})
        band_index = int(source.get("band_index", -1)) if isinstance(source, dict) else -1
        region = str(candidate.get("region", "UNKNOWN"))
        key = (region, band_index)
        if key in occupied:
            continue
        occupied.add(key)
        distinct.append(candidate)
        if len(distinct) >= config.max_hypotheses:
            break
    return distinct


def band_vertical_position(index: int, count: int) -> float:
    if count <= 1:
        return 0.0
    return ((count - 1) / 2 - index) / max(1.0, (count - 1) / 2) * 0.42


def hypothesis_region_position(region: str, index: int, count: int) -> tuple[float, float]:
    x_by_region = {
        "LEFT": -0.55,
        "CENTER": 0.0,
        "RIGHT": 0.55,
        "UNKNOWN": 0.0,
        "UNCERTAIN": 0.0,
        "UNLOCALIZED": 0.0,
    }
    y = band_vertical_position(index, count)
    return x_by_region.get(region, 0.0), y


def activity_hypothesis(
    hypothesis_id: str,
    x: object,
    y: object,
    display_x: object,
    display_y: object,
    region: str,
    confidence: float,
    activity: float,
    reason: str,
    source: dict[str, object],
) -> dict[str, object]:
    return {
        "id": hypothesis_id,
        "x": finite_or_none(x),
        "y": finite_or_none(y),
        "display_x": finite_or_none(display_x),
        "display_y": finite_or_none(display_y),
        "confidence": clamp01(confidence),
        "activity": clamp01(activity),
        "age": 0.0,
        "region": region,
        "reason": reason,
        "source": source,
    }


def finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def limited_ema(current: float, target: float, alpha: float, max_step: float) -> float:
    wanted = ema_scalar(current, target, alpha)
    delta = wanted - current
    if abs(delta) <= max_step:
        return wanted
    return current + math.copysign(max_step, delta)


def estimate_two_node_location(
    node1: NodeFeatures,
    node2: NodeFeatures,
    response1: float,
    response2: float,
    presence_score: float,
    imbalance: float,
    config: ProcessingConfig,
) -> dict[str, object]:
    pos1 = config.node_positions.get(node1.node_id)
    pos2 = config.node_positions.get(node2.node_id)
    if pos1 is None or pos2 is None:
        return localization_result(False, None, None, 0.0, "UNKNOWN", "missing node geometry")
    if not valid_position(pos1) or not valid_position(pos2):
        return localization_result(False, None, None, 0.0, "UNKNOWN", "invalid node geometry")

    total = response1 + response2
    if total < config.localization_min_response:
        return localization_result(False, None, None, 0.0, "UNKNOWN", "insufficient differential CSI response")

    weight2 = response2 / total
    x = pos1.x * (1.0 - weight2) + pos2.x * weight2
    y = pos1.y * (1.0 - weight2) + pos2.y * weight2
    balance_confidence = 1.0 - min(1.0, abs(response1 - response2) / max(total, 1e-9))
    side_confidence = min(1.0, abs(imbalance) / max(config.imbalance_threshold, 1e-9))
    confidence = clamp01(presence_score * (0.55 + 0.45 * max(balance_confidence, side_confidence)))
    region = coarse_region(imbalance, config.imbalance_threshold)

    if confidence < config.localization_min_confidence:
        return localization_result(False, None, None, confidence, "UNCERTAIN", "localization confidence below threshold")

    return localization_result(True, x, y, confidence, region, "coarse two-node differential CSI estimate")


def coarse_region(imbalance: float, threshold: float) -> str:
    if imbalance <= -threshold:
        return "LEFT"
    if imbalance >= threshold:
        return "RIGHT"
    return "CENTER"


def valid_position(position: NodePosition) -> bool:
    return math.isfinite(position.x) and math.isfinite(position.y)


def localization_result(
    localized: bool,
    x: float | None,
    y: float | None,
    confidence: float,
    region: str,
    reason: str,
) -> dict[str, object]:
    return {
        "localized": localized,
        "x": x,
        "y": y,
        "confidence": clamp01(confidence),
        "region": region,
        "reason": reason,
    }


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 1.0 if value >= edge1 else 0.0
    x = clamp01((value - edge0) / (edge1 - edge0))
    return x * x * (3.0 - 2.0 * x)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return clamp(value, 0.0, 1.0)
