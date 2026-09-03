import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from processing.pipeline import ActivityTracker, NodeFeatures, NodePosition, NodeProcessor, ProcessingConfig, fuse_two_nodes


def feature(
    node_id: int,
    presence_score: float,
    motion_energy: float,
    robust_deviation: float,
    presence: bool = True,
    motion_level: str = "LOW",
    smoothed: list[float] | None = None,
) -> NodeFeatures:
    return NodeFeatures(
        node_id=node_id,
        rssi_dbm=-45,
        packet_rate_hz=25.0,
        amplitude=[],
        filtered_amplitude=[],
        baseline=[],
        corrected=[],
        normalized=[],
        smoothed=smoothed or [],
        median_amplitude=0.0,
        mean_abs_deviation=0.0,
        temporal_variance=0.0,
        temporal_diff_energy=motion_energy,
        robust_deviation=robust_deviation,
        variance=0.0,
        baseline_distance=robust_deviation,
        motion_energy=motion_energy,
        presence_score=presence_score,
        presence=presence,
        motion_level=motion_level,
    )


def base_frame(level: float = 18.0, count: int = 64) -> list[float]:
    return [level + ((i % 9) - 4) * 0.18 + (i % 3) * 0.05 for i in range(count)]


def noisy_stationary_frame(index: int, level: float = 18.0, count: int = 64) -> list[float]:
    return [
        level
        + ((i % 9) - 4) * 0.18
        + ((index + i * 3) % 7 - 3) * 0.07
        for i in range(count)
    ]


def moved_frame(amount: float = 5.0, level: float = 18.0, count: int = 64) -> list[float]:
    frame = base_frame(level, count)
    return [
        value + amount if 12 <= i < 28 else value - amount * 0.55 if 38 <= i < 54 else value
        for i, value in enumerate(frame)
    ]


def run_frames(processor: NodeProcessor, frames: list[list[float]], rate_hz: float = 25.0):
    feature = None
    for frame in frames:
        feature = processor.process(frame, -55, rate_hz)
    return feature


def assert_finite_feature(feature):
    scalar_fields = [
        feature.median_amplitude,
        feature.mean_abs_deviation,
        feature.temporal_variance,
        feature.temporal_diff_energy,
        feature.robust_deviation,
        feature.variance,
        feature.baseline_distance,
        feature.motion_energy,
        feature.presence_score,
    ]
    assert all(math.isfinite(value) for value in scalar_fields)
    for values in [
        feature.amplitude,
        feature.filtered_amplitude,
        feature.baseline,
        feature.corrected,
        feature.normalized,
        feature.smoothed,
    ]:
        assert all(math.isfinite(value) for value in values)


def test_stationary_signal_stays_low_motion_after_baseline():
    processor = NodeProcessor(1)
    feature = run_frames(processor, [base_frame() for _ in range(80)], 25.0)

    assert feature is not None
    assert feature.motion_level == "STATIONARY"
    assert feature.motion_energy < processor.config.motion_threshold_low
    assert feature.presence_score < processor.config.presence_threshold


def test_clear_amplitude_change_produces_high_motion():
    processor = NodeProcessor(1)
    run_frames(processor, [base_frame() for _ in range(50)], 25.0)

    feature = processor.process(moved_frame(8.0), -55, 25.0)

    assert feature.motion_level == "HIGH"
    assert feature.motion_energy >= processor.config.motion_threshold_high


def test_noisy_stationary_signal_does_not_constantly_trigger_motion():
    processor = NodeProcessor(1)
    frames = [noisy_stationary_frame(i) for i in range(160)]
    feature = None
    triggered = 0
    for frame in frames:
        feature = processor.process(frame, -55, 25.0)
        if feature.motion_level != "STATIONARY":
            triggered += 1

    assert feature is not None
    assert triggered < 8
    assert feature.motion_energy < processor.config.motion_threshold_low
    assert feature.presence_score < processor.config.presence_threshold


def test_person_like_sustained_change_increases_presence():
    processor = NodeProcessor(1)
    run_frames(processor, [base_frame() for _ in range(60)], 25.0)

    feature = run_frames(processor, [moved_frame(5.0) for _ in range(90)], 25.0)

    assert feature.presence is True
    assert feature.presence_score >= processor.config.presence_threshold
    assert feature.robust_deviation > processor.config.presence_deviation_threshold


def test_return_to_baseline_reduces_presence():
    processor = NodeProcessor(1)
    run_frames(processor, [base_frame() for _ in range(60)], 25.0)
    active = run_frames(processor, [moved_frame(5.0) for _ in range(80)], 25.0)
    returned = run_frames(processor, [base_frame() for _ in range(180)], 25.0)

    assert active.presence_score > processor.config.presence_threshold
    assert returned.presence_score < processor.config.presence_threshold
    assert returned.motion_level == "STATIONARY"


def test_packet_rate_variation_keeps_stationary_stable():
    processor = NodeProcessor(1)
    rates = [18.0, 22.0, 31.0, 37.0, 25.0] * 32
    feature = None
    for idx, rate in enumerate(rates):
        feature = processor.process(noisy_stationary_frame(idx), -55, rate)

    assert feature.motion_level == "STATIONARY"
    assert feature.motion_energy < processor.config.motion_threshold_low
    assert_finite_feature(feature)


def test_low_packet_rate_does_not_report_motion_from_sparse_gaps():
    processor = NodeProcessor(1)
    run_frames(processor, [base_frame() for _ in range(80)], 25.0)

    feature = processor.process(moved_frame(8.0), -55, 1.2)

    assert feature.motion_level == "STATIONARY"
    assert feature.motion_energy < processor.config.motion_threshold_low
    assert feature.amplitude


def test_mixed_packet_lengths_do_not_create_false_motion():
    processor = NodeProcessor(1)
    short = base_frame(count=64)
    long = base_frame(count=192)
    feature = None

    for _ in range(80):
        feature = processor.process(long, -55, 30.0)
        processor.process(short, -55, 30.0)

    assert feature.motion_level == "STATIONARY"
    assert feature.motion_energy < processor.config.motion_threshold_low
    assert_finite_feature(feature)


def test_baseline_adapts_slowly_during_stationary_periods():
    processor = NodeProcessor(1)
    run_frames(processor, [base_frame() for _ in range(80)], 25.0)
    shifted = base_frame(21.0)

    early = run_frames(processor, [shifted for _ in range(20)], 25.0)
    late = run_frames(processor, [shifted for _ in range(350)], 25.0)

    assert late.baseline_distance < early.baseline_distance
    assert late.motion_level == "STATIONARY"


def test_empty_and_invalid_input_returns_finite_empty_features():
    processor = NodeProcessor(1)
    feature = processor.process([float("nan"), float("inf"), None], -55, 25.0)

    assert feature.amplitude == []
    assert feature.presence is False
    assert feature.motion_level == "STATIONARY"
    assert_finite_feature(feature)


def test_two_node_fusion_reports_strong_imbalance():
    config = ProcessingConfig(imbalance_threshold=0.1, spatial_calibrated=True)
    features = {
        1: feature(1, presence_score=0.2, motion_energy=0.02, robust_deviation=0.02),
        2: feature(2, presence_score=0.8, motion_energy=0.4, robust_deviation=8.0, motion_level="HIGH"),
    }

    fused = fuse_two_nodes(features, config)

    assert fused["presence"] is True
    assert fused["direction"] == "TOWARD NODE 2"
    assert fused["motion"] == "HIGH"
    assert fused["localization"]["localized"] is True
    assert fused["localization"]["region"] == "RIGHT"
    assert fused["localization"]["x"] > 0.0
    assert fused["hypotheses"]


def test_single_node_fusion_reports_unlocalized_presence():
    fused = fuse_two_nodes({1: feature(1, 0.8, 0.3, 4.0, motion_level="HIGH")}, ProcessingConfig())

    assert fused["presence"] is True
    assert fused["localization"]["localized"] is False
    assert fused["localization"]["region"] == "UNLOCALIZED"
    assert fused["localization"]["x"] is None
    assert "single node" in fused["localization"]["reason"]
    assert fused["hypotheses"] == []


def test_two_node_fusion_center_when_responses_are_balanced():
    features = {
        1: feature(1, presence_score=0.75, motion_energy=0.2, robust_deviation=4.0),
        2: feature(2, presence_score=0.73, motion_energy=0.2, robust_deviation=4.0),
    }

    fused = fuse_two_nodes(features, ProcessingConfig(spatial_calibrated=True))

    assert fused["localization"]["localized"] is True
    assert fused["localization"]["region"] == "CENTER"
    assert abs(fused["localization"]["x"]) < 0.1


def test_two_node_fusion_refuses_weak_localization():
    features = {
        1: feature(1, presence_score=0.1, motion_energy=0.0, robust_deviation=0.1, presence=False),
        2: feature(2, presence_score=0.1, motion_energy=0.0, robust_deviation=0.1, presence=False),
    }

    fused = fuse_two_nodes(features, ProcessingConfig())

    assert fused["presence"] is False
    assert fused["localization"]["localized"] is False
    assert fused["localization"]["region"] == "UNKNOWN"
    assert fused["hypotheses"] == []


def test_two_node_fusion_reports_node1_dominant_response():
    config = ProcessingConfig(imbalance_threshold=0.1, spatial_calibrated=True)
    features = {
        1: feature(1, presence_score=0.85, motion_energy=0.45, robust_deviation=8.0, motion_level="HIGH"),
        2: feature(2, presence_score=0.25, motion_energy=0.03, robust_deviation=0.2),
    }

    fused = fuse_two_nodes(features, config)

    assert fused["direction"] == "TOWARD NODE 1"
    assert fused["localization"]["localized"] is True
    assert fused["localization"]["region"] == "LEFT"
    assert fused["localization"]["x"] < 0.0


def test_two_node_fusion_rejects_missing_node_geometry():
    config = ProcessingConfig(spatial_calibrated=True, node_positions={1: NodePosition(-1.0, 0.0)})
    features = {
        1: feature(1, presence_score=0.8, motion_energy=0.3, robust_deviation=5.0),
        2: feature(2, presence_score=0.8, motion_energy=0.3, robust_deviation=5.0),
    }

    fused = fuse_two_nodes(features, config)

    assert fused["localization"]["localized"] is False
    assert fused["localization"]["region"] == "UNKNOWN"
    assert "missing node geometry" in fused["localization"]["reason"]


def test_two_node_fusion_rejects_invalid_node_geometry():
    config = ProcessingConfig(
        spatial_calibrated=True,
        node_positions={
            1: NodePosition(float("nan"), 0.0),
            2: NodePosition(1.0, 0.0),
        }
    )
    features = {
        1: feature(1, presence_score=0.8, motion_energy=0.3, robust_deviation=5.0),
        2: feature(2, presence_score=0.8, motion_energy=0.3, robust_deviation=5.0),
    }

    fused = fuse_two_nodes(features, config)

    assert fused["localization"]["localized"] is False
    assert fused["localization"]["x"] is None
    assert fused["localization"]["y"] is None
    assert "invalid node geometry" in fused["localization"]["reason"]


def test_two_node_hypotheses_can_represent_separable_band_activity():
    node1_signal = [5.0] * 21 + [0.3] * 22 + [1.0] * 21
    node2_signal = [0.6] * 21 + [0.4] * 22 + [5.5] * 21
    features = {
        1: feature(1, presence_score=0.85, motion_energy=0.22, robust_deviation=5.0, smoothed=node1_signal),
        2: feature(2, presence_score=0.88, motion_energy=0.24, robust_deviation=5.2, smoothed=node2_signal),
    }

    fused = fuse_two_nodes(
        features,
        ProcessingConfig(spatial_calibrated=True, hypothesis_min_activity=0.2, hypothesis_separation_ratio=0.5),
    )

    assert len(fused["hypotheses"]) >= 2
    assert {hypothesis["region"] for hypothesis in fused["hypotheses"]} >= {"LEFT", "RIGHT"}
    assert all(hypothesis["source"]["mode"] == "two_node_band" for hypothesis in fused["hypotheses"])


def test_two_node_hypotheses_fall_back_to_single_fused_source_when_not_separable():
    signal = [2.0] * 64
    features = {
        1: feature(1, presence_score=0.7, motion_energy=0.12, robust_deviation=3.0, smoothed=signal),
        2: feature(2, presence_score=0.7, motion_energy=0.12, robust_deviation=3.0, smoothed=signal),
    }

    fused = fuse_two_nodes(features, ProcessingConfig(spatial_calibrated=True, hypothesis_min_activity=0.5))

    assert len(fused["hypotheses"]) == 1
    assert fused["hypotheses"][0]["source"]["mode"] == "two_node_fused"
    assert "no separable multi-source evidence" in fused["hypotheses"][0]["reason"]


def test_uncalibrated_two_node_fusion_never_emits_spatial_hypotheses():
    features = {
        1: feature(1, presence_score=0.85, motion_energy=0.45, robust_deviation=8.0, motion_level="HIGH"),
        2: feature(2, presence_score=0.9, motion_energy=0.5, robust_deviation=9.0, motion_level="HIGH"),
    }

    fused = fuse_two_nodes(features, ProcessingConfig())

    assert fused["presence"] is True
    assert fused["localization"]["localized"] is False
    assert fused["localization"]["region"] == "UNCALIBRATED"
    assert fused["hypotheses"] == []


def tracked_after_updates(features, config=None, updates=5):
    tracker = ActivityTracker()
    config = config or ProcessingConfig()
    result = {}
    for _ in range(updates):
        result = tracker.update(features, config)
    return result


def band_signal(first=0.2, second=0.2, third=0.2, fourth=None, fifth=None):
    if fourth is None and fifth is None:
        return [first] * 21 + [second] * 22 + [third] * 21
    fourth = second if fourth is None else fourth
    fifth = third if fifth is None else fifth
    return [first] * 13 + [second] * 13 + [third] * 13 + [fourth] * 12 + [fifth] * 13


def test_activity_tracker_empty_noisy_csi_emits_zero_hypotheses():
    features = {
        1: feature(1, 0.1, 0.01, 0.1, presence=False, smoothed=band_signal(0.2, 0.25, 0.22)),
        2: feature(2, 0.1, 0.01, 0.1, presence=False, smoothed=band_signal(0.2, 0.24, 0.21)),
    }

    result = tracked_after_updates(features)

    assert result["hypotheses"] == []
    assert result["hypothesis_debug"]["candidates_rejected"]


def test_activity_tracker_sustained_left_activity_creates_left_hypothesis():
    features = {
        1: feature(1, 0.9, 0.3, 6.0, smoothed=band_signal(5.0, 0.2, 0.2)),
        2: feature(2, 0.9, 0.3, 6.0, smoothed=band_signal(0.2, 0.2, 0.2)),
    }

    result = tracked_after_updates(features)

    assert result["hypotheses"]
    assert {hypothesis["region"] for hypothesis in result["hypotheses"]} == {"LEFT"}
    assert all(hypothesis["display_x"] < 0 for hypothesis in result["hypotheses"])


def test_activity_tracker_single_node_presence_creates_unlocalized_hypothesis():
    features = {
        2: feature(2, 0.62, 0.14, 3.0, smoothed=band_signal(1.2, 0.8, 1.1)),
    }

    result = tracked_after_updates(features)

    assert [hypothesis["id"] for hypothesis in result["hypotheses"]] == ["candidate-unlocalized"]
    assert result["hypotheses"][0]["region"] == "UNLOCALIZED"
    assert result["hypotheses"][0]["x"] is None
    assert result["hypotheses"][0]["display_x"] == 0.0
    assert result["hypothesis_debug"]["unlocalized_score"] > 0.0


def test_activity_tracker_single_node_without_presence_emits_zero_hypotheses():
    features = {
        2: feature(2, 0.12, 0.02, 0.2, presence=False, smoothed=band_signal()),
    }

    result = tracked_after_updates(features)

    assert result["hypotheses"] == []
    assert result["hypothesis_debug"]["candidates_rejected"][0]["reason"] == "single fresh node has no sustained CSI presence"


def test_activity_tracker_single_node_low_rate_emits_zero_hypotheses():
    low_rate = feature(2, 0.8, 0.3, 4.0, smoothed=band_signal(2.0, 2.2, 1.9))
    low_rate.packet_rate_hz = 1.0

    result = tracked_after_updates({2: low_rate})

    assert result["hypotheses"] == []
    assert result["hypothesis_debug"]["candidates_rejected"][0]["reason"] == "single-node packet rate below motion threshold"


def test_activity_tracker_sustained_right_activity_creates_right_hypothesis():
    features = {
        1: feature(1, 0.9, 0.3, 6.0, smoothed=band_signal(0.2, 0.2, 0.2)),
        2: feature(2, 0.9, 0.3, 6.0, smoothed=band_signal(0.2, 0.2, 5.0)),
    }

    result = tracked_after_updates(features)

    assert result["hypotheses"]
    assert {hypothesis["region"] for hypothesis in result["hypotheses"]} == {"RIGHT"}
    assert all(hypothesis["display_x"] > 0 for hypothesis in result["hypotheses"])


def test_activity_tracker_sustained_center_activity_creates_center_hypothesis():
    features = {
        1: feature(1, 0.9, 0.3, 6.0, smoothed=band_signal(5.0, 0.2, 0.2)),
        2: feature(2, 0.9, 0.3, 6.0, smoothed=band_signal(5.0, 0.2, 0.2)),
    }

    result = tracked_after_updates(features)

    assert result["hypotheses"]
    assert {hypothesis["region"] for hypothesis in result["hypotheses"]} == {"CENTER"}
    assert all(abs(hypothesis["display_x"]) < 0.01 for hypothesis in result["hypotheses"])


def test_activity_tracker_two_independent_regions_emit_two_hypotheses():
    features = {
        1: feature(1, 0.9, 0.3, 6.0, smoothed=band_signal(5.0, 0.2, 0.2)),
        2: feature(2, 0.9, 0.3, 6.0, smoothed=band_signal(0.2, 0.2, 5.0)),
    }

    result = tracked_after_updates(features)

    assert {hypothesis["region"] for hypothesis in result["hypotheses"]} == {"LEFT", "RIGHT"}


def test_activity_tracker_allows_more_than_two_hypotheses_when_evidence_exists():
    features = {
        1: feature(1, 0.9, 0.3, 6.0, smoothed=[5.0, 0.2, 0.2, 5.0, 0.2]),
        2: feature(2, 0.9, 0.3, 6.0, smoothed=[0.2, 5.0, 0.2, 0.2, 5.0]),
    }

    result = tracked_after_updates(features, ProcessingConfig(hypothesis_band_count=5, hypothesis_separation_ratio=0.1))

    assert len(result["hypotheses"]) > 2
    assert len(result["hypotheses"]) <= ProcessingConfig().max_hypotheses


def test_activity_tracker_does_not_emit_stale_region_without_current_evidence():
    tracker = ActivityTracker()
    active = {
        1: feature(1, 0.9, 0.3, 6.0, smoothed=band_signal(5.0, 0.2, 0.2)),
        2: feature(2, 0.9, 0.3, 6.0, smoothed=band_signal(0.2, 0.2, 0.2)),
    }
    quiet = {
        1: feature(1, 0.1, 0.01, 0.1, presence=False, smoothed=band_signal()),
        2: feature(2, 0.1, 0.01, 0.1, presence=False, smoothed=band_signal()),
    }

    for _ in range(5):
        result = tracker.update(active, ProcessingConfig())
    assert result["hypotheses"]

    result = tracker.update(quiet, ProcessingConfig())

    assert result["hypotheses"] == []
    assert result["hypothesis_debug"]["missed_frames"]["candidate-left-band-1-2"] == 1


def test_activity_tracker_one_frame_spike_does_not_create_hypothesis():
    tracker = ActivityTracker()
    config = ProcessingConfig(candidate_min_persistence_frames=4)
    spike = {
        1: feature(1, 0.9, 0.3, 6.0, smoothed=band_signal(5.0, 0.2, 0.2)),
        2: feature(2, 0.9, 0.3, 6.0, smoothed=band_signal(0.2, 0.2, 0.2)),
    }
    quiet = {
        1: feature(1, 0.1, 0.01, 0.1, presence=False, smoothed=band_signal()),
        2: feature(2, 0.1, 0.01, 0.1, presence=False, smoothed=band_signal()),
    }

    assert tracker.update(spike, config)["hypotheses"] == []
    for _ in range(4):
        result = tracker.update(quiet, config)

    assert result["hypotheses"] == []


def test_activity_tracker_threshold_oscillation_does_not_blink():
    tracker = ActivityTracker()
    config = ProcessingConfig(candidate_min_persistence_frames=3, candidate_create_threshold=0.45)
    weak = {
        1: feature(1, 0.35, 0.04, 1.2, smoothed=band_signal(1.3, 0.2, 0.2)),
        2: feature(2, 0.35, 0.04, 1.2, smoothed=band_signal(0.2, 0.2, 0.2)),
    }

    emitted = []
    for _ in range(10):
        emitted.append(bool(tracker.update(weak, config)["hypotheses"]))

    assert emitted == [False] * 10


def test_activity_tracker_keeps_stable_candidate_id():
    features = {
        1: feature(1, 0.9, 0.3, 6.0, smoothed=band_signal(5.0, 0.2, 0.2)),
        2: feature(2, 0.9, 0.3, 6.0, smoothed=band_signal(0.2, 0.2, 0.2)),
    }

    first = tracked_after_updates(features, updates=5)["hypotheses"][0]["id"]
    second = tracked_after_updates(features, updates=8)["hypotheses"][0]["id"]

    assert first == second == "candidate-left-band-1-2"


def test_activity_tracker_emits_new_id_for_distinct_band_signature():
    tracker = ActivityTracker()
    config = ProcessingConfig(
        candidate_min_persistence_frames=1,
        candidate_release_threshold=0.1,
        candidate_max_step=0.05,
        candidate_position_alpha=1.0,
    )
    upper_band = {
        1: feature(1, 0.9, 0.3, 6.0, smoothed=band_signal(5.0, 0.2, 0.2)),
        2: feature(2, 0.9, 0.3, 6.0, smoothed=band_signal(0.2, 0.2, 0.2)),
    }
    lower_band = {
        1: feature(1, 0.9, 0.3, 6.0, smoothed=band_signal(0.2, 0.2, 5.0)),
        2: feature(2, 0.9, 0.3, 6.0, smoothed=band_signal(0.2, 0.2, 0.2)),
    }

    start = tracker.update(upper_band, config)["hypotheses"][0]
    moved = tracker.update(lower_band, config)["hypotheses"][0]

    assert start["id"] == "candidate-left-band-1-2"
    assert moved["id"] != start["id"]
    assert moved["source"]["band_index"] != start["source"]["band_index"]
