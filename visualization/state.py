from __future__ import annotations

import threading
import time

from processing.pipeline import ActivityTracker, NodeFeatures, NodePosition, NodeProcessor, ProcessingConfig, fuse_two_nodes
from receiver.csi_receiver import CsiReceiver
from visualization.stimulator import CsiStimulator

UI_VERSION = "node-measurements-v12"


class DashboardState:
    fresh_age_limit_s = 2.5

    def __init__(self, receiver: CsiReceiver, stimulator: CsiStimulator | None = None):
        self.receiver = receiver
        self.stimulator = stimulator
        self.config = ProcessingConfig()
        self.processors: dict[int, NodeProcessor] = {}
        self.features: dict[int, NodeFeatures] = {}
        self.tracker = ActivityTracker()
        self.lock = threading.Lock()

    def ingest(self) -> None:
        while True:
            packet = self.receiver.receive_once()
            if packet is None:
                continue
            with self.lock:
                processor = self.processors.setdefault(packet.node_id, NodeProcessor(packet.node_id, self.config))
                stats = self.receiver.nodes[packet.node_id]
                self.features[packet.node_id] = processor.process(
                    packet.amplitudes,
                    packet.rssi_dbm,
                    stats.rate_hz,
                )

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            nodes = {}
            receiver_snapshot = self.receiver.snapshot()
            fresh_features = {}
            for node_id, feature in sorted(self.features.items()):
                node_snapshot = receiver_snapshot["nodes"].get(str(node_id), {})
                age = node_snapshot.get("last_seen_age_s")
                if isinstance(age, (float, int)) and age <= self.fresh_age_limit_s:
                    fresh_features[node_id] = feature
                nodes[str(node_id)] = {
                    "rate_hz": feature.packet_rate_hz,
                    "rssi_dbm": feature.rssi_dbm,
                    "amplitude": feature.amplitude,
                    "filtered_amplitude": feature.filtered_amplitude,
                    "baseline": feature.baseline,
                    "smoothed": feature.smoothed,
                    "corrected": feature.corrected,
                    "normalized": feature.normalized,
                    "median_amplitude": feature.median_amplitude,
                    "mean_abs_deviation": feature.mean_abs_deviation,
                    "temporal_variance": feature.temporal_variance,
                    "temporal_diff_energy": feature.temporal_diff_energy,
                    "robust_deviation": feature.robust_deviation,
                    "variance": feature.variance,
                    "baseline_distance": feature.baseline_distance,
                    "motion_energy": feature.motion_energy,
                    "presence_score": feature.presence_score,
                    "presence": feature.presence,
                    "motion_level": feature.motion_level,
                }
            fused = fuse_two_nodes(fresh_features, self.config)
            tracked = self.tracker.update(fresh_features, self.config)
            fused["hypotheses"] = tracked["hypotheses"]
            fused["hypothesis_debug"] = tracked["hypothesis_debug"]
            node_status = build_node_status(receiver_snapshot, self.config.node_positions, self.fresh_age_limit_s)
            return {
                "ui_version": UI_VERSION,
                "time": time.time(),
                "diagnostics": build_diagnostics(receiver_snapshot, node_status, self.stimulator),
                "receiver": receiver_snapshot,
                "geometry": serialize_geometry(self.config.node_positions),
                "node_status": node_status,
                "measurements": build_measurements(nodes, node_status),
                "nodes": nodes,
                "fused": fused,
            }


def serialize_geometry(node_positions: dict[int, NodePosition]) -> dict[str, dict[str, float]]:
    return {
        str(node_id): {"x": position.x, "y": position.y}
        for node_id, position in sorted(node_positions.items())
    }


def build_node_status(
    receiver_snapshot: dict[str, object],
    node_positions: dict[int, NodePosition],
    fresh_age_limit_s: float,
) -> dict[str, dict[str, object]]:
    receiver_nodes = receiver_snapshot.get("nodes", {})
    statuses = {}
    for node_id in sorted(node_positions):
        key = str(node_id)
        node = receiver_nodes.get(key, {}) if isinstance(receiver_nodes, dict) else {}
        age = node.get("last_seen_age_s") if isinstance(node, dict) else None
        fresh = isinstance(age, (float, int)) and age <= fresh_age_limit_s
        statuses[key] = {
            "expected": True,
            "fresh": fresh,
            "online": isinstance(node, dict) and bool(node),
            "last_seen_age_s": age,
            "rate_hz": node.get("rate_hz") if isinstance(node, dict) else 0.0,
            "rssi_dbm": node.get("rssi_dbm") if isinstance(node, dict) else None,
            "distance_m": None,
            "distance_reason": "meter distance is unavailable until physical calibration; RSSI/CSI are not reliable distance sensors here",
            "source_count": node.get("source_count", 0) if isinstance(node, dict) else 0,
            "source_hosts": node.get("source_hosts", []) if isinstance(node, dict) else [],
            "last_source": node.get("last_source") if isinstance(node, dict) else None,
            "possible_duplicate_node_id": bool(node.get("possible_duplicate_node_id")) if isinstance(node, dict) else False,
        }
    return statuses


def build_measurements(
    nodes: dict[str, dict[str, object]],
    node_status: dict[str, dict[str, object]],
) -> dict[str, object]:
    per_node = {}
    for node_id, status in sorted(node_status.items()):
        feature = nodes.get(node_id, {})
        per_node[node_id] = {
            "fresh": status["fresh"],
            "rate_hz": status["rate_hz"],
            "rssi_dbm": status["rssi_dbm"],
            "rssi_quality": rssi_quality(status["rssi_dbm"]),
            "presence_score": feature.get("presence_score") if feature else 0.0,
            "motion_energy": feature.get("motion_energy") if feature else 0.0,
            "robust_deviation": feature.get("robust_deviation") if feature else 0.0,
            "source_count": status.get("source_count", 0),
            "source_hosts": status.get("source_hosts", []),
            "last_source": status.get("last_source"),
            "possible_duplicate_node_id": status.get("possible_duplicate_node_id", False),
            "units": {
                "rssi": "dBm",
                "rate": "packets/second",
                "scores": "normalized CSI heuristic units",
                "distance": "not calibrated; meters unavailable",
            },
        }
    return {
        "per_node": per_node,
        "distance_model": "not_calibrated",
        "distance_note": "The current hardware can report CSI/RSSI activity per node, but cannot honestly estimate person-to-node distance in meters without calibration captures.",
    }


def rssi_quality(rssi_dbm: object) -> str:
    if not isinstance(rssi_dbm, (float, int)):
        return "UNKNOWN"
    if rssi_dbm >= -50:
        return "STRONG"
    if rssi_dbm >= -67:
        return "GOOD"
    if rssi_dbm >= -75:
        return "WEAK"
    return "VERY_WEAK"


def build_diagnostics(
    receiver_snapshot: dict[str, object],
    node_status: dict[str, dict[str, object]],
    stimulator: CsiStimulator | None = None,
) -> dict[str, object]:
    nodes = receiver_snapshot.get("nodes", {})
    received_any = isinstance(nodes, dict) and bool(nodes)
    expected_fresh = [
        node_id for node_id, status in sorted(node_status.items())
        if status.get("expected") and status.get("fresh")
    ]
    if received_any:
        message = "CSI UDP is reaching the dashboard."
    else:
        message = "No CSI UDP packets have reached the dashboard process."
    return {
        "udp_received_any": received_any,
        "fresh_expected_nodes": expected_fresh,
        "recommended_check": None if received_any else "allow inbound UDP/5005 from the hotspot subnet in the laptop firewall",
        "message": message,
        "stimulator": stimulator.snapshot() if stimulator else {"enabled": False},
    }
