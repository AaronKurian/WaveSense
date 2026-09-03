import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from processing.pipeline import NodeFeatures
from receiver.csi_receiver import CsiReceiver
from receiver.packet import build_packet, parse_packet
from visualization.dashboard import DashboardState


def feature(node_id: int, presence_score: float = 0.8, motion_energy: float = 0.3) -> NodeFeatures:
    return NodeFeatures(
        node_id=node_id,
        rssi_dbm=-55,
        packet_rate_hz=25.0,
        amplitude=[],
        filtered_amplitude=[],
        baseline=[],
        corrected=[],
        normalized=[],
        smoothed=[],
        median_amplitude=0.0,
        mean_abs_deviation=0.0,
        temporal_variance=0.0,
        temporal_diff_energy=motion_energy,
        robust_deviation=5.0,
        variance=0.0,
        baseline_distance=5.0,
        motion_energy=motion_energy,
        presence_score=presence_score,
        presence=True,
        motion_level="HIGH",
    )


def record(receiver: CsiReceiver, node_id: int, sequence: int) -> None:
    packet = parse_packet(
        build_packet(
            node_id=node_id,
            channel=6,
            sequence=sequence,
            timestamp_us=sequence,
            rssi_dbm=-55,
            iq=[(3, 4), (5, 12)],
        )
    )
    receiver._record(packet)


def test_dashboard_snapshot_ignores_stale_node_for_fusion():
    receiver = CsiReceiver()
    state = DashboardState(receiver)
    record(receiver, 1, 10)
    record(receiver, 2, 10)
    receiver.nodes[2].last_seen_monotonic = time.monotonic() - state.fresh_age_limit_s - 1.0
    state.features[1] = feature(1)
    state.features[2] = feature(2)

    snapshot = state.snapshot()

    assert snapshot["fused"]["presence"] is True
    assert snapshot["fused"]["localization"]["localized"] is False
    assert snapshot["fused"]["localization"]["region"] == "UNLOCALIZED"
    assert "single node" in snapshot["fused"]["localization"]["reason"]


def test_dashboard_snapshot_localizes_when_both_nodes_are_fresh():
    receiver = CsiReceiver()
    state = DashboardState(receiver)
    state.config.spatial_calibrated = True
    record(receiver, 1, 10)
    record(receiver, 2, 10)
    state.features[1] = feature(1, presence_score=0.25, motion_energy=0.05)
    state.features[2] = feature(2, presence_score=0.9, motion_energy=0.5)

    snapshot = state.snapshot()

    assert snapshot["fused"]["localization"]["localized"] is True
    assert snapshot["fused"]["localization"]["region"] == "RIGHT"
    assert snapshot["fused"]["localization"]["x"] > 0.0


def test_dashboard_default_two_node_snapshot_requires_calibration():
    receiver = CsiReceiver()
    state = DashboardState(receiver)
    record(receiver, 1, 10)
    record(receiver, 2, 10)
    state.features[1] = feature(1, presence_score=0.8, motion_energy=0.3)
    state.features[2] = feature(2, presence_score=0.9, motion_energy=0.4)

    snapshot = state.snapshot()

    assert snapshot["fused"]["presence"] is True
    assert snapshot["fused"]["localization"]["localized"] is False
    assert snapshot["fused"]["localization"]["region"] == "UNCALIBRATED"
    assert snapshot["fused"]["hypotheses"] == []


def test_dashboard_reports_expected_offline_node_and_measurement_units():
    receiver = CsiReceiver()
    state = DashboardState(receiver)
    record(receiver, 2, 10)
    state.features[2] = feature(2, presence_score=0.8, motion_energy=0.3)

    snapshot = state.snapshot()

    assert snapshot["node_status"]["1"]["expected"] is True
    assert snapshot["node_status"]["1"]["fresh"] is False
    assert snapshot["node_status"]["1"]["distance_m"] is None
    assert snapshot["node_status"]["2"]["fresh"] is True
    assert snapshot["measurements"]["per_node"]["2"]["units"]["rssi"] == "dBm"
    assert snapshot["measurements"]["distance_model"] == "not_calibrated"


def test_dashboard_reports_duplicate_sender_hosts_for_same_node_id():
    receiver = CsiReceiver()
    state = DashboardState(receiver)
    first = parse_packet(
        build_packet(
            node_id=2,
            channel=6,
            sequence=10,
            timestamp_us=10,
            rssi_dbm=-55,
            iq=[(3, 4), (5, 12)],
        )
    )
    second = parse_packet(
        build_packet(
            node_id=2,
            channel=6,
            sequence=11,
            timestamp_us=11,
            rssi_dbm=-56,
            iq=[(3, 4), (5, 12)],
        )
    )
    receiver._record(first, ("192.168.44.10", 40000))
    receiver._record(second, ("192.168.44.11", 40000))
    state.features[2] = feature(2)

    snapshot = state.snapshot()

    assert snapshot["node_status"]["2"]["possible_duplicate_node_id"] is True
    assert snapshot["measurements"]["per_node"]["2"]["source_count"] == 2
    assert {source["host"] for source in snapshot["node_status"]["2"]["source_hosts"]} == {
        "192.168.44.10",
        "192.168.44.11",
    }
