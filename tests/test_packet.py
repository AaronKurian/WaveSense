import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver.packet import HEADER_LEN, MAGIC, PacketError, build_packet, parse_packet
from receiver.csi_receiver import CsiReceiver


def test_parse_valid_packet():
    raw = build_packet(
        node_id=1,
        channel=6,
        sequence=42,
        timestamp_us=123456,
        rssi_dbm=-43,
        noise_floor=-96,
        iq=[(3, 4), (-5, 12)],
    )

    packet = parse_packet(raw)

    assert packet.node_id == 1
    assert packet.channel == 6
    assert packet.sequence == 42
    assert packet.rssi_dbm == -43
    assert packet.subcarrier_count == 2
    assert packet.amplitudes == [5.0, 13.0]


def test_reject_short_packet():
    try:
        parse_packet(b"\x00" * (HEADER_LEN - 1))
    except PacketError as exc:
        assert "too short" in str(exc)
    else:
        raise AssertionError("short packet should fail")


def test_reject_bad_magic():
    raw = bytearray(build_packet(
        node_id=1,
        channel=1,
        sequence=1,
        timestamp_us=1,
        rssi_dbm=-50,
        iq=[(1, 2)],
    ))
    raw[0:4] = (MAGIC + 1).to_bytes(4, "little")

    try:
        parse_packet(bytes(raw))
    except PacketError as exc:
        assert "bad magic" in str(exc)
    else:
        raise AssertionError("bad magic should fail")


def test_reject_truncated_payload():
    raw = build_packet(
        node_id=1,
        channel=1,
        sequence=1,
        timestamp_us=1,
        rssi_dbm=-50,
        iq=[(1, 2), (3, 4)],
    )

    try:
        parse_packet(raw[:-1])
    except PacketError as exc:
        assert "bad length" in str(exc)
    else:
        raise AssertionError("truncated payload should fail")


def test_receiver_treats_backward_sequence_as_reset():
    receiver = CsiReceiver()
    first = parse_packet(build_packet(node_id=2, channel=6, sequence=100, timestamp_us=1, rssi_dbm=-50, iq=[(1, 2)]))
    after_reset = parse_packet(build_packet(node_id=2, channel=6, sequence=5, timestamp_us=2, rssi_dbm=-51, iq=[(1, 2)]))

    receiver._record(first)
    receiver._record(after_reset)

    assert receiver.nodes[2].lost == 0


def test_receiver_snapshot_reports_packet_freshness():
    receiver = CsiReceiver()
    packet = parse_packet(build_packet(node_id=2, channel=6, sequence=1, timestamp_us=1, rssi_dbm=-50, iq=[(1, 2)]))

    receiver._record(packet)
    snapshot = receiver.snapshot()

    assert snapshot["nodes"]["2"]["last_seen_age_s"] >= 0.0


def test_receiver_snapshot_reports_sender_hosts_and_duplicate_node_id():
    receiver = CsiReceiver()
    first = parse_packet(build_packet(node_id=2, channel=6, sequence=1, timestamp_us=1, rssi_dbm=-50, iq=[(1, 2)]))
    second = parse_packet(build_packet(node_id=2, channel=6, sequence=2, timestamp_us=2, rssi_dbm=-51, iq=[(1, 2)]))

    receiver._record(first, ("192.168.44.10", 40000))
    receiver._record(second, ("192.168.44.11", 40000))
    snapshot = receiver.snapshot()

    node = snapshot["nodes"]["2"]
    assert node["source_count"] == 2
    assert node["possible_duplicate_node_id"] is True
    assert {source["host"] for source in node["source_hosts"]} == {"192.168.44.10", "192.168.44.11"}
