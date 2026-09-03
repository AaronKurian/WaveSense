from __future__ import annotations

from dataclasses import dataclass
import math
import struct


MAGIC = 0x31534357
VERSION = 1
HEADER_LEN = 28
HEADER_STRUCT = struct.Struct("<IBBBBIQbbHHH")


class PacketError(ValueError):
    pass


@dataclass(frozen=True)
class CsiPacket:
    node_id: int
    channel: int
    sequence: int
    timestamp_us: int
    rssi_dbm: int
    noise_floor: int
    flags: int
    iq: tuple[tuple[int, int], ...]

    @property
    def subcarrier_count(self) -> int:
        return len(self.iq)

    @property
    def amplitudes(self) -> list[float]:
        return [math.hypot(i, q) for i, q in self.iq]


def parse_packet(data: bytes) -> CsiPacket:
    if len(data) < HEADER_LEN:
        raise PacketError(f"packet too short: {len(data)} bytes")

    (
        magic,
        version,
        header_len,
        node_id,
        channel,
        sequence,
        timestamp_us,
        rssi_dbm,
        noise_floor,
        csi_len,
        subcarrier_count,
        flags,
    ) = HEADER_STRUCT.unpack_from(data)

    if magic != MAGIC:
        raise PacketError(f"bad magic: 0x{magic:08x}")
    if version != VERSION:
        raise PacketError(f"unsupported version: {version}")
    if header_len != HEADER_LEN:
        raise PacketError(f"bad header length: {header_len}")
    if len(data) != header_len + csi_len:
        raise PacketError(f"bad length: got {len(data)}, expected {header_len + csi_len}")
    if csi_len % 2 != 0:
        raise PacketError("CSI payload length must be even")
    if subcarrier_count * 2 != csi_len:
        raise PacketError("subcarrier count does not match payload length")

    payload = data[header_len:]
    iq = tuple(
        (struct.unpack_from("<b", payload, idx)[0], struct.unpack_from("<b", payload, idx + 1)[0])
        for idx in range(0, len(payload), 2)
    )

    return CsiPacket(
        node_id=node_id,
        channel=channel,
        sequence=sequence,
        timestamp_us=timestamp_us,
        rssi_dbm=rssi_dbm,
        noise_floor=noise_floor,
        flags=flags,
        iq=iq,
    )


def build_packet(
    *,
    node_id: int,
    channel: int,
    sequence: int,
    timestamp_us: int,
    rssi_dbm: int,
    noise_floor: int = 0,
    iq: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    flags: int = 0,
) -> bytes:
    payload = bytearray()
    for i_val, q_val in iq:
        payload.extend(struct.pack("<bb", int(i_val), int(q_val)))
    csi_len = len(payload)
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        HEADER_LEN,
        node_id,
        channel,
        sequence,
        timestamp_us,
        rssi_dbm,
        noise_floor,
        csi_len,
        csi_len // 2,
        flags,
    )
    return header + payload
