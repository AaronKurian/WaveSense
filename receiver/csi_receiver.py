from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, field
import socket
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from receiver.packet import CsiPacket, PacketError, parse_packet


@dataclass
class NodeStats:
    packets: int = 0
    malformed: int = 0
    lost: int = 0
    last_sequence: int | None = None
    last_rssi: int = 0
    last_channel: int = 0
    arrival_times: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    last_packet: CsiPacket | None = None
    last_seen_monotonic: float | None = None
    source_hosts: dict[str, deque[float]] = field(default_factory=dict)
    last_source: tuple[str, int] | None = None

    @property
    def rate_hz(self) -> float:
        if len(self.arrival_times) < 2:
            return 0.0
        elapsed = self.arrival_times[-1] - self.arrival_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.arrival_times) - 1) / elapsed


class CsiReceiver:
    def __init__(self, bind: str = "0.0.0.0", port: int = 5005, timeout: float = 0.25):
        self.bind = bind
        self.port = port
        self.timeout = timeout
        self.nodes: dict[int, NodeStats] = defaultdict(NodeStats)
        self.total_malformed = 0
        self._socket: socket.socket | None = None
        self._running = False

    def open(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.bind, self.port))
        sock.settimeout(self.timeout)
        self._socket = sock
        self._running = True

    def close(self) -> None:
        self._running = False
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def receive_once(self) -> CsiPacket | None:
        if self._socket is None:
            raise RuntimeError("receiver is not open")
        try:
            data, addr = self._socket.recvfrom(4096)
        except socket.timeout:
            return None

        try:
            packet = parse_packet(data)
        except PacketError:
            self.total_malformed += 1
            return None

        self._record(packet, addr)
        return packet

    def run_forever(self, print_interval: float = 1.0) -> None:
        self.open()
        print(f"CSI receiver listening on {self.bind}:{self.port}")
        next_print = time.monotonic()
        try:
            while self._running:
                self.receive_once()
                now = time.monotonic()
                if now >= next_print:
                    print(self.summary_text(), end="\r", flush=True)
                    next_print = now + print_interval
        except KeyboardInterrupt:
            print()
        finally:
            self.close()

    def snapshot(self) -> dict[str, object]:
        nodes = {}
        now = time.monotonic()
        for node_id, stats in sorted(self.nodes.items()):
            sources = []
            for host, arrivals in sorted(stats.source_hosts.items()):
                sources.append({
                    "host": host,
                    "rate_hz": rate_from_arrivals(arrivals),
                    "last_seen_age_s": now - arrivals[-1] if arrivals else None,
                    "packets_window": len(arrivals),
                })
            nodes[str(node_id)] = {
                "packets": stats.packets,
                "rate_hz": stats.rate_hz,
                "lost": stats.lost,
                "rssi_dbm": stats.last_rssi,
                "channel": stats.last_channel,
                "subcarriers": stats.last_packet.subcarrier_count if stats.last_packet else 0,
                "last_seen_age_s": now - stats.last_seen_monotonic if stats.last_seen_monotonic is not None else None,
                "source_count": len(stats.source_hosts),
                "source_hosts": sources,
                "last_source": format_source(stats.last_source),
                "possible_duplicate_node_id": len(stats.source_hosts) > 1,
            }
        return {"nodes": nodes, "malformed": self.total_malformed}

    def summary_text(self) -> str:
        parts = ["CSI receiver | "]
        total_rate = 0.0
        for node_id, stats in sorted(self.nodes.items()):
            total_rate += stats.rate_hz
            parts.append(
                f"Node {node_id}: {stats.rate_hz:5.1f} pkt/s "
                f"RSSI {stats.last_rssi:4d} dBm lost {stats.lost} | "
            )
        parts.append(f"Total: {total_rate:5.1f} pkt/s malformed {self.total_malformed}")
        return "".join(parts)

    def _record(self, packet: CsiPacket, addr: tuple[str, int] | None = None) -> None:
        stats = self.nodes[packet.node_id]
        if stats.last_sequence is not None:
            expected = (stats.last_sequence + 1) & 0xFFFFFFFF
            if packet.sequence != expected:
                gap = (packet.sequence - expected) & 0xFFFFFFFF
                if gap < 0x80000000:
                    stats.lost += gap
        stats.last_sequence = packet.sequence
        stats.packets += 1
        stats.last_rssi = packet.rssi_dbm
        stats.last_channel = packet.channel
        now = time.monotonic()
        stats.arrival_times.append(now)
        stats.last_seen_monotonic = now
        stats.last_packet = packet
        if addr is not None:
            host, port = addr
            stats.last_source = (host, port)
            arrivals = stats.source_hosts.setdefault(host, deque(maxlen=200))
            arrivals.append(now)


def rate_from_arrivals(arrivals: deque[float]) -> float:
    if len(arrivals) < 2:
        return 0.0
    elapsed = arrivals[-1] - arrivals[0]
    if elapsed <= 0:
        return 0.0
    return (len(arrivals) - 1) / elapsed


def format_source(source: tuple[str, int] | None) -> str | None:
    if source is None:
        return None
    return f"{source[0]}:{source[1]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Receive Wi-Fi CSI UDP packets")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()

    CsiReceiver(args.bind, args.port).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
