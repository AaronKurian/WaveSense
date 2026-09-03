from __future__ import annotations

import socket
import threading
import time

from receiver.csi_receiver import CsiReceiver


class CsiStimulator:
    def __init__(self, receiver: CsiReceiver, seed_hosts: list[str], rate_hz: float = 30.0, port: int = 9):
        self.receiver = receiver
        self.seed_hosts = set(seed_hosts)
        self.rate_hz = max(1.0, min(60.0, rate_hz))
        self.port = port
        self.sent = 0
        self.last_hosts: list[str] = []
        self._running = False
        self._socket: socket.socket | None = None

    def start(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        self._running = False
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def snapshot(self) -> dict[str, object]:
        return {
            "enabled": self._running,
            "rate_hz": self.rate_hz,
            "port": self.port,
            "last_hosts": self.last_hosts,
            "sent": self.sent,
        }

    def _run(self) -> None:
        interval = 1.0 / self.rate_hz
        sequence = 0
        while self._running:
            hosts = self._hosts()
            self.last_hosts = hosts
            payload = f"wifi-csi-stim {sequence}".encode("ascii")
            for host in hosts:
                try:
                    if self._socket is not None:
                        self._socket.sendto(payload, (host, self.port))
                        self.sent += 1
                except OSError:
                    pass
            sequence += 1
            time.sleep(interval)

    def _hosts(self) -> list[str]:
        hosts = set(self.seed_hosts)
        for stats in self.receiver.nodes.values():
            hosts.update(stats.source_hosts)
        return sorted(host for host in hosts if is_ipv4(host))


def is_ipv4(value: str) -> bool:
    try:
        socket.inet_aton(value)
    except OSError:
        return False
    return value.count(".") == 3
