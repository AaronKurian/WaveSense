from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean
from urllib.request import urlopen


def finite_values(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    clean = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            clean.append(numeric)
    return clean


def vector_correlation(left: list[float], right: list[float]) -> float | None:
    count = min(len(left), len(right))
    if count < 3:
        return None
    left = left[:count]
    right = right[:count]
    left_mean = mean(left)
    right_mean = mean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_delta, right_delta))
    left_power = sum(a * a for a in left_delta)
    right_power = sum(b * b for b in right_delta)
    denom = math.sqrt(left_power * right_power)
    if denom <= 1e-12:
        return None
    return numerator / denom


def differential_response(nodes: dict[str, object]) -> dict[str, object]:
    node1 = nodes.get("1", {}) if isinstance(nodes.get("1"), dict) else {}
    node2 = nodes.get("2", {}) if isinstance(nodes.get("2"), dict) else {}
    if not node1 or not node2:
        return {"available": False, "reason": "requires fresh Node 1 and Node 2 features"}

    n1_motion = float(node1.get("motion_energy") or 0.0)
    n2_motion = float(node2.get("motion_energy") or 0.0)
    n1_presence = float(node1.get("presence_score") or 0.0)
    n2_presence = float(node2.get("presence_score") or 0.0)
    n1_deviation = float(node1.get("robust_deviation") or 0.0)
    n2_deviation = float(node2.get("robust_deviation") or 0.0)
    response1 = n1_presence * 0.45 + min(1.0, n1_deviation / 6.0) * 0.35 + n1_motion * 0.20
    response2 = n2_presence * 0.45 + min(1.0, n2_deviation / 6.0) * 0.35 + n2_motion * 0.20
    total = response1 + response2
    imbalance = (response2 - response1) / total if total > 1e-9 else 0.0

    return {
        "available": True,
        "node1_response": response1,
        "node2_response": response2,
        "imbalance": imbalance,
        "smoothed_correlation": vector_correlation(
            finite_values(node1.get("smoothed")),
            finite_values(node2.get("smoothed")),
        ),
        "amplitude_correlation": vector_correlation(
            finite_values(node1.get("amplitude")),
            finite_values(node2.get("amplitude")),
        ),
    }


def compact_node(node: dict[str, object]) -> dict[str, object]:
    return {
        "rate_hz": node.get("rate_hz"),
        "rssi_dbm": node.get("rssi_dbm"),
        "presence": node.get("presence"),
        "presence_score": node.get("presence_score"),
        "motion": node.get("motion_level"),
        "motion_energy": node.get("motion_energy"),
        "temporal_diff_energy": node.get("temporal_diff_energy"),
        "robust_deviation": node.get("robust_deviation"),
        "baseline_distance": node.get("baseline_distance"),
        "median_amplitude": node.get("median_amplitude"),
        "mean_abs_deviation": node.get("mean_abs_deviation"),
        "amplitude": node.get("amplitude"),
        "filtered_amplitude": node.get("filtered_amplitude"),
        "baseline": node.get("baseline"),
        "smoothed": node.get("smoothed"),
    }


def capture(url: str, label: str, duration_s: float, interval_s: float, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + duration_s
    count = 0
    with output.open("a", encoding="utf-8") as handle:
        while time.monotonic() < deadline:
            with urlopen(url, timeout=5) as response:
                snapshot = json.load(response)
            nodes = snapshot.get("nodes", {})
            record = {
                "label": label,
                "host_time_s": time.time(),
                "snapshot_time_s": snapshot.get("time"),
                "ui_version": snapshot.get("ui_version"),
                "geometry": snapshot.get("geometry"),
                "receiver": snapshot.get("receiver"),
                "nodes": {
                    node_id: compact_node(node)
                    for node_id, node in nodes.items()
                    if isinstance(node, dict)
                },
                "fusion": snapshot.get("fused"),
                "differential": differential_response(nodes if isinstance(nodes, dict) else {}),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            count += 1
            time.sleep(interval_s)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Record real Wi-Fi CSI calibration snapshots")
    parser.add_argument("--url", default="http://127.0.0.1:8088/api/snapshot")
    parser.add_argument("--label", required=True, help="Physical test label, for example empty_room or near_node_1")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("captures/calibration.jsonl"))
    args = parser.parse_args()

    count = capture(args.url, args.label, args.duration, args.interval, args.output)
    print(f"wrote {count} calibration records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
