from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from receiver.csi_receiver import CsiReceiver
from visualization.http_server import make_handler
from visualization.state import DashboardState, UI_VERSION
from visualization.stimulator import CsiStimulator

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wi-Fi CSI Sensing</title>
<style>
body { margin: 0; font-family: system-ui, sans-serif; background: #080b0f; color: #eee; }
header { padding: 12px 16px; border-bottom: 1px solid #23313a; display: flex; gap: 24px; align-items: center; }
main { display: grid; grid-template-columns: 1fr; gap: 12px; padding: 12px; }
canvas { width: 100%; background: #181818; border: 1px solid #333; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
#plots { display: grid; gap: 10px; }
#radar { min-height: 540px; background: #05090d; border-color: #244657; }
#side { display: none; }
.panel { border: 1px solid #263741; background: #10161b; padding: 12px; }
.label { color: #ffd24a; font-weight: 700; }
.muted { color: #aaa; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <strong>WaveSense</strong>
</header>
<main>
  <section id="plots">
    <canvas id="radar" width="900" height="560"></canvas>
    <canvas id="raw" width="900" height="190"></canvas>
    <canvas id="filtered" width="900" height="160"></canvas>
    <canvas id="deviation" width="900" height="160"></canvas>
    <canvas id="scores" width="900" height="130"></canvas>
  </section>
</main>
<script>
const colors = ["#53d769", "#4aa3ff", "#ff6b6b", "#ffd24a"];
function drawSeries(canvas, title, nodes, field) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#181818";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#333";
  ctx.beginPath();
  ctx.moveTo(0, canvas.height / 2);
  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();
  ctx.fillStyle = "#ddd";
  ctx.fillText(title, 12, 10);
  Object.entries(nodes).forEach(([id, node], idx) => {
    const values = node[field] || [];
    if (!values.length) return;
    const min = Math.min(...values), max = Math.max(...values);
    const span = Math.max(1e-6, max - min);
    ctx.strokeStyle = colors[idx % colors.length];
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = 20 + i * ((canvas.width - 40) / Math.max(1, values.length - 1));
      const y = canvas.height - 20 - ((v - min) / span) * (canvas.height - 45);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = colors[idx % colors.length];
    ctx.fillText(`Node ${id}`, 12 + idx * 70, 28);
  });
}
function drawOverlay(canvas, title, nodes, fields) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#181818";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ddd";
  ctx.fillText(title, 12, 12);
  Object.entries(nodes).forEach(([id, node], nodeIdx) => {
    const all = fields.flatMap(f => node[f.key] || []);
    if (!all.length) return;
    const min = Math.min(...all), max = Math.max(...all);
    const span = Math.max(1e-6, max - min);
    fields.forEach((field, fieldIdx) => {
      const values = node[field.key] || [];
      if (!values.length) return;
      ctx.strokeStyle = field.color;
      ctx.setLineDash(field.dash ? [6, 5] : []);
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = 20 + i * ((canvas.width - 40) / Math.max(1, values.length - 1));
        const y = canvas.height - 18 - ((v - min) / span) * (canvas.height - 42);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = field.color;
      ctx.fillText(`Node ${id} ${field.label}`, 12 + nodeIdx * 190 + fieldIdx * 82, 30);
    });
  });
}
function drawScores(canvas, data) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#181818";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ddd";
  ctx.fillText("Motion and presence scores", 12, 12);
  const nodes = Object.entries(data.nodes);
  nodes.forEach(([id, node], idx) => {
    const y = 38 + idx * 42;
    drawBar(ctx, 120, y, 260, 12, node.motion_energy || 0, "#4aa3ff", `Node ${id} motion`);
    drawBar(ctx, 470, y, 260, 12, node.presence_score || 0, "#ffd24a", `Node ${id} presence`);
  });
  drawBar(ctx, 120, canvas.height - 24, 260, 12, data.fused.motion_energy || 0, "#53d769", "fused motion");
  drawBar(ctx, 470, canvas.height - 24, 260, 12, data.fused.presence_score || 0, "#ff6b6b", "fused presence");
}
function drawBar(ctx, x, y, w, h, value, color, label) {
  const clamped = Math.max(0, Math.min(1, value));
  ctx.strokeStyle = "#444";
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = color;
  ctx.fillRect(x, y, w * clamped, h);
  ctx.fillStyle = "#ccc";
  ctx.fillText(`${label}: ${value.toFixed(3)}`, x + w + 12, y + 10);
}
function freshNodes(data) {
  return Object.entries(data.receiver.nodes || {}).filter(([, node]) => {
    const age = node.last_seen_age_s;
    return typeof age === "number" && age <= 2.5;
  });
}
function drawRadar(canvas, data) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#090d11";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const cx = canvas.width / 2;
  const cy = canvas.height / 2 - 4;
  const radius = Math.min(canvas.width, canvas.height) * 0.40;
  const fresh = freshNodes(data);
  const fused = data.fused || {};
  const geometry = data.geometry || {};

  const gradient = ctx.createRadialGradient(cx, cy, radius * 0.08, cx, cy, radius);
  gradient.addColorStop(0, "rgba(83, 215, 105, 0.08)");
  gradient.addColorStop(1, "rgba(74, 163, 255, 0.02)");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(120, 180, 220, 0.28)";
  ctx.lineWidth = 1;
  for (let ring = 1; ring <= 4; ring++) {
    const ringRadius = radius * ring / 4;
    ctx.beginPath();
    ctx.arc(cx, cy, ringRadius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = "rgba(180, 210, 225, 0.55)";
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillText(`${ring * 25}% rel`, cx + ringRadius + 6, cy - 4);
  }

  for (let spoke = 0; spoke < 16; spoke++) {
    const angle = spoke * Math.PI * 2 / 16;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(83, 215, 105, 0.45)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "#8fa3ad";
  ctx.fillText("Live sensing field", 12, 18);

  drawSensorMarkers(ctx, cx, cy, radius, data.node_status || {}, geometry);
  drawUnknownRegion(ctx, cx, cy, radius);
  drawNodeWarnings(ctx, data);

  if (!fresh.length) {
    ctx.fillStyle = "#ff6b6b";
    ctx.font = "18px system-ui, sans-serif";
    ctx.fillText("Waiting for sensor data", cx - 96, cy + 6);
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("Check sensor power and network connection.", cx - 134, cy + 30);
    ctx.font = "10px sans-serif";
    return;
  }

  if (!fused.presence) {
    ctx.fillStyle = "#888";
    ctx.fillText("No CSI presence detected", cx - 70, cy + 6);
    return;
  }

  const nodeCount = fresh.length;
  const hypotheses = Array.isArray(fused.hypotheses) ? fused.hypotheses : [];
  const pointHypotheses = hypotheses.filter(hypothesis => hypothesis.region !== "UNLOCALIZED");
  const unlocalizedHypotheses = hypotheses.filter(hypothesis => hypothesis.region === "UNLOCALIZED");
  if (pointHypotheses.length) {
    pointHypotheses.forEach((hypothesis, index) => drawHypothesis(ctx, cx, cy, radius, hypothesis, index));
  } else if (unlocalizedHypotheses.length) {
    drawUnlocalizedActivityZone(ctx, cx, cy, radius, unlocalizedHypotheses[0], nodeCount);
  } else {
    drawUnlocalizedActivityZone(ctx, cx, cy, radius, fused, nodeCount);
  }

  ctx.fillStyle = "#cdebd2";
  const localization = fused.localization || {};
  const label = pointHypotheses.length
    ? `CSI ACTIVITY POINTS: ${pointHypotheses.length}`
    : unlocalizedHypotheses.length
      ? "CSI ACTIVITY DETECTED - LOCATION UNKNOWN"
    : fused.presence
      ? "PRESENCE DETECTED - BUILDING STABLE CANDIDATE"
      : "NO PEOPLE DETECTED";
  ctx.font = "18px system-ui, sans-serif";
  ctx.fillText(label, 22, canvas.height - 48);
  ctx.fillStyle = "#8fa3ad";
  ctx.font = "12px system-ui, sans-serif";
  const detail = pointHypotheses.length > 1
      ? "Multiple points require simultaneous coarse-region evidence; not a people count."
    : pointHypotheses.length === 1
      ? (pointHypotheses[0]?.reason || "coarse CSI activity region")
      : nodeCount < 2
        ? "Only one fresh node is receiving CSI, so people cannot be separated or placed on the radar."
        : (unlocalizedHypotheses[0]?.reason || localization.reason || "CSI presence/activity is real; waiting for sustained region evidence.");
  ctx.fillText(detail, 22, canvas.height - 28);
  ctx.font = "10px sans-serif";
}
function drawUnlocalizedActivityZone(ctx, cx, cy, radius, source, nodeCount) {
  const confidence = Math.max(0, Math.min(1, source.confidence ?? source.presence_score ?? 0));
  const activity = Math.max(0, Math.min(1, source.activity ?? source.motion_energy ?? 0));
  const zoneRadius = radius * (0.20 + confidence * 0.08);
  const glowRadius = zoneRadius * (1.15 + activity * 0.22);
  const alpha = 0.12 + confidence * 0.16;

  const glow = ctx.createRadialGradient(cx, cy, 1, cx, cy, glowRadius);
  glow.addColorStop(0, `rgba(83, 215, 105, ${alpha})`);
  glow.addColorStop(0.58, `rgba(83, 215, 105, ${0.08 + activity * 0.12})`);
  glow.addColorStop(1, "rgba(83, 215, 105, 0)");
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(cx, cy, glowRadius, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(83, 215, 105, 0.70)";
  ctx.lineWidth = 3;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.arc(cx, cy, zoneRadius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#d8f2df";
  ctx.font = "13px system-ui, sans-serif";
  ctx.fillText("UNLOCALIZED CSI ACTIVITY", cx - 78, cy - zoneRadius - 16);
  ctx.fillStyle = "#8fa3ad";
  ctx.font = "12px system-ui, sans-serif";
  const reason = nodeCount < 2 ? "one fresh node only" : "insufficient separable spatial evidence";
  ctx.fillText(`${reason}  confidence ${confidence.toFixed(2)} activity ${activity.toFixed(2)}`, cx - 126, cy - zoneRadius - 1);
  ctx.font = "10px sans-serif";
}
function drawHypothesis(ctx, cx, cy, radius, hypothesis, index) {
  const confidence = Math.max(0, Math.min(1, hypothesis.confidence || 0));
  const activity = Math.max(0, Math.min(1, hypothesis.activity || 0));
  const dx = typeof hypothesis.display_x === "number" ? hypothesis.display_x : 0;
  const dy = typeof hypothesis.display_y === "number" ? hypothesis.display_y : 0;
  const px = cx + Math.max(-0.85, Math.min(0.85, dx)) * radius * 0.72;
  const py = cy - Math.max(-0.85, Math.min(0.85, dy)) * radius * 0.72;
  const coreRadius = 9 + confidence * 8 + activity * 7;
  const glowRadius = coreRadius * (2.8 + activity * 1.4);
  const palette = ["83, 215, 105", "255, 210, 74", "74, 163, 255"];
  const color = palette[index % palette.length];

  const glow = ctx.createRadialGradient(px, py, 1, px, py, glowRadius);
  glow.addColorStop(0, `rgba(${color}, ${0.76 + activity * 0.20})`);
  glow.addColorStop(0.42, `rgba(${color}, ${0.24 + confidence * 0.24})`);
  glow.addColorStop(1, `rgba(${color}, 0)`);
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(px, py, glowRadius, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = `rgba(${color}, 0.92)`;
  ctx.beginPath();
  ctx.arc(px, py, coreRadius, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = `rgba(${color}, 0.72)`;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(px, py, coreRadius + 4, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = `rgba(${color}, 0.30)`;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(px, py, coreRadius + 11, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "#d8f2df";
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText(`${hypothesis.region || "UNKNOWN"} ${confidence.toFixed(2)}`, px + 10, py - 8);
  ctx.font = "10px sans-serif";
}
function drawSensorMarkers(ctx, cx, cy, radius, nodeStatus, geometry) {
  const entries = Object.entries(geometry);
  const geometryValues = Object.values(geometry);
  const maxAbs = Math.max(1, ...geometryValues.flatMap(pos => [Math.abs(pos.x || 0), Math.abs(pos.y || 0)]));
  entries.forEach(([id, pos], idx) => {
    const status = nodeStatus[id] || {};
    const fallback = entries.length >= 2 ? [-1 + idx * 2, 0] : [0, 1];
    const gx = pos ? pos.x : fallback[0];
    const gy = pos ? pos.y : fallback[1];
    const x = cx + (gx / maxAbs) * radius;
    const y = cy - (gy / maxAbs) * radius;
    const fresh = status.fresh === true;
    const duplicate = status.possible_duplicate_node_id === true;
    ctx.fillStyle = fresh ? "#4aa3ff" : "#555";
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = duplicate ? "#ff6b6b" : (fresh ? "#b8dcff" : "#777");
    ctx.fillText(`Node ${id} (${gx}, ${gy})`, x - 42, y + 20);
    ctx.fillText(duplicate ? "DUPLICATE ID?" : (fresh ? "ONLINE" : "OFFLINE"), x - 34, y + 34);
  });
}
function drawNodeWarnings(ctx, data) {
  const statuses = Object.entries(data.node_status || {});
  const missing = statuses.filter(([, status]) => status.expected && !status.fresh).map(([id]) => id);
  const duplicate = statuses.filter(([, status]) => status.possible_duplicate_node_id).map(([id]) => id);
  const lines = [];
  if (missing.length) lines.push(`Offline sensor(s): ${missing.join(", ")}`);
  if (duplicate.length) lines.push(`Multiple sender IPs using node id: ${duplicate.join(", ")}`);
  if (!lines.length) return;
  ctx.fillStyle = "rgba(255, 107, 107, 0.92)";
  ctx.font = "13px system-ui, sans-serif";
  lines.forEach((line, idx) => ctx.fillText(line, 22, 52 + idx * 18));
  ctx.font = "10px sans-serif";
}
function drawUnknownRegion(ctx, cx, cy, radius) {
  ctx.strokeStyle = "rgba(255, 210, 74, 0.28)";
  ctx.setLineDash([8, 8]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 0.23, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
}
async function refresh() {
  const res = await fetch("/api/snapshot", { cache: "no-store" });
  const data = await res.json();
  drawRadar(document.getElementById("radar"), data);
  drawSeries(document.getElementById("raw"), "Raw CSI amplitude", data.nodes, "amplitude");
  drawOverlay(document.getElementById("filtered"), "Filtered amplitude and adaptive baseline", data.nodes, [
    { key: "filtered_amplitude", label: "filtered", color: "#4aa3ff" },
    { key: "baseline", label: "baseline", color: "#ffd24a", dash: true }
  ]);
  drawSeries(document.getElementById("deviation"), "Baseline deviation / robust normalized CSI", data.nodes, "smoothed");
  drawScores(document.getElementById("scores"), data);
}
setInterval(refresh, 250);
refresh();
</script>
</body>
</html>
""".replace("__UI_VERSION__", UI_VERSION)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Wi-Fi CSI dashboard")
    parser.add_argument("--udp-bind", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--http-bind", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8088)
    parser.add_argument("--stimulate-host", action="append", default=[], help="ESP32 IPv4 address to keep active with low-rate UDP traffic")
    parser.add_argument("--stimulate-rate-hz", type=float, default=30.0)
    args = parser.parse_args()

    receiver = CsiReceiver(args.udp_bind, args.udp_port)
    receiver.open()
    stimulator = CsiStimulator(receiver, args.stimulate_host, args.stimulate_rate_hz)
    stimulator.start()
    state = DashboardState(receiver, stimulator)
    thread = threading.Thread(target=state.ingest, daemon=True)
    thread.start()

    server = ThreadingHTTPServer((args.http_bind, args.http_port), make_handler(state, HTML))
    print(f"Dashboard: http://{args.http_bind}:{args.http_port}/")
    print(f"UDP CSI:   {args.udp_bind}:{args.udp_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        stimulator.stop()
        receiver.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
