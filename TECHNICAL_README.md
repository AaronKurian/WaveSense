# Wi-Fi CSI Sensing

Independent two-node Wi-Fi CSI sensing project for Seeed Studio XIAO ESP32-S3
boards and a Linux laptop receiver.

This project is intentionally separate from `~/Desktop/RuView`. RuView is only a
reference for ESP32 CSI configuration, compact UDP framing, parsing, and
signal-processing ideas.

## Architecture

```text
Phone hotspot
  |-- ESP32-S3 node 1
  |-- ESP32-S3 node 2
          |
          +-- UDP CSI packets --> laptop receiver
                                --> preprocessing
                                --> presence / motion
                                --> calibration metrics
                                --> live dashboard
```

The ESP32 nodes are not exchanging Wi-Fi frames with each other. Each node is
extracting CSI from frames on its own phone-hotspot link:

```text
Phone hotspot -> ESP32-S3 node 1
Phone hotspot -> ESP32-S3 node 2
```

That means the strongest sensing regions are the two phone-to-node RF links,
not automatically the area between the ESP32 boards.

## Hardware

- 2x Seeed Studio XIAO ESP32-S3
- 8 MB flash
- External antennas attached
- Linux Mint laptop connected to the same 2.4 GHz phone hotspot

## Packet Format

The firmware sends compact binary UDP packets documented in
`docs/packet-format.md`.

Payload bytes are the raw ESP-IDF CSI buffer: signed 8-bit interleaved I/Q
samples.

## Firmware

Firmware lives in `firmware/esp32-csi-node`.

Configuration is exposed through ESP-IDF Kconfig:

- `CONFIG_CSI_WIFI_SSID`
- `CONFIG_CSI_WIFI_PASSWORD`
- `CONFIG_CSI_TARGET_IP`
- `CONFIG_CSI_TARGET_PORT`
- `CONFIG_CSI_NODE_ID`

The channel follows the connected AP. The firmware does not hardcode a channel
for the phone hotspot.

## Receiver

Run the receiver from the project root:

```bash
python3 receiver/csi_receiver.py --bind 0.0.0.0 --port 5005
```

It prints a low-rate status summary instead of one line per packet.

## Visualization

Run the dashboard from the project root:

```bash
python3 visualization/dashboard.py --udp-bind 0.0.0.0 --udp-port 5005 --http-port 8088
```

Open:

```text
http://127.0.0.1:8088/
```

The dashboard uses only live received packets. It has no demo mode and no random
animation.

The dashboard always shows the expected Node 1 and Node 2 markers from the
configured geometry. If a node is missing or stale, it is marked offline instead
of silently disappearing.

## Signal Processing

Pipeline:

```text
raw I/Q
  -> amplitude
  -> robust clipping
  -> subcarrier median filtering
  -> fixed-bin feature vector
  -> slow adaptive baseline
  -> baseline removal
  -> robust temporal normalization
  -> motion / presence / calibration features
```

Current no-ML features:

- raw amplitude
- filtered amplitude
- adaptive baseline
- baseline deviation
- robust temporal deviation
- temporal motion energy
- RSSI
- two-node differential response imbalance
- correlation between node feature vectors during calibration capture

## Measurement Units

Current live measurements are:

- RSSI in `dBm`
- packet rate in packets/second
- CSI motion/presence/deviation in normalized heuristic units

The system does not currently output person-to-node distance in meters. RSSI and
CSI amplitude are affected by phone position, antenna orientation, walls,
people, multipath, and hotspot traffic, so converting them directly to meters
would be misleading. The snapshot exposes `distance_m = null` and explains that
meter distance is unavailable until calibration captures establish a usable
model.

## Presence

Presence is a heuristic score from measured CSI changes. It is not AI accuracy.

## Motion

Motion levels:

- `STATIONARY`
- `LOW`
- `MEDIUM`
- `HIGH`

The level is derived from temporal changes in processed CSI amplitudes.

## Two-Node Fusion

Each node is processed independently first. With one fresh node, the system can
report presence and activity but cannot estimate an XY location.

Default node geometry is a normalized placeholder axis:

```text
Node 1 = (-1.0, 0.0)
Node 2 = ( 1.0, 0.0)
```

These are documented placeholders for showing the two receivers on the link
activity display, not measured physical coordinates. The current default
configuration has `spatial_calibrated = False`, so the processing pipeline will
not emit spatial coordinates or radar person points.

The pipeline still computes relative CSI response metrics:

- temporal CSI deviation
- motion energy
- sustained presence score
- differential response between Node 1 and Node 2

It does not treat amplitude as distance and does not claim centimeter-level
positioning. Until calibration data proves a stable mapping, the localization
output remains:

```text
localized = false
region = UNCALIBRATED
reason = phone-to-node link geometry is not calibrated; not emitting spatial coordinates
```

The localization output shape is:

- `localized`
- `x`
- `y`
- `confidence`
- `region`
- `reason`

If a calibrated model is enabled later, region must remain intentionally coarse:

- `LEFT`
- `CENTER`
- `RIGHT`

On the default geometry, `LEFT` is the Node 1 side and `RIGHT` is the Node 2
side.

## Calibration Capture

Use the calibration recorder to collect real synchronized dashboard snapshots
for known physical positions:

```bash
python3 scripts/record_calibration.py --label empty_room --duration 10 --interval 0.5
python3 scripts/record_calibration.py --label person_near_node_1_link --duration 10 --interval 0.5
python3 scripts/record_calibration.py --label person_near_node_2_link --duration 10 --interval 0.5
```

The output is JSONL under `captures/`, which is gitignored because it contains
environment-specific CSI measurements. Each record includes:

- timestamp
- receiver/node packet rates and freshness
- RSSI
- raw per-subcarrier amplitude
- filtered amplitude
- adaptive baseline
- temporal CSI change
- baseline deviation
- Node 1 / Node 2 correlation
- differential response
- current fusion output

Keep the phone and both ESP32 boards fixed while collecting a calibration run.
If the phone hotspot is idle, CSI packet rates can fall below the useful range
because there are few Wi-Fi frames to observe. During calibration, keep a steady
real Wi-Fi traffic source active, for example pinging the hotspot gateway from
the laptop:

```bash
ping 10.140.196.178
```

This does not create synthetic CSI; it increases real Wi-Fi airtime on the
phone hotspot link so the ESP32 nodes can observe CSI consistently.

## CSI Activity Hypotheses

The fusion output also includes `hypotheses[]`. A hypothesis is a CSI-derived
activity signature, not a detected person count. The pipeline can emit up to
three tracked coarse-region candidates:

- `LEFT`
- `CENTER`
- `RIGHT`

Candidates require sustained two-node CSI evidence. A one-frame spike does not
create a point. Existing tracks use smoothed position/confidence, deletion
hysteresis, and a maximum per-update movement step so points fade or move
gradually instead of teleporting.

Each hypothesis includes:

- `id`
- `x` / `y` when calibrated geometry supports them, otherwise `null`
- `display_x` / `display_y` for coarse radar rendering
- `confidence`
- `activity`
- `region`
- `reason`
- `source`

Multiple radar points are shown only when multiple coarse regions have
independent sustained evidence. The dashboard does not duplicate one
localization result or turn a single changed subcarrier into fake people.

The snapshot also includes `hypothesis_debug`, with region scores, active bands,
persistence counters, thresholds, and rejection reasons. Use this when tuning
against real calibration captures.

## Radar Visualization

The dashboard shows a radar-style circular sensing view:

```text
CSI HEURISTIC - NOT TRUE POSE
```

With one fresh node, the radar shows:

```text
PRESENCE - LOCATION UNKNOWN
```

With two fresh nodes, the radar shows tracked CSI activity candidates when
sustained region evidence exists. If presence exists but no candidate has met
the persistence threshold yet, it shows:

```text
PRESENCE DETECTED - BUILDING STABLE CANDIDATE
```

Candidate positions are coarse visualization coordinates, not true physical XY.
The radar has no random animation. If packets stop arriving, it shows
`NO FRESH CSI DATA`.

## Build Firmware

After ESP-IDF is installed and exported:

```bash
cd ~/Desktop/wifi-csi-sensing/firmware/esp32-csi-node
idf.py set-target esp32s3
idf.py menuconfig
idf.py build
```

In `menuconfig`, set the hotspot SSID/password, laptop IP, target port, and node
ID.

## Flash Firmware

Only flash after the board has been detected and the firmware has built:

```bash
python3 scripts/detect_esp32.py
cd ~/Desktop/wifi-csi-sensing/firmware/esp32-csi-node
idf.py -p /dev/ttyACM0 flash monitor
```

Replace `/dev/ttyACM0` with the detected port.

## Test Procedure

1. Start the dashboard.
2. Flash/configure one node.
3. Confirm packet rate and RSSI.
4. Flash/configure the second node with a different node ID.
5. Test empty room, standing still, walking, arm movement, toward node 1, and
   toward node 2.
6. Record observations before tuning thresholds.

## Limitations

- No trained pose model is used.
- No pose accuracy is claimed.
- Presence, motion, region, and radar points are heuristic
  interpretations of real CSI measurements.
- Breathing estimation is intentionally deferred until raw signal quality and
  motion/presence behavior are stable.
