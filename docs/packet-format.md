# CSI UDP Packet Format

Version: 1

All integer fields are little-endian.

## Header

Header length: 28 bytes.

| Offset | Size | Type | Field | Description |
|---:|---:|---|---|---|
| 0 | 4 | u32 | magic | `0x31534357` (`WCS1`) |
| 4 | 1 | u8 | version | `1` |
| 5 | 1 | u8 | header_len | `28` |
| 6 | 1 | u8 | node_id | Configured node ID |
| 7 | 1 | u8 | channel | Current Wi-Fi channel from ESP-IDF RX metadata |
| 8 | 4 | u32 | sequence | Monotonic per-node packet sequence |
| 12 | 8 | u64 | timestamp_us | `esp_timer_get_time()` on the node |
| 20 | 1 | i8 | rssi_dbm | ESP-IDF RX RSSI |
| 21 | 1 | i8 | noise_floor | ESP-IDF RX noise floor, if available |
| 22 | 2 | u16 | csi_len | Payload length in bytes |
| 24 | 2 | u16 | subcarrier_count | `csi_len / 2` for one antenna I/Q pairs |
| 26 | 2 | u16 | flags | Reserved for future use, currently `0` |

## Payload

The payload starts at byte 28 and is exactly `csi_len` bytes.

It is the raw ESP-IDF CSI buffer:

```text
i0, q0, i1, q1, i2, q2, ...
```

Each `i` and `q` value is a signed 8-bit integer.

Amplitude for subcarrier `k`:

```text
sqrt(i_k^2 + q_k^2)
```

Phase for subcarrier `k`:

```text
atan2(q_k, i_k)
```

## Validation Rules

Receiver must reject packets when:

- length is less than 28 bytes
- magic is not `0x31534357`
- version is not `1`
- `header_len` is not `28`
- packet length does not equal `header_len + csi_len`
- `csi_len` is odd
- `subcarrier_count * 2 != csi_len`

## Rationale

This project uses its own minimal packet format instead of copying the full
RuView server architecture. The design keeps the useful parts:

- compact binary UDP
- explicit magic/version
- node ID
- sequence tracking
- RSSI/channel metadata
- raw I/Q payload for local processing
