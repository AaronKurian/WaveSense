#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def candidate_ports() -> list[str]:
    ports = sorted(str(p) for p in Path("/dev").glob("ttyACM*"))
    ports += sorted(str(p) for p in Path("/dev").glob("ttyUSB*"))
    return ports


def run_esptool(port: str, command: str) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "esptool",
        "--chip",
        "esp32s3",
        "--port",
        port,
        command,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=15)
    return result.returncode == 0, result.stdout + result.stderr


def main() -> int:
    if shutil.which("python3") is None:
        print("python3 not found")
        return 1

    ports = candidate_ports()
    if not ports:
        print("No /dev/ttyACM* or /dev/ttyUSB* devices found.")
        return 1

    print("ESP32 non-destructive detection")
    print("--------------------------------")
    for port in ports:
        print(f"\nPort: {port}")
        ok_chip, chip_out = run_esptool(port, "chip-id")
        print(chip_out.strip())
        ok_flash, flash_out = run_esptool(port, "flash-id")
        print(flash_out.strip())
        if ok_chip and ok_flash:
            print(f"RESULT: {port} responded to chip_id and flash_id")
        else:
            print(f"RESULT: {port} did not complete ESP32-S3 checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
