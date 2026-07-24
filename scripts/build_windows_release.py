"""Build a shippable Windows release of the SPOT application."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
RELEASE_DIR = DIST_DIR / "SPOT-windows"
STANDALONE_DIR = DIST_DIR / "SPOT-standalone"


def run(command: list[str]) -> None:
    print("\n> " + " ".join(command))
    subprocess.check_call(command, cwd=ROOT)


def main() -> int:
    if importlib.util.find_spec("PyInstaller") is None:
        raise SystemExit(
            "PyInstaller is not installed. Run: "
            f'"{sys.executable}" -m pip install -r config/requirements.txt'
        )

    DIST_DIR.mkdir(exist_ok=True)
    separator = ";" if sys.platform.startswith("win") else ":"
    run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onefile",
            "--windowed",
            "--name",
            "SPOT",
            "--paths",
            str(ROOT / "src"),
            "--add-data",
            f"{ROOT / 'data' / 'models'}{separator}models",
            "--add-data",
            f"{ROOT / 'README.md'}{separator}.",
            "--hidden-import",
            "mediapipe.tasks.python.vision",
            "--hidden-import",
            "spot.realtime",
            "--hidden-import",
            "spot.autotrain",
            "--hidden-import",
            "spot.dashboard",
            "--hidden-import",
            "spot.photo_library_feedback_trainer",
            "--hidden-import",
            "spot.calibrate_far_frr",
            "--hidden-import",
            "spot.gemini_auto_trainer",
            "--workpath",
            str(ROOT / "build" / "pyinstaller"),
            "--specpath",
            str(ROOT / "build" / "spec"),
            "--distpath",
            str(DIST_DIR),
            str(ROOT / "main.py"),
        ]
    )

    executable = DIST_DIR / "SPOT.exe"
    if not executable.exists():
        raise SystemExit(f"Expected build artifact was not created: {executable}")

    if STANDALONE_DIR.exists():
        shutil.rmtree(STANDALONE_DIR)
    STANDALONE_DIR.mkdir(parents=True)
    shutil.copy2(executable, STANDALONE_DIR / executable.name)

    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    RELEASE_DIR.mkdir(parents=True)
    shutil.move(str(executable), str(RELEASE_DIR / executable.name))
    shutil.copytree(ROOT / "data" / "models", RELEASE_DIR / "models")
    shutil.copy2(ROOT / "README.md", RELEASE_DIR / "README.md")
    shutil.copy2(ROOT / "config" / ".env.example", RELEASE_DIR / ".env.example")
    shutil.make_archive(str(DIST_DIR / "SPOT-windows"), "zip", root_dir=RELEASE_DIR)

    print(f"\nWindows release ready: {RELEASE_DIR}")
    print(f"Single-file executable ready: {STANDALONE_DIR / executable.name}")
    print("Copy .env.example to .env beside SPOT.exe to enable Gemini features.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
