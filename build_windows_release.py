import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist" / "spot_windows"
BUILD_DIR = ROOT / "build"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n> " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd or ROOT))


def main() -> int:
    if shutil.which("pyinstaller") is None:
        print("PyInstaller is not installed. Installing it now...")
        run([sys.executable, "-m", "pip", "install", "pyinstaller"])

    spec_path = ROOT / "spot_windows.spec"
    if spec_path.exists():
        spec_path.unlink()

    run([
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--name",
        "SPOT",
        "--add-data",
        "models;models",
        "--add-data",
        "README.md;.",
        "triangulated_face_realtime.py",
    ], cwd=ROOT)

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    exe_path = ROOT / "dist" / "SPOT.exe"
    if not exe_path.exists():
        raise SystemExit("Expected build artifact was not created: " + str(exe_path))

    shutil.move(str(exe_path), str(DIST_DIR / "SPOT.exe"))
    shutil.copytree(ROOT / "models", DIST_DIR / "models", dirs_exist_ok=True)
    shutil.copy2(ROOT / "README.md", DIST_DIR / "README.md")

    print(f"\nWindows build ready: {DIST_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
