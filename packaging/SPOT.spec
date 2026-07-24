# -*- mode: python ; coding: utf-8 -*-
"""Optional reproducible PyInstaller specification for SPOT."""

from pathlib import Path


PROJECT_ROOT = Path(SPECPATH).resolve().parent

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=[],
    datas=[
        (str(PROJECT_ROOT / "data" / "models"), "models"),
        (str(PROJECT_ROOT / "README.md"), "."),
    ],
    hiddenimports=[
        "spot.realtime",
        "spot.autotrain",
        "spot.dashboard",
        "spot.photo_library_feedback_trainer",
        "spot.calibrate_far_frr",
        "spot.gemini_auto_trainer",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="SPOT",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
