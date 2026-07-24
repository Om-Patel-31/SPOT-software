"""Unified command-line launcher for every SPOT computer-vision workflow."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable

def _run_with_arguments(command: Callable[[], int | None], arguments: list[str]) -> int:
    """Run a legacy command whose parser reads ``sys.argv``."""
    previous_argv = sys.argv[:]
    try:
        sys.argv = [previous_argv[0], *arguments]
        result = command()
        return 0 if result is None else int(result)
    finally:
        sys.argv = previous_argv


def _interactive_choice() -> str:
    choices = {
        "1": "realtime",
        "2": "autotrain",
        "3": "dashboard",
        "4": "photo-train",
        "5": "calibrate",
    }
    print("SPOT computer-vision suite")
    print("1) Real-time recognition")
    print("2) Gemini-assisted auto-training")
    print("3) Accuracy dashboard")
    print("4) Train from a photo library")
    print("5) Calibrate recognition thresholds")
    return choices.get(input("Select a workflow [1-5]: ").strip(), "")


def _load_workflow(command: str) -> Callable[[], int | None]:
    """Import only the selected workflow and its optional dependencies."""
    module_names = {
        "realtime": "realtime",
        "autotrain": "autotrain",
        "dashboard": "dashboard",
        "photo-train": "photo_library_feedback_trainer",
        "calibrate": "calibrate_far_frr",
    }
    module = importlib.import_module(f".{module_names[command]}", package=__package__)
    return module.main


def main() -> int:
    parser = argparse.ArgumentParser(description="SPOT computer-vision suite")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("realtime", "autotrain", "dashboard", "photo-train", "calibrate"),
        help="Workflow to run; omit for the interactive menu.",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments passed to the selected workflow.")
    args = parser.parse_args()
    # Double-clicking the windowed executable should immediately start the
    # primary webcam workflow rather than wait for invisible console input.
    command = args.command or ("realtime" if getattr(sys, "frozen", False) else _interactive_choice())

    if command in {"realtime", "autotrain", "dashboard", "photo-train", "calibrate"}:
        return _run_with_arguments(_load_workflow(command), args.arguments)

    parser.error("Choose a workflow from the menu.")
    return 2
