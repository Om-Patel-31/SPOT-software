"""
Calibrate face recognition thresholds using FAR/FRR from labeled embeddings.

Input formats:
1) Face identities file (default):
   models/face_identities.json
   Each entry contains {id, label, templates, threshold}

2) Generic labeled dataset JSON:
   [
     {"label": "alice", "embeddings": [[...], [...]]},
     {"label": "bob", "templates": [[...], [...]]}
   ]

Usage examples:
    python calibrate_far_frr.py
    python calibrate_far_frr.py --dataset models/face_identities.json --output models/threshold_calibration.json
    python calibrate_far_frr.py --dataset my_embeddings.json --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def normalize_embedding(vec: Sequence[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n > 1e-6:
        arr = arr / n
    return arr


def extract_labeled_embeddings(raw: Sequence[Dict]) -> Dict[str, List[np.ndarray]]:
    labeled: Dict[str, List[np.ndarray]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue

        label = str(item.get("label", "")).strip()
        if not label:
            # Skip unlabeled groups for calibration.
            continue

        vectors: List[Sequence[float]] = []
        if isinstance(item.get("templates"), list) and item["templates"]:
            vectors = item["templates"]
        elif isinstance(item.get("embeddings"), list) and item["embeddings"]:
            vectors = item["embeddings"]
        elif isinstance(item.get("encodings"), list) and item["encodings"]:
            vectors = item["encodings"]
        elif isinstance(item.get("encoding"), list) and item["encoding"]:
            vectors = [item["encoding"]]

        for v in vectors:
            if not isinstance(v, list) or not v:
                continue
            emb = normalize_embedding(v)
            if emb.size == 0:
                continue
            labeled.setdefault(label, []).append(emb)

    return labeled


def pairwise_distances(labeled: Dict[str, List[np.ndarray]]) -> Tuple[List[float], List[float]]:
    genuine: List[float] = []
    impostor: List[float] = []

    labels = list(labeled.keys())

    # Genuine pairs: same label.
    for label in labels:
        arrs = labeled[label]
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                if arrs[i].shape != arrs[j].shape:
                    continue
                d = float(1.0 - np.dot(arrs[i], arrs[j]))
                genuine.append(d)

    # Impostor pairs: different labels.
    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            aa = labeled[la]
            bb = labeled[lb]
            for va in aa:
                for vb in bb:
                    if va.shape != vb.shape:
                        continue
                    d = float(1.0 - np.dot(va, vb))
                    impostor.append(d)

    return genuine, impostor


def metrics_at_threshold(genuine: Sequence[float], impostor: Sequence[float], threshold: float) -> Dict[str, float]:
    g_total = max(1, len(genuine))
    i_total = max(1, len(impostor))

    false_rejects = sum(1 for d in genuine if d >= threshold)
    false_accepts = sum(1 for d in impostor if d < threshold)

    frr = float(false_rejects / g_total)
    far = float(false_accepts / i_total)
    return {
        "threshold": float(threshold),
        "far": far,
        "frr": frr,
        "gap": float(abs(far - frr)),
    }


def compute_eer_threshold(genuine: Sequence[float], impostor: Sequence[float]) -> Dict[str, float]:
    if not genuine or not impostor:
        return {
            "threshold": 0.22,
            "far": 1.0 if impostor else 0.0,
            "frr": 1.0 if genuine else 0.0,
            "gap": 1.0,
        }

    lo = float(max(0.0, min(min(genuine), min(impostor)) - 0.05))
    hi = float(min(1.0, max(max(genuine), max(impostor)) + 0.05))

    best = None
    for t in np.linspace(lo, hi, num=300):
        m = metrics_at_threshold(genuine, impostor, float(t))
        if best is None or m["gap"] < best["gap"]:
            best = m

    assert best is not None
    return best


def per_label_thresholds(labeled: Dict[str, List[np.ndarray]], default_threshold: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for label, arrs in labeled.items():
        if len(arrs) < 2:
            out[label] = float(default_threshold)
            continue

        intra = []
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                if arrs[i].shape != arrs[j].shape:
                    continue
                intra.append(float(1.0 - np.dot(arrs[i], arrs[j])))

        if not intra:
            out[label] = float(default_threshold)
            continue

        mean_d = float(np.mean(intra))
        std_d = float(np.std(intra))
        # Same heuristic used in runtime, but bounded.
        th = max(default_threshold, mean_d + 2.5 * std_d)
        out[label] = float(min(max(th, 0.16), 0.45))

    return out


def apply_thresholds_to_identities(path: Path, label_thresholds: Dict[str, float]) -> int:
    if not path.exists():
        return 0

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0

    if not isinstance(raw, list):
        return 0

    changed = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        if label not in label_thresholds:
            continue
        new_t = float(label_thresholds[label])
        old_t = float(item.get("threshold", new_t))
        if abs(old_t - new_t) > 1e-6:
            item["threshold"] = new_t
            changed += 1

    if changed > 0:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate FAR/FRR thresholds from labeled face embeddings.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("models/face_identities.json"),
        help="Path to identities/embeddings JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/threshold_calibration.json"),
        help="Path to write calibration summary JSON.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply per-label thresholds to models/face_identities.json when labels match.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        return 1

    try:
        raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Failed to read dataset: {exc}")
        return 1

    if not isinstance(raw, list):
        print("Dataset JSON must be a list of labeled embedding objects.")
        return 1

    labeled = extract_labeled_embeddings(raw)
    if not labeled:
        print("No labeled embeddings found. Add labels and templates/embeddings first.")
        return 1

    genuine, impostor = pairwise_distances(labeled)
    eer = compute_eer_threshold(genuine, impostor)
    label_thresholds = per_label_thresholds(labeled, default_threshold=float(eer["threshold"]))

    summary = {
        "dataset": str(args.dataset),
        "labels": sorted(labeled.keys()),
        "num_labels": len(labeled),
        "num_genuine_pairs": len(genuine),
        "num_impostor_pairs": len(impostor),
        "eer_threshold": float(eer["threshold"]),
        "eer_far": float(eer["far"]),
        "eer_frr": float(eer["frr"]),
        "recommended_label_thresholds": label_thresholds,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Calibration summary:")
    print(json.dumps(summary, indent=2))

    if args.apply:
        identities_path = Path("models/face_identities.json")
        changed = apply_thresholds_to_identities(identities_path, label_thresholds)
        print(f"Applied thresholds to {changed} identities in {identities_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
