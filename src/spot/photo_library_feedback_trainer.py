"""
Interactive photo-library feedback trainer for face identities.

What it does:
- Lets you choose a photo library folder (or pass --library).
- Scans images and detects faces.
- Shows each detected face with predicted identity/group.
- Collects feedback (right/wrong/unknown/skip/quit).
- Updates the shared face_identities.json store with new templates from feedback.
- Recalibrates per-label thresholds from updated templates.

Usage:
    python main.py photo-train
    python main.py photo-train --library "C:/Users/om31d/Downloads/Photos-3-001"
    python main.py photo-train --identities data/models/face_identities.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np

from .paths import model_path

try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit("mediapipe is required. Install with: pip install mediapipe") from exc

try:
    import insightface
except ImportError:
    insightface = None


Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]

DESCRIPTOR_LANDMARKS = [1, 33, 61, 78, 133, 152, 199, 234, 263, 291, 308, 362, 454]
DESCRIPTOR_DISTANCE_PAIRS = [
    (33, 263),
    (1, 152),
    (61, 291),
    (78, 308),
    (133, 362),
    (234, 454),
    (61, 152),
    (291, 152),
]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def normalize_embedding(vec: Sequence[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n > 1e-6:
        arr = arr / n
    return arr


def load_identities(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(raw, list):
        return []

    identities: List[Dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        templates = item.get("templates")
        if not isinstance(templates, list):
            templates = []

        identities.append(
            {
                "id": int(item.get("id", 0) or 0),
                "version": int(item.get("version", 1) or 1),
                "label": str(item.get("label", "")).strip(),
                "templates": [t for t in templates if isinstance(t, list) and t],
                "threshold": float(item.get("threshold", 0.22)),
                "pose_counts": item.get("pose_counts", {}),
            }
        )

    # Ensure identity IDs are populated.
    next_id = 1
    for ident in identities:
        if ident["id"] <= 0:
            ident["id"] = next_id
        next_id = max(next_id, int(ident["id"]) + 1)

    return identities


def save_identities(path: Path, identities: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identities, indent=2), encoding="utf-8")


def next_identity_id(identities: Sequence[Dict]) -> int:
    ids = [int(i.get("id", 0)) for i in identities if isinstance(i, dict)]
    return (max(ids) + 1) if ids else 1


def list_image_files(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    files.sort()
    return files


def ensure_face_landmarker_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Face Landmarker model to: {model_path}")
    urlretrieve(MODEL_URL, str(model_path))
    return model_path


def create_landmark_detector():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=8,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        def detect_with_solutions(rgb_frame: np.ndarray):
            result = mesh.process(rgb_frame)
            return result.multi_face_landmarks if result and result.multi_face_landmarks else []

        return detect_with_solutions, mesh.close

    if hasattr(mp, "tasks"):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_file = ensure_face_landmarker_model(model_path("face_landmarker.task"))

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_file)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=8,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)

        def detect_with_tasks(rgb_frame: np.ndarray):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)
            return result.face_landmarks if result and result.face_landmarks else []

        return detect_with_tasks, landmarker.close

    raise SystemExit("Installed mediapipe package does not provide supported face landmark APIs.")


def build_embedder():
    if insightface is None:
        return None
    try:
        app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception:
        return None


def clamp_point(pt: Point, w: int, h: int) -> Point:
    return (min(max(pt[0], 0), w - 1), min(max(pt[1], 0), h - 1))


def extract_face_points(face_landmarks, w: int, h: int) -> List[Point]:
    points: List[Point] = []
    landmarks = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append(clamp_point((x, y), w, h))
    return points


def compute_bbox(points: Sequence[Point], w: int, h: int, margin: int = 8) -> BBox:
    if not points:
        return (0, 0, 0, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(min(xs) - margin, 0)
    y1 = max(min(ys) - margin, 0)
    x2 = min(max(xs) + margin, w - 1)
    y2 = min(max(ys) + margin, h - 1)
    return (x1, y1, x2, y2)


def compute_landmark_descriptor(points: Sequence[Point]) -> List[float]:
    if not points:
        return []

    max_idx = max(DESCRIPTOR_LANDMARKS)
    if len(points) <= max_idx:
        arr = np.array(points, dtype=np.float32)
        arr -= arr.mean(axis=0)
        scale = float(np.max(np.linalg.norm(arr, axis=1)))
        if scale > 1e-6:
            arr /= scale
        vec = arr.flatten()
        n = float(np.linalg.norm(vec))
        if n > 1e-6:
            vec = vec / n
        return vec.tolist()

    selected = np.array([points[i] for i in DESCRIPTOR_LANDMARKS], dtype=np.float32)
    idx_map = {lm_idx: i for i, lm_idx in enumerate(DESCRIPTOR_LANDMARKS)}

    left_eye = selected[idx_map[33]]
    right_eye = selected[idx_map[263]]
    eye_center = (left_eye + right_eye) * 0.5
    selected -= eye_center

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.arctan2(dy, dx)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    x = selected[:, 0].copy()
    y = selected[:, 1].copy()
    selected[:, 0] = x * cos_a - y * sin_a
    selected[:, 1] = x * sin_a + y * cos_a

    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist <= 1e-6:
        eye_dist = 1.0
    selected /= eye_dist

    coords = selected.flatten()
    distances = []
    for a, b in DESCRIPTOR_DISTANCE_PAIRS:
        pa = selected[idx_map[a]]
        pb = selected[idx_map[b]]
        distances.append(float(np.linalg.norm(pa - pb)))

    vec = np.concatenate([coords, np.array(distances, dtype=np.float32)], axis=0)
    n = float(np.linalg.norm(vec))
    if n > 1e-6:
        vec = vec / n
    return vec.tolist()


def compute_embedding(
    frame_bgr: np.ndarray,
    points: Sequence[Point],
    bbox: BBox,
    face_embedder_app,
) -> List[float]:
    if face_embedder_app is not None and points:
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            crop = frame_bgr[y1:y2, x1:x2]
            try:
                detected = face_embedder_app.get(crop)
                if detected:
                    best_face = max(
                        detected,
                        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
                    )
                    emb = getattr(best_face, "normed_embedding", None)
                    if emb is None:
                        emb = getattr(best_face, "embedding", None)
                    if emb is not None:
                        vec = normalize_embedding(emb)
                        if vec.size > 0:
                            return vec.tolist()
            except Exception:
                pass

    # Fallback descriptor from landmarks.
    return compute_landmark_descriptor(points)


def person_samples(identity: Dict) -> List[np.ndarray]:
    templates = identity.get("templates", [])
    if not isinstance(templates, list):
        return []
    out = []
    for t in templates:
        if not isinstance(t, list) or not t:
            continue
        arr = normalize_embedding(t)
        if arr.size > 0:
            out.append(arr)
    return out


def match_descriptor(
    desc: List[float],
    identities: List[Dict],
    default_threshold: float = 0.22,
    min_second_best_margin: float = 0.035,
):
    if not desc or not identities:
        return None, None, None

    d = normalize_embedding(desc)
    if d.size == 0:
        return None, None, None

    candidates = []
    for i, ident in enumerate(identities):
        samples = person_samples(ident)
        if not samples:
            continue

        best = float("inf")
        for s in samples:
            if s.shape != d.shape:
                continue
            dist = float(1.0 - np.dot(d, s))
            if dist < best:
                best = dist

        if best < float("inf"):
            candidates.append(
                {
                    "idx": i,
                    "dist": best,
                    "threshold": float(ident.get("threshold", default_threshold)),
                }
            )

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: x["dist"])
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    second_dist = float(second["dist"]) if second else None
    margin_ok = True
    if second_dist is not None:
        margin_ok = (second_dist - float(best["dist"])) >= min_second_best_margin

    if float(best["dist"]) < float(best["threshold"]) and margin_ok:
        return int(best["idx"]), float(best["dist"]), second_dist

    return None, float(best["dist"]), second_dist


def add_template(identity: Dict, embedding: Sequence[float], max_templates: int = 40) -> bool:
    templates = identity.get("templates")
    if not isinstance(templates, list):
        templates = []
        identity["templates"] = templates

    emb = normalize_embedding(embedding).tolist()
    if not emb:
        return False

    should_add = True
    for t in templates[-8:]:
        if not isinstance(t, list) or not t:
            continue
        ta = normalize_embedding(t)
        ea = normalize_embedding(emb)
        if ta.shape == ea.shape:
            d = float(1.0 - np.dot(ta, ea))
            if d < 0.02:
                should_add = False
                break

    if should_add:
        templates.append(emb)
        if len(templates) > max_templates:
            del templates[:-max_templates]
        return True
    return False


def recalibrate_thresholds(identities: List[Dict]) -> None:
    for ident in identities:
        samples = person_samples(ident)
        if len(samples) < 2:
            continue

        dists = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                if samples[i].shape != samples[j].shape:
                    continue
                dists.append(float(1.0 - np.dot(samples[i], samples[j])))

        if not dists:
            continue

        mean_d = float(np.mean(dists))
        std_d = float(np.std(dists))
        calibrated = max(0.22, mean_d + 2.5 * std_d)
        ident["threshold"] = float(min(max(calibrated, 0.16), 0.45))


def pick_library_folder(arg_library: str | None) -> Path | None:
    if arg_library:
        p = Path(arg_library).expanduser().resolve()
        return p if p.exists() else None

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        picked = filedialog.askdirectory(title="Select photo library folder")
        root.destroy()
        if not picked:
            return None
        p = Path(picked).resolve()
        return p if p.exists() else None
    except Exception:
        return None


def get_or_create_identity_by_label(identities: List[Dict], label: str) -> Dict:
    label_norm = label.strip()
    for ident in identities:
        if str(ident.get("label", "")).strip().lower() == label_norm.lower() and label_norm:
            return ident

    new_ident = {
        "id": next_identity_id(identities),
        "version": 1,
        "label": label_norm,
        "templates": [],
        "threshold": 0.22,
        "pose_counts": {},
    }
    identities.append(new_ident)
    return new_ident


def draw_face_overlay(
    image: np.ndarray,
    bbox: BBox,
    points: Sequence[Point],
    title: str,
    subtitle: str,
) -> np.ndarray:
    vis = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 255), 2)

    for p in points[::8]:
        cv2.circle(vis, p, 1, (0, 255, 0), -1)

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 92), (20, 20, 20), -1)
    cv2.putText(vis, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, subtitle, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(
        vis,
        "Keys: y=correct, n=wrong(label), u=unknown group, s=skip, q=quit",
        (10, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return vis


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive library feedback trainer.")
    parser.add_argument("--library", type=str, default=None, help="Photo library folder path.")
    parser.add_argument(
        "--identities",
        type=Path,
        default=model_path("face_identities.json"),
        help="Path to identity clusters JSON.",
    )
    args = parser.parse_args()

    library = pick_library_folder(args.library)
    if library is None or not library.exists():
        print("No valid library selected.")
        return 1

    image_files = list_image_files(library)
    if not image_files:
        print(f"No images found in: {library}")
        return 1

    identities = load_identities(args.identities)
    detect_landmarks, close_detector = create_landmark_detector()
    embedder = build_embedder()

    print(f"Library: {library}")
    print(f"Images found: {len(image_files)}")
    print(f"Loaded identities: {len(identities)}")
    print(f"Embedding backend: {'insightface' if embedder is not None else 'landmark-geometry'}")

    reviewed = 0
    corrected = 0
    reinforced = 0
    unlabeled_created = 0

    try:
        for image_index, image_path in enumerate(image_files, start=1):
            frame_bgr = cv2.imread(str(image_path))
            if frame_bgr is None:
                continue

            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            faces = detect_landmarks(rgb)

            if not faces:
                continue

            for face_i, face_landmarks in enumerate(faces, start=1):
                points = extract_face_points(face_landmarks, w, h)
                bbox = compute_bbox(points, w, h)
                embedding = compute_embedding(frame_bgr, points, bbox, embedder)
                if not embedding:
                    continue

                matched_idx, matched_dist, second_dist = match_descriptor(embedding, identities)
                if matched_idx is not None:
                    ident = identities[matched_idx]
                    ident_label = str(ident.get("label", "")).strip() or f"Group {ident.get('id')}"
                    predicted = f"{ident_label}  d={matched_dist:.3f}"
                    if second_dist is not None:
                        predicted += f"  margin={second_dist - matched_dist:.3f}"
                else:
                    predicted = "Unknown"
                    if matched_dist is not None:
                        predicted += f"  best_d={matched_dist:.3f}"

                title = f"[{image_index}/{len(image_files)}] {image_path.name}  face {face_i}"
                subtitle = f"Prediction: {predicted}"
                vis = draw_face_overlay(frame_bgr, bbox, points, title, subtitle)
                cv2.imshow("Photo Library Feedback Trainer", vis)

                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    raise KeyboardInterrupt

                if key == ord("s"):
                    reviewed += 1
                    continue

                if key == ord("y"):
                    if matched_idx is not None:
                        if add_template(identities[matched_idx], embedding):
                            reinforced += 1
                    else:
                        label = input("Prediction was correct but unknown. Enter label (blank to keep unknown): ").strip()
                        if label:
                            target = get_or_create_identity_by_label(identities, label)
                            if add_template(target, embedding):
                                corrected += 1
                    reviewed += 1
                    continue

                if key == ord("n"):
                    label = input("Wrong prediction. Enter correct label: ").strip()
                    if label:
                        target = get_or_create_identity_by_label(identities, label)
                        if add_template(target, embedding):
                            corrected += 1
                    reviewed += 1
                    continue

                if key == ord("u"):
                    # Keep as unknown but retain a reusable unlabeled cluster.
                    unknown_identity = {
                        "id": next_identity_id(identities),
                        "version": 1,
                        "label": "",
                        "templates": [normalize_embedding(embedding).tolist()],
                        "threshold": 0.22,
                        "pose_counts": {},
                    }
                    identities.append(unknown_identity)
                    unlabeled_created += 1
                    reviewed += 1
                    continue

                # Any other key is treated as skip.
                reviewed += 1

    except KeyboardInterrupt:
        print("Stopping review.")
    finally:
        close_detector()
        cv2.destroyAllWindows()

    recalibrate_thresholds(identities)
    save_identities(args.identities, identities)

    summary = {
        "library": str(library),
        "images": len(image_files),
        "reviewed_faces": reviewed,
        "reinforced_matches": reinforced,
        "corrected_or_new_labels": corrected,
        "unlabeled_groups_created": unlabeled_created,
        "identities_total": len(identities),
        "saved_to": str(args.identities),
    }

    print("Review summary:")
    print(json.dumps(summary, indent=2))
    print("Tip: run python main.py calibrate --apply after larger review sessions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
