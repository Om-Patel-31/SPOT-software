"""
Real-time face recognition with simple enrollment.

Features:
- Detects and recognizes faces from webcam
- Displays names above detected faces with bounding boxes
- Simple enrollment: press 'e' to enroll, space to capture, 's' to save

Controls:
  q: quit
  e: start enrollment
  space: capture frame during enrollment
  s: save enrollment
  c: cancel enrollment

Install requirements:
    pip install opencv-python mediapipe numpy

Run:
    python main.py realtime
"""

import json
import importlib
import os
import sys
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np

from .paths import application_root, environment_file, model_path

try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit(
        "mediapipe is not installed. Install with: pip install mediapipe"
    ) from exc

try:
    # Silence protobuf deprecation warnings from google packages
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
    # Prefer the newer google.genai package if available
    genai = importlib.import_module("google.genai")
except Exception:
    try:
        genai = importlib.import_module("google.generativeai")
    except Exception:
        genai = None

try:
    from PIL import Image
except Exception:
    Image = None


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Key landmarks for geometry-based face descriptor
DESCRIPTOR_LANDMARKS = [1, 33, 61, 78, 133, 152, 199, 234, 263, 291, 308, 362, 454]
DESCRIPTOR_DISTANCE_PAIRS = [
    (33, 263), (1, 152), (61, 291), (78, 308), (133, 362), (234, 454), (61, 152), (291, 152),
]


def load_environment_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    if not env_path.exists():
        return

    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        print(f"Warning: could not load {env_path.name}: {exc}")


load_environment_file(environment_file())


def get_app_root() -> Path:
    """Return the directory that should be used for bundled app assets."""
    return application_root()


def get_data_path(*parts: str) -> Path:
    """Resolve a path in SPOT's persistent model store."""
    if parts and parts[0] == "models":
        return model_path(*parts[1:])
    return get_app_root().joinpath(*parts)


def ensure_face_landmarker_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Face Landmarker model...")
    urlretrieve(MODEL_URL, str(model_path))
    return model_path


def create_landmark_detector():
    """Return detector callable and cleanup function."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = ensure_face_landmarker_model(get_data_path("models", "face_landmarker.task"))

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=4,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(rgb_frame: np.ndarray, timestamp_ms: int):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        return result.face_landmarks if result.face_landmarks else []

    return detect, landmarker.close


def extract_face_points(face_landmarks, width: int, height: int) -> List[Tuple[int, int]]:
    """Extract 2D face landmark points."""
    points = []
    landmarks = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        points.append((x, y))
    return points


def compute_face_centroid(points: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Compute center of face points."""
    if not points:
        return (0, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (int(np.mean(xs)), int(np.mean(ys)))


def compute_bbox(points: List[Tuple[int, int]], width: int, height: int, margin: int = 20) -> Tuple[int, int, int, int]:
    """Compute bounding box from points."""
    if not points:
        return (0, 0, 0, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(min(xs) - margin, 0)
    y1 = max(min(ys) - margin, 0)
    x2 = min(max(xs) + margin, width - 1)
    y2 = min(max(ys) + margin, height - 1)
    return (x1, y1, x2, y2)


def normalize_embedding(vec: List[float]) -> np.ndarray:
    """Normalize embedding vector."""
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 1e-6:
        arr = arr / norm
    return arr


def compute_landmark_descriptor(points: List[Tuple[int, int]]) -> List[float]:
    """Compute stable geometry descriptor from face landmarks."""
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
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec.tolist()

    # Use selected key landmarks
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
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        vec = vec / norm
    return vec.tolist()


def load_identities() -> List[Dict]:
    """Load face identity clusters from JSON."""
    identities_path = get_data_path("models", "face_identities.json")
    if not identities_path.exists():
        return []

    try:
        with open(identities_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading identities: {e}")
        return []


def save_identities(data: List[Dict]):
    """Save face identity clusters to JSON."""
    identities_path = get_data_path("models", "face_identities.json")
    identities_path.parent.mkdir(parents=True, exist_ok=True)
    with open(identities_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class GeminiAdvisor:
    """Optional Gemini helper for uncertain face matches."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        if genai is None or Image is None:
            raise RuntimeError("google-genai (or google.generativeai) and pillow are required for Gemini support")
        # configure API key for either client
        try:
            if hasattr(genai, "configure"):
                genai.configure(api_key=api_key)
        except Exception:
            pass
        self._model_name = model_name
        self._client = genai

    def suggest_label(self, frame_bgr: np.ndarray, identities: List[Dict]) -> Optional[Tuple[str, float]]:
        """Return (label, confidence) or ("UNKNOWN", score) or None on failure.

        Confidence is a heuristic when the SDK doesn't provide one (default 0.85).
        """
        if frame_bgr.size == 0:
            return None

        labels = [str(item.get("label", "")).strip() for item in identities if str(item.get("label", "")).strip()]
        labels_text = ", ".join(labels) if labels else "none"
        prompt = (
            "You are helping a local face recognition system choose the most likely label for a face crop. "
            "Choose exactly one label from the provided list, or UNKNOWN if none fit. "
            "Reply in JSON with keys: label (string) and confidence (number 0-1). Do not invent a new label.\n\n"
            f"Available labels: {labels_text}\n"
        )

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        try:
            # Try google.genai style first
            if hasattr(self._client, "TextGenerationModel") or hasattr(self._client, "generate"):
                # Best-effort: call a text-generation endpoint with prompt + image.
                # Different client versions have different APIs; attempt common ones.
                try:
                    # Newer google.genai: use client.generate with multimodal input
                    resp = None
                    if hasattr(self._client, "generate"):
                        resp = self._client.generate(model=self._model_name, input=prompt)
                        text = getattr(resp, "output", None) or getattr(resp, "text", None) or ""
                        if isinstance(text, list):
                            text = text[0]
                        text = str(text)
                    else:
                        # Fallback to older generativeai style
                        model = getattr(self._client, "GenerativeModel", None)
                        if model is not None:
                            instance = model(self._model_name)
                            out = instance.generate(prompt)
                            text = getattr(out, "text", "")
                        else:
                            text = ""
                except Exception:
                    text = ""

            else:
                text = ""

            if not text:
                # Fallback: ask legacy client to return a label (best-effort)
                # We'll return a default low-confidence UNKNOWN
                return ("UNKNOWN", 0.0)

            txt = text.strip()
            # parse JSON-like first line if present
            first = txt.splitlines()[0].strip()
            try:
                import json as _json

                parsed = _json.loads(first)
                label = str(parsed.get("label", "UNKNOWN")).strip()
                confidence = float(parsed.get("confidence", 0.0))
                return (label, confidence)
            except Exception:
                # If not JSON, treat as plain label with heuristic confidence
                candidate = first.strip().strip('"`')
                if not candidate:
                    return None
                if candidate.upper() == "UNKNOWN":
                    return ("UNKNOWN", 0.0)
                # find exact label match
                for label in labels:
                    if candidate.lower() == label.lower():
                        return (label, 0.90)
                # not in labels -> reject
                return ("UNKNOWN", 0.0)
        except Exception:
            return None


def best_face_match(embedding: List[float], identities: List[Dict]) -> Optional[Dict[str, object]]:
    """Return the best identity candidate even if it does not pass threshold."""
    if not embedding or not identities:
        return None

    embedding_arr = normalize_embedding(embedding)
    if embedding_arr.size == 0:
        return None

    best_idx = None
    best_dist = float("inf")
    best_person_threshold = 0.22

    for i, identity in enumerate(identities):
        templates = identity.get("templates", [])
        if not templates:
            continue

        person_best = float("inf")
        for template in templates:
            if not template:
                continue
            template_emb = normalize_embedding(template)
            if template_emb.shape != embedding_arr.shape:
                continue
            dist = float(1.0 - np.dot(embedding_arr, template_emb))
            if dist < person_best:
                person_best = dist

        if person_best < float("inf") and person_best < best_dist:
            best_dist = person_best
            best_idx = i
            best_person_threshold = float(identity.get("threshold", 0.22))

    if best_idx is None:
        return None

    return {
        "idx": best_idx,
        "dist": best_dist,
        "threshold": best_person_threshold,
        "label": str(identities[best_idx].get("label", f"Person {best_idx}")),
    }


def match_face(embedding: List[float], identities: List[Dict]) -> Optional[Tuple[int, float, str]]:
    """Match embedding to known identity. Returns (identity_idx, distance, label)."""
    candidate = best_face_match(embedding, identities)
    if candidate is None:
        return None

    if float(candidate["dist"]) < float(candidate["threshold"]):
        return int(candidate["idx"]), float(candidate["dist"]), str(candidate["label"])

    return None


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    detect_landmarks, close_detector = create_landmark_detector()
    identities = load_identities()
    gemini_advisor = None
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_api_key and genai is not None and Image is not None:
        try:
            gemini_advisor = GeminiAdvisor(gemini_api_key)
            print("Gemini is enabled and ready.")
        except Exception as exc:
            print(f"Gemini disabled: {exc}")
    else:
        if not gemini_api_key:
            print("Gemini is disabled: GEMINI_API_KEY is not set.")
        elif genai is None:
            print("Gemini is disabled: google-generativeai is not installed.")
        elif Image is None:
            print("Gemini is disabled: pillow is not installed.")
        else:
            print("Gemini is disabled: unknown startup condition.")

    print(f"\nLoaded {len(identities)} identities")
    print("\nControls:")
    print("  q: quit")
    print("  e: start enrollment")
    print("  space: capture frame during enrollment")
    print("  s: save enrollment")
    print("  c: cancel enrollment\n")

    # Enrollment state
    enrolling = False
    embedding_buffer: deque = deque(maxlen=8)
    pending_gemini_suggestion: Optional[Dict[str, object]] = None
    unknown_consecutive: Dict[int, int] = {}
    face_conf_history: Dict[int, deque] = {}
    # Auto-accept confidence read from .env (AUTO_ACCEPT_CONFIDENCE)
    try:
        auto_accept_conf = float(os.environ.get("AUTO_ACCEPT_CONFIDENCE", os.environ.get("AUTO_ACCEPT", "0.80")))
    except Exception:
        auto_accept_conf = 0.80
    auto_accept_flash: Optional[Dict[str, float]] = None

    # Face tracking
    face_tracks: Dict[int, deque] = {}  # face_id -> embeddings
    next_face_id = 1
    face_centroids: Dict[int, Tuple[int, int]] = {}
    face_bboxes: Dict[int, Tuple[int, int, int, int]] = {}
    face_labels: Dict[int, str] = {}
    face_match_stats: Dict[int, Dict[str, float]] = {}

    last_time = time.time()
    fps = 0.0

    try:
        cv2.namedWindow("Face Recognition")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame")
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int(time.time() * 1000)
            faces = detect_landmarks(rgb, timestamp_ms)

            out = frame.copy()

            # Process each detected face
            current_face_ids = []
            if faces:
                for face_landmarks in faces:
                    points = extract_face_points(face_landmarks, w, h)
                    centroid = compute_face_centroid(points)
                    bbox = compute_bbox(points, w, h)
                    embedding = compute_landmark_descriptor(points)

                    # Simple face tracking: match to nearest existing track
                    best_track_id = None
                    best_dist = float("inf")

                    for track_id, centroid_prev in face_centroids.items():
                        dist = ((centroid[0] - centroid_prev[0]) ** 2 + (centroid[1] - centroid_prev[1]) ** 2) ** 0.5
                        if dist < 100 and dist < best_dist:
                            best_dist = dist
                            best_track_id = track_id

                    if best_track_id is None:
                        best_track_id = next_face_id
                        next_face_id += 1
                        face_tracks[best_track_id] = deque(maxlen=8)

                    current_face_ids.append(best_track_id)

                    # Smooth embedding over frames
                    if embedding:
                        face_tracks[best_track_id].append(np.array(embedding, dtype=np.float32))
                        if len(face_tracks[best_track_id]) > 0:
                            stack = np.stack(list(face_tracks[best_track_id]), axis=0)
                            emb_mean = np.mean(stack, axis=0)
                            emb_norm = np.linalg.norm(emb_mean)
                            if emb_norm > 1e-6:
                                emb_mean = emb_mean / emb_norm
                            embedding = emb_mean.tolist()

                    # Match to known identity
                    candidate = best_face_match(embedding, identities)
                    match = match_face(embedding, identities)
                    if match:
                        idx, dist, label = match
                        face_labels[best_track_id] = label
                        face_match_stats[best_track_id] = {
                            "dist": float(dist),
                            "threshold": float(identities[idx].get("threshold", 0.22)),
                            "gemini": 0.0,
                        }
                        unknown_consecutive.pop(best_track_id, None)
                    else:
                        face_labels[best_track_id] = "Unknown"
                        face_match_stats[best_track_id] = {
                            "dist": float(candidate["dist"]) if candidate is not None else 1.0,
                            "threshold": float(candidate["threshold"]) if candidate is not None else 0.22,
                            "gemini": 1.0 if gemini_advisor is not None else 0.0,
                        }
                        unknown_consecutive[best_track_id] = unknown_consecutive.get(best_track_id, 0) + 1
                        if (
                            gemini_advisor is not None
                            and pending_gemini_suggestion is None
                            and unknown_consecutive[best_track_id] >= 3
                        ):
                            x1, y1, x2, y2 = bbox
                            crop = frame[y1:y2, x1:x2]
                            suggestion = gemini_advisor.suggest_label(crop, identities)
                            if suggestion:
                                label_sugg, conf = suggestion if isinstance(suggestion, tuple) else (suggestion, 0.0)
                                pending_gemini_suggestion = {
                                    "track_id": best_track_id,
                                    "label": label_sugg,
                                    "confidence": float(conf),
                                    "embedding": embedding,
                                }
                                print(f"Gemini suggestion for face {best_track_id}: {label_sugg} (conf={conf:.2f})")
                                # Auto-accept if confidence high (configurable via .env)
                                if conf >= float(auto_accept_conf) and label_sugg.upper() != "UNKNOWN":
                                    # apply immediately
                                    suggestion_label = label_sugg
                                    suggestion_embedding = embedding
                                    target_identity = None
                                    for identity in identities:
                                        if str(identity.get("label", "")).strip().lower() == suggestion_label.lower():
                                            target_identity = identity
                                            break

                                    if target_identity is None:
                                        identities.append(
                                            {
                                                "id": max([int(i.get("id", 0)) for i in identities], default=0) + 1,
                                                "version": 1,
                                                "label": suggestion_label,
                                                "templates": [list(suggestion_embedding)],
                                                "threshold": 0.22,
                                            }
                                        )
                                        print(f"[AUTO] Created new identity from Gemini suggestion: {suggestion_label}")
                                    else:
                                        templates = target_identity.get("templates", [])
                                        if isinstance(templates, list):
                                            templates.append(list(suggestion_embedding))
                                            target_identity["templates"] = templates[-20:]
                                        print(f"[AUTO] Adapted identity with Gemini suggestion: {suggestion_label}")
                                    save_identities(identities)
                                    # show an on-screen flash for a short duration
                                    try:
                                        auto_accept_flash = {"text": f"AUTO-ACCEPT: {suggestion_label}", "until": time.time() + 1.6}
                                    except Exception:
                                        auto_accept_flash = None
                                    pending_gemini_suggestion = None

                    face_centroids[best_track_id] = centroid
                    face_bboxes[best_track_id] = bbox

                    # Enrollment capture
                    if enrolling and embedding:
                        embedding_buffer.append(embedding)

                    # Draw bbox and label
                    x1, y1, x2, y2 = bbox
                    label = face_labels.get(best_track_id, "Unknown")
                    
                    if label == "Unknown":
                        color = (0, 165, 255)  # Orange
                    else:
                        color = (0, 255, 0)  # Green

                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

                    # Draw label background
                    text = label
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    label_x = x1
                    label_y = max(20, y1 - 10)
                    cv2.rectangle(out, (label_x - 4, label_y - text_size[1] - 8),
                                (label_x + text_size[0] + 4, label_y + 4), color, -1)
                    cv2.putText(out, text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                              0.8, (0, 0, 0), 2, cv2.LINE_AA)

                    if pending_gemini_suggestion and pending_gemini_suggestion.get("track_id") == best_track_id:
                        suggestion_text = f"Gemini: {pending_gemini_suggestion.get('label', 'UNKNOWN')} (conf={pending_gemini_suggestion.get('confidence', 0.0):.2f})"
                        cv2.putText(
                            out,
                            suggestion_text,
                            (x1, min(h - 10, y2 + 24)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                    stats = face_match_stats.get(best_track_id)
                    if stats is not None:
                        panel_x = x1
                        panel_y = min(h - 10, y2 + 34)
                        panel_w = min(220, max(140, x2 - x1))
                        panel_h = 54
                        if panel_y + panel_h > h:
                            panel_y = max(10, y1 - panel_h - 8)

                        cv2.rectangle(out, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (18, 18, 18), -1)
                        cv2.rectangle(out, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), color, 1)

                        threshold = max(0.001, float(stats.get("threshold", 0.22)))
                        dist = max(0.0, float(stats.get("dist", 0.0)))
                        local_score = max(0.0, min(1.0, 1.0 - (dist / max(threshold, 0.40))))
                        gemini_flag = float(stats.get("gemini", 0.0))

                        # record history
                        if best_track_id not in face_conf_history:
                            face_conf_history[best_track_id] = deque(maxlen=48)
                        face_conf_history[best_track_id].append(local_score)

                        cv2.putText(out, "Local match", (panel_x + 6, panel_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
                        cv2.putText(out, f"dist {dist:.3f} / thr {threshold:.3f}", (panel_x + 6, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

                        bar_x = panel_x + 92
                        bar_y = panel_y + 10
                        bar_w = panel_w - 98
                        bar_h = 10
                        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
                        cv2.rectangle(out, (bar_x, bar_y), (bar_x + int(bar_w * local_score), bar_y + bar_h), (0, 220, 0), -1)
                        cv2.line(out, (bar_x + int(bar_w * min(1.0, threshold / 0.40)), bar_y - 2), (bar_x + int(bar_w * min(1.0, threshold / 0.40)), bar_y + bar_h + 2), (0, 120, 255), 1)

                        # draw sparkline for history
                        hist = list(face_conf_history.get(best_track_id, []))
                        if hist:
                            sx = panel_x + 6
                            sy = panel_y + panel_h - 6
                            sw = panel_w - 14
                            sh = 10
                            pts = []
                            for i, v in enumerate(hist):
                                px = int(sx + (i / max(1, len(hist) - 1)) * sw)
                                py = int(sy - v * sh)
                                pts.append((px, py))
                            if len(pts) >= 2:
                                for a, b in zip(pts[:-1], pts[1:]):
                                    cv2.line(out, a, b, (180, 250, 180), 1)

                        cv2.putText(out, "Gemini", (panel_x + 6, panel_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
                        gem_color = (0, 200, 255) if gemini_flag > 0 else (90, 90, 90)
                        cv2.rectangle(out, (bar_x, panel_y + 38), (bar_x + bar_w, panel_y + 48), (60, 60, 60), -1)
                        cv2.rectangle(out, (bar_x, panel_y + 38), (bar_x + int(bar_w * gemini_flag), panel_y + 48), gem_color, -1)

            # Clean up inactive tracks
            for track_id in list(face_centroids.keys()):
                if track_id not in current_face_ids:
                    face_centroids.pop(track_id, None)
                    face_bboxes.pop(track_id, None)
                    face_labels.pop(track_id, None)
                    face_tracks.pop(track_id, None)
                    face_match_stats.pop(track_id, None)

            # Draw FPS
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.putText(out, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw enrollment status
            if enrolling:
                status = f"ENROLLING: {len(embedding_buffer)}/5 frames"
                cv2.putText(out, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(out, "Press SPACE to capture, S to save, C to cancel",
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            else:
                cv2.putText(out, "Press E to enroll, Q to quit", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw transient auto-accept notification if present
            if auto_accept_flash is not None:
                try:
                    if time.time() < float(auto_accept_flash.get("until", 0)):
                        txt = str(auto_accept_flash.get("text", ""))
                        scale = 0.9
                        thickness = 2
                        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                        cx = max(10, (w - tw) // 2)
                        cy = 40
                        cv2.rectangle(out, (cx - 8, cy - th - 8), (cx + tw + 8, cy + 6), (0, 180, 0), -1)
                        cv2.putText(out, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
                    else:
                        auto_accept_flash = None
                except Exception:
                    auto_accept_flash = None

            cv2.imshow("Face Recognition", out)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("e") and not enrolling:
                enrolling = True
                embedding_buffer.clear()
                print("Enrollment started. Keep face in view and press space to capture.")

            if key == ord(" ") and enrolling:
                print(f"Captured frame {len(embedding_buffer) + 1}")

            if key == ord("s") and enrolling and len(embedding_buffer) >= 3:
                # Average embeddings and save
                embs = np.stack(list(embedding_buffer), axis=0)
                final_emb = np.mean(embs, axis=0)
                emb_norm = np.linalg.norm(final_emb)
                if emb_norm > 1e-6:
                    final_emb = final_emb / emb_norm
                
                # Get name from user
                name = input("\nEnter name for this person: ").strip()
                if name:
                    new_id = max([int(i.get("id", 0)) for i in identities], default=0) + 1
                    identities.append({
                        "id": new_id,
                        "version": 1,
                        "label": name,
                        "templates": [final_emb.tolist()],
                        "threshold": 0.22,
                    })
                    save_identities(identities)
                    print(f"Enrolled {name} successfully!")
                
                enrolling = False
                embedding_buffer.clear()

            if key == ord("c") and enrolling:
                enrolling = False
                embedding_buffer.clear()
                print("Enrollment cancelled.")

            if key == ord("g") and gemini_advisor is not None and faces:
                target_index = 0
                target_points = extract_face_points(faces[target_index], w, h)
                target_bbox = compute_bbox(target_points, w, h)
                target_embedding = compute_landmark_descriptor(target_points)
                crop = frame[target_bbox[1]:target_bbox[3], target_bbox[0]:target_bbox[2]]
                suggestion = gemini_advisor.suggest_label(crop, identities)
                if suggestion:
                    pending_gemini_suggestion = {
                        "track_id": current_face_ids[target_index] if current_face_ids else 0,
                        "label": suggestion,
                        "embedding": target_embedding,
                    }
                    print(f"Gemini suggestion: {suggestion} (press a to accept)")

            if key == ord("a") and pending_gemini_suggestion is not None:
                suggestion_label = str(pending_gemini_suggestion.get("label", "")).strip()
                suggestion_embedding = pending_gemini_suggestion.get("embedding", [])
                if suggestion_label and suggestion_label.upper() != "UNKNOWN" and suggestion_embedding:
                    target_identity = None
                    for identity in identities:
                        if str(identity.get("label", "")).strip().lower() == suggestion_label.lower():
                            target_identity = identity
                            break

                    if target_identity is None:
                        identities.append(
                            {
                                "id": max([int(i.get("id", 0)) for i in identities], default=0) + 1,
                                "version": 1,
                                "label": suggestion_label,
                                "templates": [list(suggestion_embedding)],
                                "threshold": 0.22,
                            }
                        )
                        print(f"Created new identity from Gemini suggestion: {suggestion_label}")
                    else:
                        templates = target_identity.get("templates", [])
                        if isinstance(templates, list):
                            templates.append(list(suggestion_embedding))
                            target_identity["templates"] = templates[-20:]
                        print(f"Adapted identity with Gemini suggestion: {suggestion_label}")

                    save_identities(identities)
                    pending_gemini_suggestion = None

    finally:
        close_detector()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    exit(main())
