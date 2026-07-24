"""
Real-time face recognition with Gemini auto-training.

Features:
- Real-time face detection and recognition from webcam
- Gemini-powered auto-labeling and auto-enrollment
- User confirmation for auto-trained faces
- Confidence-based retraining flagging
- Full audit logging

Controls:
  q: quit
  e: start enrollment (manual mode)
  space: capture frame during manual enrollment
  s: save manual enrollment
  c: cancel manual enrollment
  y: approve pending auto-enrollment
  n: reject pending auto-enrollment
  TAB: show audit log
  l: list all pending confirmations

Install requirements:
    pip install opencv-python mediapipe numpy google-genai pillow

Set GEMINI_API_KEY in .env or environment:
    GEMINI_API_KEY=your-key-here

Run:
    python main.py autotrain
"""

import json
import importlib
import os
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit(
        "mediapipe is not installed. Install with: pip install mediapipe"
    ) from exc

try:
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
    genai = importlib.import_module("google.genai")
except Exception:
    try:
        genai = importlib.import_module("google.generativeai")
    except Exception:
        genai = None

from .gemini_auto_trainer import GeminiAutoTrainer
from .paths import environment_file, model_path

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]

DESCRIPTOR_LANDMARKS = [1, 33, 61, 78, 133, 152, 199, 234, 263, 291, 308, 362, 454]
DESCRIPTOR_DISTANCE_PAIRS = [
    (33, 263), (1, 152), (61, 291), (78, 308), (133, 362), (234, 454), (61, 152), (291, 152),
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def load_environment_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a local .env file if present."""
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


def ensure_face_landmarker_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Face Landmarker model...")
    urlretrieve(MODEL_URL, str(model_path))
    return model_path


def create_landmark_detector():
    """Return detector callable and cleanup function."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_file = ensure_face_landmarker_model(model_path("face_landmarker.task"))

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_file)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=4,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(rgb_frame: np.ndarray, timestamp_ms: int):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        return result.face_landmarks if result.face_landmarks else []

    return detect, landmarker.close


def extract_face_points(face_landmarks, width: int, height: int) -> List[Point]:
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


def compute_face_centroid(points: List[Point]) -> Point:
    """Compute center of face points."""
    if not points:
        return (0, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (int(np.mean(xs)), int(np.mean(ys)))


def compute_bbox(points: List[Point], width: int, height: int, margin: int = 20) -> BBox:
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


def compute_landmark_descriptor(points: List[Point]) -> List[float]:
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
    identities_path = model_path("face_identities.json")
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
    identities_path = model_path("face_identities.json")
    identities_path.parent.mkdir(parents=True, exist_ok=True)
    with open(identities_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def find_best_match(descriptor: List[float], identities: List[Dict]) -> Tuple[Optional[str], float]:
    """Find best matching identity for a descriptor."""
    if not identities or not descriptor:
        return None, 0.0

    best_match = None
    best_score = 0.0

    for identity in identities:
        label = identity.get("label", "").strip()
        threshold = identity.get("threshold", 0.22)
        templates = identity.get("templates", [])

        if not templates:
            continue

        # Compute similarity with all templates
        desc_norm = np.array(descriptor, dtype=np.float32)
        desc_norm = desc_norm / (np.linalg.norm(desc_norm) + 1e-6)

        max_similarity = 0.0
        for template in templates:
            template_norm = np.array(template, dtype=np.float32)
            template_norm = template_norm / (np.linalg.norm(template_norm) + 1e-6)
            similarity = float(np.dot(desc_norm, template_norm))
            max_similarity = max(max_similarity, similarity)

        # Check if above threshold
        if max_similarity >= threshold and max_similarity > best_score:
            best_match = label
            best_score = max_similarity

    return best_match, best_score


def display_pending_confirmations(frame: np.ndarray, confirmations: Dict, y_offset: int = 30) -> int:
    """Display pending confirmations on frame. Returns next y_offset."""
    if not confirmations:
        return y_offset

    cv2.putText(
        frame, 
        f"PENDING CONFIRMATIONS: {len(confirmations)}", 
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
    )
    y_offset += 30

    for i, (temp_id, decision) in enumerate(list(confirmations.items())[:3]):
        label = decision["label"]
        confidence = decision["confidence"]
        text = f"  [{i+1}] {temp_id}: {label} ({confidence:.2f}) - Press Y/N"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
        y_offset += 25

    return y_offset


def display_auto_training_status(frame: np.ndarray, decision: Optional[Dict], y_offset: int = 30) -> int:
    """Display auto-training decision on frame. Returns next y_offset."""
    if not decision:
        return y_offset

    action = decision.get("action", "skip")
    label = decision.get("label", "?")
    confidence = decision.get("confidence", 0.0)
    
    color_map = {
        "skip": (200, 200, 200),
        "enroll_pending": (0, 165, 255),
        "enroll_confident": (0, 255, 0),
        "flag_retraining": (0, 0, 255),
    }
    color = color_map.get(action, (200, 200, 200))

    status_text = f"AUTO-TRAIN: {action} | {label} ({confidence:.2f})"
    cv2.putText(frame, status_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return y_offset + 25


def show_audit_log(trainer: GeminiAutoTrainer, max_lines: int = 10):
    """Print recent audit log entries."""
    log_entries = trainer.get_audit_log(limit=max_lines)
    print("\n" + "="*80)
    print(f"RECENT AUTO-TRAINING AUDIT LOG ({len(log_entries)} entries)")
    print("="*80)
    for entry in log_entries:
        print(f"[{entry['timestamp']}] {entry['action']}: {entry}")
    print("="*80 + "\n")


def main():
    """Main real-time recognition loop with Gemini auto-training."""
    # Get Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment or .env file")
        print("Set it in .env or environment variable")
        return

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"

    # Initialize auto-trainer
    try:
        trainer = GeminiAutoTrainer(
            api_key=api_key,
            auto_enroll_threshold=0.85,
            confidence_retraining_threshold=0.65,
            require_user_confirmation=True,
            model_name=model_name,
        )
        print(f"✓ Gemini auto-trainer initialized (model={model_name})")
    except Exception as e:
        print(f"ERROR: Could not initialize auto-trainer: {e}")
        return

    # Initialize face detector
    detect_faces, close_detector = create_landmark_detector()
    print("✓ Face detector initialized")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    print("✓ Webcam opened")
    print("\nControls:")
    print("  q: quit")
    print("  e: start manual enrollment")
    print("  space: capture frame during manual enrollment")
    print("  s: save manual enrollment")
    print("  c: cancel manual enrollment")
    print("  y: approve pending auto-enrollment")
    print("  n: reject pending auto-enrollment")
    print("  l: list pending confirmations")
    print("  TAB: show audit log\n")

    enrolling = False
    enroll_frames = deque(maxlen=5)
    enroll_label = ""
    frame_count = 0
    timestamp_ms = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            height, width = frame.shape[:2]

            # Convert to RGB for detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            face_landmarks_list = detect_faces(rgb_frame, timestamp_ms)

            # Load current identities
            identities = load_identities()

            # Track auto-training decision for display
            latest_decision = None

            # Process each detected face
            for face_landmarks in face_landmarks_list:
                points = extract_face_points(face_landmarks, width, height)
                bbox = compute_bbox(points, width, height, margin=20)
                x1, y1, x2, y2 = bbox

                # Extract face crop
                face_crop_bgr = frame[y1:y2, x1:x2].copy()
                if face_crop_bgr.size == 0:
                    continue

                # Compute descriptor
                descriptor = compute_landmark_descriptor(points)
                if not descriptor:
                    continue

                # Find local match
                matched_label, matched_conf = find_best_match(descriptor, identities)

                # Get Gemini auto-training decision
                if not enrolling:
                    decision = trainer.process_detected_face(
                        frame_bgr=face_crop_bgr,
                        face_descriptor=descriptor,
                        current_match=(matched_label, matched_conf) if matched_label else None,
                        identities=identities,
                    )
                    latest_decision = decision
                    
                    # Handle auto-enrollment if confident
                    if decision["action"] == "enroll_confident":
                        trainer.enroll_face(decision, identities)
                        identities = load_identities()  # Reload

                # Draw bounding box
                display_label = matched_label or "UNKNOWN"
                confidence = matched_conf if matched_label else 0.0
                
                color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{display_label} ({confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                # Draw landmarks
                for point in points:
                    cv2.circle(frame, point, 2, (255, 0, 0), -1)

                if enrolling:
                    enroll_frames.append(descriptor)

            # Display status
            y_pos = 30
            pending = trainer.get_pending_confirmations()
            if pending:
                y_pos = display_pending_confirmations(frame, pending, y_pos)
            if latest_decision and not enrolling:
                y_pos = display_auto_training_status(frame, latest_decision, y_pos)

            # Display help text
            if enrolling:
                cv2.putText(
                    frame,
                    f"ENROLLING: {enroll_label} ({len(enroll_frames)} frames) - space to capture, s to save, c to cancel",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Press 'e' to enroll manually, 'l' for pending, 'TAB' for audit log, 'q' to quit",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

            # Show frame
            cv2.imshow("Face Recognition with Gemini Auto-Training", frame)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("e") and not enrolling:
                enrolling = True
                enroll_label = input("Enter identity label: ").strip()
                if not enroll_label:
                    enrolling = False
            elif key == ord(" ") and enrolling:
                pass  # Frames automatically added to deque
            elif key == ord("s") and enrolling:
                if enroll_frames and enroll_label:
                    avg_descriptor = list(np.mean(enroll_frames, axis=0))
                    decision = {
                        "label": enroll_label,
                        "frame_descriptor": avg_descriptor,
                        "temp_id": "manual",
                    }
                    trainer.enroll_face(decision, identities)
                    identities = load_identities()
                    print(f"✓ Enrolled {len(enroll_frames)} frames for {enroll_label}")
                enrolling = False
                enroll_frames.clear()
            elif key == ord("c") and enrolling:
                enrolling = False
                enroll_frames.clear()
                print("Enrollment cancelled")
            elif key == ord("y"):  # Approve pending
                pending = trainer.get_pending_confirmations()
                if pending:
                    temp_id = list(pending.keys())[0]
                    trainer.confirm_pending(temp_id, approved=True, identities=identities)
                    identities = load_identities()
                    print(f"✓ Approved: {pending[temp_id]['label']}")
            elif key == ord("n"):  # Reject pending
                pending = trainer.get_pending_confirmations()
                if pending:
                    temp_id = list(pending.keys())[0]
                    trainer.confirm_pending(temp_id, approved=False, identities=identities)
                    print(f"✗ Rejected: {pending[temp_id]['label']}")
            elif key == ord("l"):  # List pending
                pending = trainer.get_pending_confirmations()
                print(f"\nPending confirmations: {len(pending)}")
                for temp_id, decision in pending.items():
                    print(f"  [{temp_id}] {decision['label']} ({decision['confidence']:.2f})")
                print()
            elif key == 9:  # TAB
                show_audit_log(trainer)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        close_detector()
        print("\n✓ Cleanup complete")


if __name__ == "__main__":
    main()
