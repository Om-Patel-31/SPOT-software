"""
SPOT Computer-Vision Suite - Deep Learning Standalone Hub
=========================================================
Production-grade desktop application featuring:
- OpenCV YuNet (CNN Face Detection) & SFace (128D Deep Feature Embeddings)
- Automated ONNX Model Provisioning & Dependency Management
- Mandatory AR 3D Biometric Calibration Gate (Locks Live Stream)
- Interactive 3-Step AR Calibration Wizard (Center, Left, Right Pose Validation)
- Real-time PyQt6 Dashboard with System KPIs, Matplotlib Analytics, & Identity Manager

Usage:
    python main.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# ==============================================================================
# 1. DEPENDENCY CHECKER & AUTO-INSTALLER
# ==============================================================================

def ensure_dependencies() -> None:
    """Verify required packages and auto-install via pip if missing."""
    required = {
        "PyQt6": "PyQt6",
        "cv2": "opencv-python",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
    }

    missing = []
    for module_name, package_name in required.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        should_install = False
        try:
            from PyQt6.QtWidgets import QApplication, QMessageBox
            _ = QApplication.instance() or QApplication(sys.argv)

            reply = QMessageBox.question(
                None,
                "Missing Dependencies",
                f"The following required packages are missing:\n\n• {', '.join(missing)}\n\nInstall automatically?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            should_install = (reply == QMessageBox.StandardButton.Yes)
        except Exception:
            print("\n" + "=" * 60)
            print("  SPOT DESKTOP HUB DEPENDENCY CHECK")
            print("=" * 60)
            print(f"  Missing required packages: {', '.join(missing)}")
            response = input("  Install them now via pip? [y/N]: ").strip().lower()
            should_install = response in ("y", "yes")

        if should_install:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
                import site
                importlib.reload(site)
            except subprocess.CalledProcessError as err:
                print(f"\n✗ Failed to install dependencies: {err}")
                sys.exit(1)
        else:
            sys.exit(1)


ensure_dependencies()

import cv2
import numpy as np
from PyQt6.QtCore import QProcess, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QIcon, QImage, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSplitter, QTabBar, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SPOT.DeepFace")


# ==============================================================================
# 2. PATHS, STORAGE ENGINE & AUTOMATED ONNX PROVISIONING
# ==============================================================================

def get_project_root() -> Path:
    return Path(__file__).resolve().parent

def get_models_dir() -> Path:
    models_dir = get_project_root() / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

def get_identities_file_path() -> Path:
    return get_models_dir() / "face_identities.json"

def get_audit_log_path() -> Path:
    return get_models_dir() / "auto_training_audit.log"

def ensure_onnx_models() -> Tuple[str, str]:
    """Auto-downloads OpenCV YuNet (Detection) and SFace (Recognition) ONNX models."""
    yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    sface_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    
    yunet_path = get_models_dir() / "face_detection_yunet.onnx"
    sface_path = get_models_dir() / "face_recognition_sface.onnx"
    
    for url, path in [(yunet_url, yunet_path), (sface_url, sface_path)]:
        if not path.exists():
            print(f"[Model Provisioning] Downloading {path.name} (~10-15MB)...")
            try:
                urllib.request.urlretrieve(url, str(path))
            except Exception as e:
                raise RuntimeError(
                    f"Unable to download {path.name}.\n\n"
                    "Please check your Internet connection "
                    "or manually place the ONNX model into data/models.\n\n"
                    f"Original error:\n{e}"
                )
                
    return str(yunet_path), str(sface_path)

def load_identities() -> List[Dict[str, Any]]:
    path = get_identities_file_path()
    if not path.exists(): return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception: return []

def save_identities(identities: List[Dict[str, Any]]) -> None:
    try:
        with open(get_identities_file_path(), "w", encoding="utf-8") as f:
            json.dump(identities, f, indent=2)
    except Exception as e: logger.error(f"Save error: {e}")

def append_audit_log(entry: Dict[str, Any]) -> None:
    try:
        with open(get_audit_log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception: pass

def has_calibrated_profiles() -> bool:
    for identity in load_identities():
        if identity.get("templates") and len(identity["templates"]) > 0:
            return True
    return False


# ==============================================================================
# 3. DEEP LEARNING MATCHING & YAW ESTIMATION ENGINE
# ==============================================================================

def estimate_head_yaw(face_data: np.ndarray) -> float:
    """Estimates head yaw using YuNet's 5 facial landmarks."""
    # YuNet layout: [x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y, nose_x, nose_y, right_mouth_x, ...]
    re_x = face_data[4]
    le_x = face_data[6]
    nose_x = face_data[8]

    # Calculate horizontal distances from nose to eyes
    dist_right = abs(nose_x - re_x)
    dist_left = abs(nose_x - le_x)
    total = dist_right + dist_left + 1e-6

    # Ratio maps roughly from -1.0 (turned right) to 1.0 (turned left)
    ratio = (dist_right - dist_left) / total
    return float(ratio * 60.0)

def match_face_sface(live_feature: np.ndarray, identities: List[Dict[str, Any]], recognizer) -> Tuple[str, float]:
    """Uses OpenCV SFace Cosine Distance to match 128D Deep Embeddings."""
    if live_feature is None or len(identities) == 0:
        return "UNKNOWN", 0.0

    best_label = "UNKNOWN"
    best_score = 0.0

    for identity in identities:
        label = identity.get("label", "UNKNOWN")
        templates = identity.get("templates", [])
        req_threshold = identity.get("threshold", 0.363)  # 0.363 is SFace standard threshold

        for tmpl in templates:
            # --- SAFEGUARD: Skip legacy or corrupted templates ---
            if len(tmpl) != 128:
                continue
            
            # Reshape stored 1D list back into (1, 128) float32 matrix
            stored_feature = np.array(tmpl, dtype=np.float32).reshape(1, 128)
            
            # SFace natively calculates Cosine Similarity (Higher is better, >= 0.363 is a match)
            score = recognizer.match(live_feature, stored_feature, cv2.FaceRecognizerSF_FR_COSINE)
            
            if score > best_score:
                best_score = score
                if score >= req_threshold:
                    best_label = label

    return best_label, best_score


# ==============================================================================
# 4. WEBCAM THREAD WITH YUNET & SFACE INFERENCE
# ==============================================================================

class WebcamThread(QThread):
    """Processes real-time camera frames using CNN Face Detection & Recognition."""
    frame_processed = pyqtSignal(QImage, np.ndarray, object, object, float)

    def __init__(self, mode: str = "realtime", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.running = True
        
        # Provision and Load ONNX Models
        try:
            yunet_path, sface_path = ensure_onnx_models()

            if not hasattr(cv2, "FaceDetectorYN"):
                raise RuntimeError(
                    "This OpenCV build does not support FaceDetectorYN.\n"
                    "Please install opencv-python >= 4.8."
                )

            if not hasattr(cv2, "FaceRecognizerSF"):
                raise RuntimeError(
                    "This OpenCV build does not support FaceRecognizerSF.\n"
                    "Please install opencv-python >= 4.8."
                )

            self.detector = cv2.FaceDetectorYN.create(
                yunet_path,
                "",
                (1280, 720),
                score_threshold=0.85,
            )

            self.recognizer = cv2.FaceRecognizerSF.create(
                sface_path,
                "",
            )

        except Exception as e:
            QMessageBox.critical(
                None,
                "CNN Initialization Failed",
                str(e),
            )
            raise

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            QMessageBox.critical(
                None,
                "Camera Error",
                "No webcam could be opened."
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)
            
            identities = load_identities()
            latest_feature = None
            primary_face_data = None
            yaw_angle = 0.0

            if faces is not None:
                # Find the largest face by bounding box area to treat as primary
                primary_face_data = max(faces, key=lambda f: f[2] * f[3])
                
                # Extract 128D Deep Feature Embedding using SFace
                latest_feature = self.recognizer.feature(frame, primary_face_data)
                yaw_angle = estimate_head_yaw(primary_face_data)

                # Process all detected faces for the visual feed
                for face in faces:
                    x1, y1, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    
                    # AR Wireframe Visuals (Connect eyes, nose, mouth)
                    pts = [
                        (int(face[4]), int(face[5])), (int(face[6]), int(face[7])),
                        (int(face[8]), int(face[9])), (int(face[10]), int(face[11])), (int(face[12]), int(face[13]))
                    ]
                    for pt in pts:
                        cv2.circle(frame, pt, 3, (74, 222, 128), -1)
                    
                    # Draw Nose Vector
                    vec_end_x = int(pts[2][0] + (estimate_head_yaw(face) * 2.5))
                    cv2.arrowedLine(frame, pts[2], (vec_end_x, pts[2][1]), (250, 204, 21), 2, tipLength=0.3)

                    # Extract Feature and Match
                    feature = self.recognizer.feature(frame, face)
                    match_label, score = match_face_sface(feature, identities, self.recognizer)
                    
                    is_known = (match_label != "UNKNOWN")
                    color = (74, 222, 128) if is_known else (56, 189, 248)

                    cv2.rectangle(frame, (x1, y1), (x1 + fw, y1 + fh), color, 2)
                    disp_text = f"{match_label} ({score*100:.0f}%)" if is_known else "UNKNOWN"
                    cv2.putText(frame, disp_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            status_text = f"Mode: {self.mode.upper()} | SFace Deep Embeddings"
            cv2.rectangle(frame, (10, 10), (350, 42), (15, 23, 42), -1)
            cv2.putText(frame, status_text, (20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 250, 252), 1)

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_out.data, w, h, w * c, QImage.Format.Format_RGB888)

            self.frame_processed.emit(qimg.copy(), frame.copy(), latest_feature, primary_face_data, yaw_angle)
            time.sleep(0.03)

        cap.release()


# ==============================================================================
# 5. VISUAL AR 3D CALIBRATION WIZARD DIALOG
# ==============================================================================

class AR3DFaceCalibrationDialog(QDialog):
    """3-Step AR Calibration Wizard enforcing deep-feature pose capture."""

    STEPS = [
        ("CENTER POSE", "Look straight into the camera", -10.0, 10.0),
        ("LEFT POSE", "Turn head slightly to the LEFT (~25°)", 15.0, 45.0),
        ("RIGHT POSE", "Turn head slightly to the RIGHT (~25°)", -45.0, -15.0),
    ]

    calibration_completed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CNN Biometric Face Calibration")
        self.resize(850, 680)
        self.setStyleSheet("""
            QDialog { background-color: #0f172a; color: #f8fafc; }
            QLabel { color: #f8fafc; font-family: 'Segoe UI', sans-serif; }
            QLineEdit { background-color: #1e293b; border: 1px solid #38bdf8; border-radius: 6px; padding: 6px 10px; }
            QPushButton { background-color: #0284c7; color: white; font-weight: bold; border-radius: 6px; padding: 10px; }
            QPushButton:hover { background-color: #0369a1; }
            QPushButton:disabled { background-color: #334155; color: #94a3b8; }
            QProgressBar { border: 1px solid #334155; border-radius: 6px; text-align: center; background-color: #1e293b; }
            QProgressBar::chunk { background-color: #4ade80; border-radius: 5px; }
        """)

        self.current_step_idx = 0
        self.captured_templates: List[np.ndarray] = []
        self.hold_start_time: Optional[float] = None
        self.is_completed = False

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        name_box = QHBoxLayout()
        lbl_name = QLabel("Enrolled Subject Name:")
        lbl_name.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("Enter full name...")
        name_box.addWidget(lbl_name)
        name_box.addWidget(self.txt_name)
        layout.addLayout(name_box)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.video_label = QLabel("Initializing SFace CNN Feed...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #020617; border: 2px solid #38bdf8; border-radius: 8px;")
        self.video_label.setMinimumSize(520, 380)
        splitter.addWidget(self.video_label)

        side_panel = QFrame()
        side_panel.setStyleSheet("background-color: #1e293b; border-radius: 8px; border: 1px solid #334155;")
        side_layout = QVBoxLayout(side_panel)

        self.lbl_step_title = QLabel("Step 1 of 3: CENTER POSE")
        self.lbl_step_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #38bdf8;")

        self.lbl_instruction = QLabel("Look directly at the camera.")
        self.lbl_instruction.setStyleSheet("font-size: 12px; color: #94a3b8;")

        self.lbl_live_feedback = QLabel("Pose Yaw: 0° (Aligning...)")
        self.lbl_live_feedback.setStyleSheet("font-size: 12px; font-weight: bold; color: #facc15;")

        telemetry_box = QFrame()
        telemetry_box.setStyleSheet("background-color: #0f172a; border-radius: 6px; padding: 6px; border: 1px solid #334155;")
        tel_layout = QVBoxLayout(telemetry_box)
        tel_title = QLabel("DEEP FEATURE TELEMETRY")
        tel_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #38bdf8;")
        
        self.lbl_tel_yaw = QLabel("Yaw Angle: 0.0°")
        self.lbl_tel_dim = QLabel("Embedding: 128D CNN Vector")
        
        for lbl in [self.lbl_tel_yaw, self.lbl_tel_dim]:
            lbl.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; color: #f8fafc;")
            tel_layout.addWidget(lbl)

        side_layout.addWidget(self.lbl_step_title)
        side_layout.addWidget(self.lbl_instruction)
        side_layout.addWidget(self.lbl_live_feedback)
        side_layout.addWidget(telemetry_box)
        side_layout.addStretch()

        splitter.addWidget(side_panel)
        splitter.setSizes([540, 280])
        layout.addWidget(splitter, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.btn_save = QPushButton("Save Biometric Calibration & Unlock Realtime Stream")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.finish_calibration)
        layout.addWidget(self.btn_save)

        self.thread = WebcamThread(mode="calibration", parent=self)
        self.thread.frame_processed.connect(self.on_frame_processed)
        self.thread.start()

    def closeEvent(self, event):
        if hasattr(self, "thread") and self.thread.isRunning():
            self.thread.stop()
        super().closeEvent(event)

    @pyqtSlot(QImage, np.ndarray, object, object, float)
    def on_frame_processed(self, qimg: QImage, bgr_frame: np.ndarray, feature: Optional[np.ndarray], face_data: Any, yaw: float):
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        if self.is_completed or feature is None or face_data is None:
            return

        step_label, instruction, min_yaw, max_yaw = self.STEPS[self.current_step_idx]

        self.lbl_tel_yaw.setText(f"Yaw Angle: {yaw:+.2f}°")
        self.lbl_step_title.setText(f"Step {self.current_step_idx + 1} of 3: {step_label}")
        self.lbl_instruction.setText(instruction)

        if min_yaw <= yaw <= max_yaw:
            if self.hold_start_time is None:
                self.hold_start_time = time.time()

            elapsed = time.time() - self.hold_start_time
            hold_progress = min(100, int((elapsed / 0.5) * 100))  # Reduced to 0.5s for snappy AR feel
            self.progress_bar.setValue(hold_progress)
            self.lbl_live_feedback.setText(f"Pose Validated ({yaw:.1f}°) - Hold steady... {hold_progress}%")
            self.lbl_live_feedback.setStyleSheet("color: #4ade80; font-weight: bold;")

            if elapsed >= 0.5:
                # Append the 1D representation of the 128D array
                self.captured_templates.append(feature[0].copy())
                self.current_step_idx += 1
                self.hold_start_time = None
                self.progress_bar.setValue(0)

                if self.current_step_idx >= len(self.STEPS):
                    self.is_completed = True
                    self.lbl_step_title.setText("Calibration Complete!")
                    self.lbl_instruction.setText("All 3 multi-angle 128D embeddings generated.")
                    self.lbl_live_feedback.setText("Ready to save identity profile and unlock Realtime Stream.")
                    self.lbl_live_feedback.setStyleSheet("color: #4ade80; font-weight: bold;")
                    self.btn_save.setEnabled(True)
        else:
            self.hold_start_time = None
            self.progress_bar.setValue(0)
            self.lbl_live_feedback.setText(f"Pose Missing (Yaw: {yaw:.1f}°, Target: {min_yaw}° to {max_yaw}°)")
            self.lbl_live_feedback.setStyleSheet("color: #f87171; font-weight: bold;")

    def finish_calibration(self):
        name = self.txt_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Input Required", "Please enter a name before completing enrollment.")
            return

        identities = load_identities()
        target_identity = next((i for i in identities if str(i.get("label", "")).lower() == name.lower()), None)
        new_templates = [t.tolist() for t in self.captured_templates]

        if target_identity is None:
            new_id = max([int(i.get("id", 0)) for i in identities], default=0) + 1
            identities.append({
                "id": new_id,
                "label": name,
                "templates": new_templates,
                "threshold": 0.363,  # OpenCV SFace Default Threshold
            })
        else:
            existing = target_identity.get("templates", [])
            existing.extend(new_templates)
            target_identity["templates"] = existing[-30:]

        save_identities(identities)

        append_audit_log({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "action": "sface_cnn_calibration",
            "label": name,
            "confidence": 1.0,
        })

        if hasattr(self, "thread") and self.thread.isRunning():
            self.thread.stop()

        self.calibration_completed.emit(name)
        QMessageBox.information(
            self,
            "Calibration Complete",
            f"Identity '{name}' calibrated with 128D CNN features!\n\nRealtime Recognition Stream is now UNLOCKED."
        )
        self.accept()


# ==============================================================================
# 6. MANAGED TAB VIEWS & DASHBOARD
# ==============================================================================

class EmbeddedWebcamTab(QWidget):
    """Realtime stream tab for calibrated recognition."""
    def __init__(self, mode: str, display_title: str, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.display_title = display_title
        self.session_captured_frames: List[np.ndarray] = []
        self.latest_bgr_frame: Optional[np.ndarray] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        top_bar = QHBoxLayout()
        self.lbl_status = QLabel(f"Program: <b>{display_title}</b> | Camera: <font color='#4ade80'>Live</font>")
        self.lbl_status.setStyleSheet("font-size: 13px; color: #f8fafc;")
        self.btn_stop = QPushButton("Stop Camera")
        self.btn_stop.setStyleSheet("background-color: #ef4444; color: white; font-weight: bold;")
        self.btn_stop.clicked.connect(self.stop_feed)
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_stop)
        layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.video_label = QLabel("Initializing CNN Feed...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #020617; border: 2px solid #334155; border-radius: 8px;")
        self.video_label.setMinimumSize(540, 400)
        splitter.addWidget(self.video_label)

        gallery_widget = QWidget()
        gallery_vbox = QVBoxLayout(gallery_widget)
        gallery_vbox.setContentsMargins(5, 0, 0, 0)
        lbl_gallery = QLabel("Captured Frames Gallery")
        lbl_gallery.setStyleSheet("font-size: 12px; font-weight: bold; color: #38bdf8;")
        gallery_vbox.addWidget(lbl_gallery)

        self.list_gallery = QListWidget()
        self.list_gallery.setIconSize(QPixmap(120, 90).size())
        self.list_gallery.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; color: #f8fafc;")
        gallery_vbox.addWidget(self.list_gallery)
        splitter.addWidget(gallery_widget)
        splitter.setSizes([700, 300])
        layout.addWidget(splitter, stretch=1)

        ctrl_bar = QHBoxLayout()
        btn_capture = QPushButton("Capture Frame")
        btn_export = QPushButton("Export Frames to Disk")
        btn_capture.clicked.connect(self.capture_frame)
        btn_export.clicked.connect(self.export_frames_to_disk)
        ctrl_bar.addWidget(btn_capture)
        ctrl_bar.addWidget(btn_export)
        layout.addLayout(ctrl_bar)

        self.thread = WebcamThread(mode=self.mode)
        self.thread.frame_processed.connect(self.update_frame)
        self.thread.start()

    @pyqtSlot(QImage, np.ndarray, object, object, float)
    def update_frame(self, qimg: QImage, bgr_frame: np.ndarray, feature: Optional[np.ndarray], face_data: Any, yaw: float):
        self.latest_bgr_frame = bgr_frame
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def capture_frame(self):
        if self.latest_bgr_frame is None: return
        self.session_captured_frames.append(self.latest_bgr_frame.copy())
        rgb = cv2.cvtColor(self.latest_bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, w * c, QImage.Format.Format_RGB888)
        self.list_gallery.addItem(QListWidgetItem(QIcon(QPixmap.fromImage(qimg).scaled(120, 90, Qt.AspectRatioMode.KeepAspectRatio)), f"Frame #{len(self.session_captured_frames)}"))

    def export_frames_to_disk(self):
        if not self.session_captured_frames: return
        target_dir = QFileDialog.getExistingDirectory(self, "Select Folder")
        if target_dir:
            out = Path(target_dir)
            ts = time.strftime("%Y%m%d_%H%M%S")
            for idx, img in enumerate(self.session_captured_frames, 1):
                cv2.imwrite(str(out / f"frame_{ts}_{idx}.jpg"), img)
            QMessageBox.information(self, "Export Complete", f"Exported {len(self.session_captured_frames)} frames.")

    def stop_feed(self):
        if hasattr(self, "thread") and self.thread.isRunning():
            self.thread.stop()
            self.lbl_status.setText(f"Program: <b>{self.display_title}</b> | Camera: <font color='#ef4444'>Stopped</font>")
            self.btn_stop.setEnabled(False)

class KPICard(QFrame):
    def __init__(self, title: str, accent_color: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QFrame { background-color: #1e293b; border-radius: 10px; border: 1px solid #334155; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        self.lbl_title = QLabel(title.upper())
        self.lbl_title.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold; border: none;")
        self.lbl_value = QLabel("--")
        self.lbl_value.setStyleSheet(f"color: {accent_color}; font-size: 24px; font-weight: bold; border: none;")
        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_value)
    def set_value(self, value: str):
        self.lbl_value.setText(value)

class SPOTDesktopWindow(QMainWindow):
    """Main Application Window enforcing Calibration Gate."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPOT Computer Vision Suite - SFace Biometric Hub")
        self.resize(1280, 850)
        self.setStyleSheet("""
            QMainWindow { background-color: #0f172a; }
            QLabel { color: #f8fafc; font-family: 'Segoe UI', sans-serif; }
            QPushButton { background-color: #0284c7; color: white; font-weight: bold; border-radius: 6px; padding: 8px 14px; border: none; font-size: 12px; }
            QPushButton:hover { background-color: #0369a1; }
            QPushButton:disabled { background-color: #334155; color: #64748b; }
            QTabWidget::pane { border: 1px solid #334155; background: #0f172a; }
            QTabBar::tab { background: #1e293b; color: #94a3b8; padding: 10px 18px; font-weight: bold; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }
            QTabBar::tab:selected { background: #0284c7; color: white; }
            QTableWidget { background-color: #020617; color: #f8fafc; gridline-color: #334155; border: 1px solid #334155; }
            QHeaderView::section { background-color: #1e293b; color: #38bdf8; font-weight: bold; border: 1px solid #334155; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)

        header = QHBoxLayout()
        title_lbl = QLabel("SPOT Computer Vision Suite")
        title_lbl.setStyleSheet("font-size: 22px; font-weight: bold; color: #38bdf8;")
        
        self.lbl_gate_badge = QLabel("UNCALIBRATED (REALTIME LOCKED)")
        self.lbl_gate_badge.setStyleSheet("background-color: #7f1d1d; color: #fca5a5; font-weight: bold; padding: 5px 12px; border-radius: 12px; border: 1px solid #ef4444; font-size: 11px;")

        header.addWidget(title_lbl)
        header.addWidget(self.lbl_gate_badge)
        header.addStretch()

        btn_run_calibration = QPushButton("Open CNN Calibration Wizard")
        btn_run_calibration.setStyleSheet("background-color: #10b981; font-weight: bold; padding: 8px 16px;")
        btn_run_calibration.clicked.connect(self.launch_calibration_wizard)
        header.addWidget(btn_run_calibration)
        main_layout.addLayout(header)

        self.guard_banner = QFrame()
        self.guard_banner.setStyleSheet("background-color: #451a03; border: 1px solid #f59e0b; border-radius: 8px;")
        guard_layout = QHBoxLayout(self.guard_banner)
        self.lbl_banner_msg = QLabel("<b>MANDATORY CALIBRATION REQUIRED:</b> You must calibrate at least one identity using the 3D Wizard before Realtime unlocks.")
        self.lbl_banner_msg.setStyleSheet("color: #fef3c7; font-size: 12px;")
        guard_layout.addWidget(self.lbl_banner_msg)
        main_layout.addWidget(self.guard_banner)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)

        self.dashboard_tab = QWidget()
        self.setup_dashboard_tab()
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.tabBar().setTabButton(0, QTabBar.ButtonPosition.RightSide, None)

        main_layout.addWidget(self.tabs)
        self.check_calibration_status()

    def check_calibration_status(self) -> bool:
        calibrated = has_calibrated_profiles()
        if calibrated:
            self.lbl_gate_badge.setText("SYSTEM CALIBRATED (REALTIME READY)")
            self.lbl_gate_badge.setStyleSheet("background-color: #064e3b; color: #6ee7b7; font-weight: bold; padding: 5px 12px; border-radius: 12px; border: 1px solid #10b981; font-size: 11px;")
            self.btn_realtime_launch.setEnabled(True)
            self.guard_banner.setVisible(False)
        else:
            self.lbl_gate_badge.setText("UNCALIBRATED (REALTIME LOCKED)")
            self.lbl_gate_badge.setStyleSheet("background-color: #7f1d1d; color: #fca5a5; font-weight: bold; padding: 5px 12px; border-radius: 12px; border: 1px solid #ef4444; font-size: 11px;")
            self.btn_realtime_launch.setEnabled(False)
            self.guard_banner.setVisible(True)
        return calibrated

    def setup_dashboard_tab(self):
        layout = QVBoxLayout(self.dashboard_tab)
        layout.setContentsMargins(10, 10, 10, 10)

        launcher_group = QFrame()
        launcher_group.setStyleSheet("background-color: #1e293b; border-radius: 8px; border: 1px solid #334155;")
        launch_layout = QHBoxLayout(launcher_group)
        lbl_launch = QLabel("Workflow Modules:")
        lbl_launch.setStyleSheet("font-weight: bold; color: #f8fafc;")
        launch_layout.addWidget(lbl_launch)

        self.btn_realtime_launch = QPushButton("Realtime Video Stream (CLICK ME!)")
        self.btn_realtime_launch.clicked.connect(self.launch_realtime_tab)
        launch_layout.addWidget(self.btn_realtime_launch)
        layout.addWidget(launcher_group)

        kpi_row = QHBoxLayout()
        self.kpi_profiles = KPICard("Calibrated Profiles", "#38bdf8")
        self.kpi_templates = KPICard("Total 128D Vectors", "#4ade80")
        self.kpi_status = KPICard("Gate Lock Status", "#facc15")
        kpi_row.addWidget(self.kpi_profiles)
        kpi_row.addWidget(self.kpi_templates)
        kpi_row.addWidget(self.kpi_status)
        layout.addLayout(kpi_row)

        lbl_table = QLabel("Enrolled CNN Biometric Identities")
        lbl_table.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8; margin-top: 10px;")
        layout.addWidget(lbl_table)

        self.identities_table = QTableWidget(0, 4)
        self.identities_table.setHorizontalHeaderLabels(["ID", "Name / Label", "CNN Vectors Stored", "Threshold"])
        self.identities_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.identities_table, stretch=1)
        self.refresh_dashboard_data()

    def refresh_dashboard_data(self):
        identities = load_identities()
        self.kpi_profiles.set_value(str(len(identities)))
        self.kpi_templates.set_value(str(sum(len(i.get("templates", [])) for i in identities)))
        self.kpi_status.set_value("UNLOCKED" if has_calibrated_profiles() else "LOCKED")

        self.identities_table.setRowCount(len(identities))
        for row, identity in enumerate(identities):
            self.identities_table.setItem(row, 0, QTableWidgetItem(str(identity.get("id", "--"))))
            self.identities_table.setItem(row, 1, QTableWidgetItem(str(identity.get("label", "--"))))
            self.identities_table.setItem(row, 2, QTableWidgetItem(str(len(identity.get("templates", [])))))
            self.identities_table.setItem(row, 3, QTableWidgetItem(str(identity.get("threshold", 0.363))))

    def launch_calibration_wizard(self):
        try:
            dialog = AR3DFaceCalibrationDialog(self)

            dialog.calibration_completed.connect(
                lambda _: (
                    self.check_calibration_status(),
                    self.refresh_dashboard_data(),
                )
            )

            dialog.exec()

        except Exception as e:
            import traceback

            traceback.print_exc()

            QMessageBox.critical(
                self,
                "Calibration Wizard Error",
                f"The CNN configuration wizard could not start.\n\n{e}",
            )

    def launch_realtime_tab(self):
        if not self.check_calibration_status():
            QMessageBox.critical(self, "Access Denied", "Cannot access Realtime Stream without calibrating first!")
            return
        idx = self.tabs.addTab(EmbeddedWebcamTab("realtime", "Live Recognition Stream", self), "🎥 Realtime Stream")
        self.tabs.setCurrentIndex(idx)

    def close_tab(self, index: int):
        if index > 0:
            w = self.tabs.widget(index)
            if hasattr(w, 'stop_feed'): w.stop_feed()
            self.tabs.removeTab(index)

def main() -> int:
    app = QApplication(sys.argv)
    window = SPOTDesktopWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())