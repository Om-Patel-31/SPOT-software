"""
Gemini-powered auto-training system for facial recognition.

Features:
- Real-time auto-labeling of detected faces using Gemini
- Auto-enrollment of high-confidence faces
- User confirmation prompts for auto-enrolled faces
- Confidence-based retraining triggers
- Audit logging of all auto-training decisions

Usage:
    from gemini_auto_trainer import GeminiAutoTrainer
    
    trainer = GeminiAutoTrainer(
        api_key="your-gemini-api-key",
        auto_enroll_threshold=0.85,
        require_user_confirmation=True
    )
    
    # During real-time detection:
    decision = trainer.process_detected_face(
        frame_bgr=frame,
        face_descriptor=descriptor,
        current_match=(matched_identity, confidence),
        identities=identities
    )
    
    if decision["action"] == "enroll":
        # Add to model automatically or with user confirmation
        trainer.enroll_face(decision)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import importlib
import warnings

import cv2
import numpy as np

from .paths import model_path

try:
    from PIL import Image
except Exception:
    Image = None

# Try to import Gemini clients
try:
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
    genai = importlib.import_module("google.genai")
except Exception:
    try:
        genai = importlib.import_module("google.generativeai")
    except Exception:
        genai = None


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiAutoTrainer:
    """Auto-trainer that uses Gemini to suggest labels and confidence thresholds."""

    def __init__(
        self,
        api_key: str,
        identities_path: Optional[Path] = None,
        auto_enroll_threshold: float = 0.85,
        confidence_retraining_threshold: float = 0.65,
        require_user_confirmation: bool = True,
        model_name: str = "gemini-2.0-flash",
        audit_log_path: Optional[Path] = None,
    ):
        """
        Initialize the auto-trainer.
        
        Args:
            api_key: Gemini API key
            identities_path: Path to face_identities.json
            auto_enroll_threshold: Confidence threshold for auto-enrollment (0-1)
            confidence_retraining_threshold: Threshold for suggesting retraining
            require_user_confirmation: Whether to require user confirmation for auto-enrollments
            model_name: Gemini model to use
            audit_log_path: Path to audit log file
        """
        if genai is None or Image is None:
            raise RuntimeError("google-genai (or google.generativeai) and pillow are required")

        self.api_key = api_key
        self.identities_path = identities_path or model_path("face_identities.json")
        self.auto_enroll_threshold = auto_enroll_threshold
        self.confidence_retraining_threshold = confidence_retraining_threshold
        self.require_user_confirmation = require_user_confirmation
        self.model_name = model_name
        self._model = None
        self._sdk_mode = "unknown"
        self._last_error_log_at = 0.0
        self._gemini_disabled_until = 0.0
        self._model_candidates = self._build_model_candidates(model_name)
        
        # Set up audit logging
        self.audit_log_path = audit_log_path or model_path("auto_training_audit.log")
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize Gemini client for either SDK variant.
        # New SDK: google.genai.Client
        # Legacy SDK: google.generativeai.GenerativeModel
        if hasattr(genai, "Client"):
            self._client = genai.Client(api_key=api_key)
            self._sdk_mode = "google_genai"
        elif hasattr(genai, "GenerativeModel"):
            if hasattr(genai, "configure"):
                genai.configure(api_key=api_key)
            self._client = genai
            self._model = genai.GenerativeModel(model_name)
            self._sdk_mode = "google_generativeai"
        else:
            self._client = genai
            self._sdk_mode = "unknown"

        self._pending_confirmations: Dict[str, Dict] = {}  # temp_id -> decision data
        
        logger.info(
            "GeminiAutoTrainer initialized with threshold=%s using sdk=%s",
            auto_enroll_threshold,
            self._sdk_mode,
        )
        logger.info("Gemini model candidates: %s", ", ".join(self._model_candidates))

    def _build_model_candidates(self, preferred_model: str) -> List[str]:
        """Build ordered fallback model candidates, starting with user preference."""
        defaults = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
        ]
        ordered = [preferred_model] + defaults
        deduped: List[str] = []
        for name in ordered:
            n = str(name or "").strip()
            if n and n not in deduped:
                deduped.append(n)
        return deduped

    def _set_active_model(self, model_name: str) -> None:
        """Switch active model and rebuild legacy SDK model object if needed."""
        self.model_name = model_name
        if self._sdk_mode == "google_generativeai":
            self._model = self._client.GenerativeModel(model_name)

    def _is_not_found_error(self, err: Exception) -> bool:
        msg = str(err)
        return "NOT_FOUND" in msg or "404" in msg

    def _throttled_error(self, msg: str, interval_sec: float = 5.0) -> None:
        now = time.time()
        if now - self._last_error_log_at >= interval_sec:
            logger.error(msg)
            self._last_error_log_at = now

    def _try_next_model_candidate(self) -> bool:
        """Move to next fallback model. Returns True if switched, else False."""
        if self.model_name not in self._model_candidates:
            self._model_candidates.insert(0, self.model_name)

        try:
            idx = self._model_candidates.index(self.model_name)
        except ValueError:
            idx = -1

        next_idx = idx + 1
        if 0 <= next_idx < len(self._model_candidates):
            new_model = self._model_candidates[next_idx]
            self._set_active_model(new_model)
            logger.warning("Switching Gemini model fallback to '%s'", new_model)
            return True
        return False

    def _extract_json_text(self, text: str) -> str:
        """Extract JSON object text even if wrapped with extra prose or markdown."""
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw[start : end + 1]
        return raw

    def _log_audit(self, action: str, details: Dict) -> None:
        """Log all auto-training decisions for auditability."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            **details
        }
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"Audit: {action} - {details}")

    def _get_gemini_suggestion(
        self,
        frame_bgr: np.ndarray,
        identities: List[Dict],
        current_match: Optional[Tuple[str, float]] = None,
    ) -> Dict:
        """Get Gemini's label and confidence suggestion for a face."""
        if time.time() < self._gemini_disabled_until:
            return {"label": "UNKNOWN", "confidence": 0.0, "reason": "gemini_temporarily_disabled"}

        if frame_bgr.size == 0:
            return {"label": "UNKNOWN", "confidence": 0.0, "reason": "empty_frame"}

        labels = [
            str(item.get("label", "")).strip()
            for item in identities
            if str(item.get("label", "")).strip()
        ]
        labels_text = ", ".join(labels) if labels else "none"
        if not labels:
            return {"label": "UNKNOWN", "confidence": 0.0, "reason": "no_known_identities"}
        
        current_match_text = ""
        if current_match:
            matched_label, matched_conf = current_match
            current_match_text = f"\nCurrent local match: {matched_label} (confidence: {matched_conf:.2f})"

        prompt = (
            "You are helping a facial recognition system auto-label detected faces. "
            "Analyze the face image and choose the most likely identity from the provided list, "
            "or UNKNOWN if you're uncertain or the face doesn't match any known identity.\n\n"
            f"Available identities: {labels_text}"
            f"{current_match_text}\n\n"
            "Respond ONLY with valid JSON (no markdown, no code blocks):\n"
            '{"label": "PERSON_NAME", "confidence": 0.95, "reasoning": "why you chose this"}\n'
            "Keys must be: label (string), confidence (0-1 float), reasoning (string)."
        )

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Try the current model, and if it's missing, auto-fallback to another candidate.
        for _ in range(2):
            try:
                if self._sdk_mode == "google_genai":
                    try:
                        resp = self._client.models.generate_content(
                            model=self.model_name,
                            contents=[prompt, pil_image],
                            config={"response_mime_type": "application/json"},
                        )
                    except TypeError:
                        resp = self._client.models.generate_content(
                            model=self.model_name,
                            contents=[prompt, pil_image],
                        )
                    text = getattr(resp, "text", "") or ""
                elif self._sdk_mode == "google_generativeai":
                    try:
                        resp = self._model.generate_content(
                            [prompt, pil_image],
                            generation_config={"response_mime_type": "application/json"},
                        )
                    except TypeError:
                        resp = self._model.generate_content([prompt, pil_image])
                    text = getattr(resp, "text", "") or ""
                else:
                    return {"label": "UNKNOWN", "confidence": 0.0, "reason": "client_api_unavailable"}

                # Parse JSON response
                result = json.loads(self._extract_json_text(str(text)))
            
                # Validate response
                if not isinstance(result, dict):
                    return {"label": "UNKNOWN", "confidence": 0.0, "reason": "invalid_response_format"}
            
                label = str(result.get("label", "UNKNOWN")).strip()
                confidence = float(result.get("confidence", 0.0))
                reasoning = str(result.get("reasoning", ""))

                # Enforce closed-set labels to avoid unexpected identity creation from hallucinated names.
                if label != "UNKNOWN" and label not in labels:
                    return {
                        "label": "UNKNOWN",
                        "confidence": 0.0,
                        "reason": "label_not_in_known_identities",
                    }
            
                # Clamp confidence
                confidence = max(0.0, min(1.0, confidence))
            
                return {
                    "label": label,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "reason": "gemini_suggestion"
                }

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Gemini JSON response: {e}")
                return {"label": "UNKNOWN", "confidence": 0.0, "reason": "json_parse_error"}
            except Exception as e:
                if self._is_not_found_error(e):
                    switched = self._try_next_model_candidate()
                    if switched:
                        continue
                    self._gemini_disabled_until = time.time() + 30.0
                    self._throttled_error(
                        "Gemini model unavailable for all candidates. Temporarily disabling API calls for 30s.",
                        interval_sec=10.0,
                    )
                    return {"label": "UNKNOWN", "confidence": 0.0, "reason": "no_supported_model_available"}

                self._throttled_error(f"Gemini API error: {e}")
                return {"label": "UNKNOWN", "confidence": 0.0, "reason": f"api_error: {str(e)}"}

        return {"label": "UNKNOWN", "confidence": 0.0, "reason": "fallback_exhausted"}

    def process_detected_face(
        self,
        frame_bgr: np.ndarray,
        face_descriptor: List[float],
        current_match: Optional[Tuple[str, float]],
        identities: List[Dict],
    ) -> Dict:
        """
        Process a detected face and decide on auto-training action.
        
        Args:
            frame_bgr: Face crop (BGR)
            face_descriptor: Face embedding vector
            current_match: (label, confidence) from local model or None
            identities: List of known identities
            
        Returns:
            Decision dict with keys:
            - action: "skip", "flag_retraining", "enroll_pending", "enroll_confident"
            - label: Suggested identity
            - confidence: Gemini's confidence
            - reasoning: Why this decision was made
            - frame_descriptor: For later enrollment
            - temp_id: For user confirmation tracking
        """
        # Get Gemini's suggestion
        gemini_suggestion = self._get_gemini_suggestion(
            frame_bgr, identities, current_match
        )
        
        gemini_label = gemini_suggestion["label"]
        gemini_confidence = gemini_suggestion["confidence"]
        
        # Determine action
        action = "skip"
        reasoning = ""
        
        if gemini_label == "UNKNOWN":
            reasoning = "Gemini marked as UNKNOWN"
            action = "skip"
        elif gemini_confidence >= self.auto_enroll_threshold:
            reasoning = f"High confidence ({gemini_confidence:.2f}) - ready for auto-enrollment"
            action = "enroll_confident" if not self.require_user_confirmation else "enroll_pending"
        elif gemini_confidence >= self.confidence_retraining_threshold:
            if current_match and current_match[0] != gemini_label:
                reasoning = f"Conflict: local={current_match[0]}, gemini={gemini_label} - flag for retraining"
                action = "flag_retraining"
            else:
                reasoning = f"Moderate confidence ({gemini_confidence:.2f}) - monitor for patterns"
                action = "skip"
        else:
            reasoning = f"Low confidence ({gemini_confidence:.2f}) - skip"
            action = "skip"
        
        # Create decision record
        import uuid
        temp_id = str(uuid.uuid4())[:8]
        
        decision = {
            "action": action,
            "label": gemini_label,
            "confidence": gemini_confidence,
            "reasoning": reasoning,
            "frame_descriptor": face_descriptor,
            "frame_bgr": frame_bgr.copy(),  # For display/confirmation
            "gemini_reasoning": gemini_suggestion.get("reasoning", ""),
            "temp_id": temp_id,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Store pending confirmation if needed
        if action == "enroll_pending":
            self._pending_confirmations[temp_id] = decision
            self._log_audit("enroll_pending", {
                "temp_id": temp_id,
                "label": gemini_label,
                "confidence": gemini_confidence,
            })
        elif action == "enroll_confident":
            self._log_audit("auto_enroll", {
                "temp_id": temp_id,
                "label": gemini_label,
                "confidence": gemini_confidence,
            })
        elif action == "flag_retraining":
            self._log_audit("flag_retraining", {
                "label": gemini_label,
                "gemini_confidence": gemini_confidence,
                "local_match": current_match,
                "reasoning": reasoning,
            })
        
        return decision

    def enroll_face(self, decision: Dict, identities: Optional[List[Dict]] = None) -> bool:
        """
        Enroll a face into the identity model.
        
        Args:
            decision: Decision dict from process_detected_face or manual enrollment
            identities: Current identities list (loads from file if not provided)
            
        Returns:
            True if enrollment succeeded
        """
        if identities is None:
            identities = self._load_identities()
        
        label = decision["label"]
        descriptor = decision["frame_descriptor"]
        temp_id = decision.get("temp_id", "manual")
        
        # Find or create identity
        identity = None
        for item in identities:
            if str(item.get("label", "")).strip() == label:
                identity = item
                break
        
        if identity is None:
            # Create new identity
            new_id = max((int(item.get("id", 0) or 0) for item in identities), default=0) + 1
            identity = {
                "id": new_id,
                "version": 1,
                "label": label,
                "templates": [],
                "threshold": 0.22,
                "pose_counts": {},
            }
            identities.append(identity)
            self._log_audit("new_identity_created", {"label": label, "id": new_id})
        
        # Add template
        identity["templates"].append(descriptor)
        identity["version"] = identity.get("version", 1) + 1
        
        # Save updated identities
        self._save_identities(identities)
        
        self._log_audit("face_enrolled", {
            "temp_id": temp_id,
            "label": label,
            "identity_id": identity["id"],
            "total_templates": len(identity["templates"]),
        })
        
        logger.info(f"Enrolled face for '{label}' (templates: {len(identity['templates'])})")
        return True

    def confirm_pending(self, temp_id: str, approved: bool, identities: Optional[List[Dict]] = None) -> bool:
        """
        Confirm or reject a pending auto-enrollment.
        
        Args:
            temp_id: Temp ID from decision dict
            approved: True to enroll, False to skip
            identities: Current identities list
            
        Returns:
            True if operation succeeded
        """
        if temp_id not in self._pending_confirmations:
            logger.warning(f"Unknown temp_id: {temp_id}")
            return False
        
        decision = self._pending_confirmations.pop(temp_id)
        
        if approved:
            self.enroll_face(decision, identities)
            logger.info(f"Confirmed enrollment for {decision['label']}")
        else:
            self._log_audit("enrollment_rejected", {
                "temp_id": temp_id,
                "label": decision["label"],
            })
            logger.info(f"Rejected enrollment for {decision['label']}")
        
        return True

    def _load_identities(self) -> List[Dict]:
        """Load identities from file."""
        if not self.identities_path.exists():
            return []
        try:
            with open(self.identities_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading identities: {e}")
            return []

    def _save_identities(self, identities: List[Dict]) -> None:
        """Save identities to file."""
        self.identities_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.identities_path, "w", encoding="utf-8") as f:
            json.dump(identities, f, indent=2)

    def get_pending_confirmations(self) -> Dict[str, Dict]:
        """Get all pending confirmations awaiting user approval."""
        return self._pending_confirmations.copy()

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Get recent audit log entries."""
        entries = []
        if not self.audit_log_path.exists():
            return entries
        
        try:
            with open(self.audit_log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-limit:]
                for line in lines:
                    entries.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Error reading audit log: {e}")
        return entries
