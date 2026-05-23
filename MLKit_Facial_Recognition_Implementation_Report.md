# Facial Recognition Implementation Report for triangulated_face_realtime.py

Date: 2026-05-15

## 1. Executive Summary

Your current program already performs a practical form of facial recognition using two backends:
- ArcFace embeddings via InsightFace (preferred when installed)
- A fallback geometric descriptor from facial landmarks

The two ML Kit pages you shared are very useful for detection and mesh tracking design, but they do not provide person recognition by themselves. In particular, Google explicitly states that face detection detects faces but does not recognize people.

Recommended direction:
- Keep your existing recognition pipeline (embeddings + identity matching) in Python.
- Use ML Kit concepts to improve detection robustness, liveness, and mobile deployment architecture.
- If you need true ML Kit usage, use it in a mobile app (Android/iOS) and send face crops or embeddings to your Python service for identity matching.

## 2. What ML Kit Adds (from the linked docs)

### 2.1 Face Detection API

Useful capabilities:
- Bounding boxes, landmarks, contours
- Tracking ID across frames
- Optional face expression signals (smile/eye open)
- Real-time on-device processing

Important limit:
- It does not identify who the person is.

### 2.2 Face Mesh Detection API

Useful capabilities:
- 468 3D points + triangle topology for each face
- Low latency for selfie-range real-time effects
- Strong fit for AR overlays and geometric face analysis

Important limits:
- Beta API (possible breaking changes)
- Best for faces within about 2 meters
- No tracking ID, no face classification, no orientation output (per comparison table)
- Not itself an identity recognition system

## 3. Current Program Assessment

Current script: [triangulated_face_realtime.py](triangulated_face_realtime.py)

### 3.1 What you already do well

- Real-time landmark/mesh extraction and triangle rendering
- Multi-face tracking via centroid-based FaceTracker
- Recognition with normalized embeddings and cosine distance matching
- Enrollment with multiple samples and per-person threshold calibration
- Basic anti-spoofing checks:
  - Blink-based liveness
  - Low-motion optical-flow spoof heuristic

### 3.2 Gaps to close for production-grade facial recognition

- Tracker identity can switch when faces cross or occlude (centroid tracker limitation)
- Spoof detection is still weak against high-quality replay attacks
- Threshold tuning is static and local; no evaluation set is used
- Known faces are stored as plain JSON vectors (no encryption, no audit)
- No confidence calibration dashboard or FAR/FRR monitoring

## 4. Practical Architecture Choices

## Option A (recommended for your current desktop Python app)

Keep detection/mesh in MediaPipe, keep recognition with InsightFace, and harden the pipeline.

Why this is best now:
- Minimal rewrite
- Works directly in your existing Python runtime
- Retains high-quality embeddings (ArcFace)
- Keeps your current triangulation and visualization logic intact

## Option B (if you must use ML Kit directly)

Use ML Kit only on Android/iOS client:
1. ML Kit detects face or mesh and outputs crop/landmarks.
2. App sends crop (or computed embedding) to backend service.
3. Python backend performs recognition against enrolled identities.
4. Backend returns identity + confidence to client.

Why this is needed:
- ML Kit APIs in those pages are mobile SDK APIs, not native desktop Python APIs.

## 5. Implementation Blueprint for Your Current Script

### Phase 1: Stabilize Identity and Tracking

1. Replace centroid-only matching with IoU + landmark-distance hybrid matching.
2. Keep a short trajectory state per face (position, velocity, embedding EMA).
3. Add track timeout and re-identification window.

Expected result:
- Fewer ID switches in crowded or moving scenes.

### Phase 2: Improve Recognition Reliability

1. Collect 15-30 enrollment samples across expression/pose/light variation.
2. Store one prototype vector plus a small sample bank per person.
3. Match rule: identity accepted only if:
   - best cosine distance < person threshold, and
   - margin to second-best identity > delta.
4. Add temporal voting: require N positive matches in last M frames.

Expected result:
- Better precision and lower false accepts.

### Phase 3: Strengthen Liveness and Anti-Spoof

1. Keep blink check, but add challenge response:
   - blink twice
   - turn head left then right
2. Add depth/parallax cue from landmark z movement over 1-2 seconds.
3. Reject if texture/flow is planar and static across most of ROI.

Expected result:
- Better resistance to photo/screen replay attacks.

### Phase 4: Security and Data Governance

1. Encrypt local identity store at rest.
2. Separate metadata from embeddings.
3. Version embedding model and descriptor format.
4. Log enrollment and verification events with timestamps.

Expected result:
- Safer data handling and easier maintenance.

### Phase 5: Evaluation and Threshold Calibration

1. Build a small validation set with genuine and impostor attempts.
2. Compute FAR, FRR, and EER.
3. Choose thresholds by risk target (security-first vs convenience-first).
4. Recalibrate when camera, lighting, or model changes.

Expected result:
- Measured system behavior instead of guesswork.

## 6. Suggested Code Changes in Your File

Target file: [triangulated_face_realtime.py](triangulated_face_realtime.py)

1. Tracking upgrade
- Add track state object with:
  - bbox
  - centroid
  - smoothed embedding
  - last_seen
  - velocity
- Replace nearest-centroid only logic with weighted cost:
  - position distance
  - bbox IoU penalty
  - embedding distance

2. Matching upgrade
- In match_descriptor, add second-best margin check.
- Add per-identity rolling confidence.
- Require consistency across several frames before final accept.

3. Liveness upgrade
- Add explicit challenge state machine:
  - idle -> challenge_requested -> challenge_in_progress -> passed/failed
- Use landmarks to detect left-right head motion in sequence.

4. Data layer upgrade
- Move JSON I/O into a small storage module.
- Add schema version field and migration handler.

## 7. ML Kit to Current Program Mapping

- ML Kit Face Detection tracking ID concept maps to your FaceTracker role.
- ML Kit Face Mesh 468-point geometry maps to your triangulation and geometric descriptor logic.
- ML Kit limitation (no identity recognition) confirms that identity should stay in your embedding matcher.

In short:
- ML Kit can improve face localization and mesh geometry.
- Identity recognition still requires a separate recognition model and database workflow.

## 8. Risk Register

1. False acceptance under low light or motion blur
- Mitigation: temporal vote, margin rule, quality gating

2. Spoof via high-resolution replay
- Mitigation: challenge-response liveness with motion sequence

3. Privacy and compliance concerns
- Mitigation: encryption, retention policy, explicit consent workflow

4. Model drift after updates
- Mitigation: schema versioning, backfill migration, periodic recalibration

## 9. 2-Week Execution Plan

Week 1:
1. Tracking and matcher improvements
2. Liveness challenge flow
3. Logging and schema versioning

Week 2:
1. Enrollment UX polish
2. Validation dataset collection
3. Threshold calibration and report generation (FAR/FRR)

## 10. Final Recommendation

For your current Python desktop project, do not replace your identity recognition with ML Kit alone. Instead:
1. Keep InsightFace embedding recognition as primary.
2. Keep mesh-based geometry as fallback or auxiliary score.
3. Apply ML Kit design patterns (tracking, mesh quality, real-time constraints) to harden your existing pipeline.
4. Use ML Kit directly only if you are building a mobile client tier.
