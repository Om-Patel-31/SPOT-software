# SPOT Windows Release

SPOT is a Windows-friendly face recognition demo that runs from a webcam and supports simple enrollment.

## Requirements
- Windows 10 or newer
- A working webcam
- Internet access for the first run so the MediaPipe face model can be downloaded
- A modern CPU; GPU is optional

## First-run notes
- Windows SmartScreen may warn about the executable. If prompted, choose "More info" and then "Run anyway".
- The first launch may take a moment while the face-landmarker model downloads into the app folder.
- If you want to use Gemini suggestions, set the GEMINI_API_KEY environment variable before launching.

## Controls
- Q: quit
- E: start enrollment
- Space: capture an enrollment frame
- S: save the enrolled identity
- C: cancel enrollment
