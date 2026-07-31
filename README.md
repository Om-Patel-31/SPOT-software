# SPOT Computer Vision Suite

SPOT Computer Vision Suite is a desktop vision hub built for SPOT, a quadruped robot dog. While desktop face recognition might look simple on the surface, real-time facial identification and head pose tracking are essential for a robot dog to safely identify humans and interact naturally in physical environments.

## Robot Dog CAD Photos

![1784934523298](image/README/1784934523298.png)

![1784934535699](image/README/1784934535699.png)

## Features

* Mandatory calibration gate that locks the live video feed until at least one identity is enrolled.
* Interactive 3-step calibration wizard (Center, Left, Right) using real-time yaw estimation and visual AR overlays.
* Deep feature extraction using OpenCV YuNet face detection and SFace 128-dimensional neural network embeddings.
* Dynamic dependency checking and automatic model file provisioning.

## Run the shipped application

Download and extract `SPOT_Suite_Windows_v1.0.zip` from the Releases page, then run `SPOT_Suite.exe`.

The application is packaged as a standalone executable and does not require Python to be installed.

### First Launch

On the first launch, the application automatically downloads the required OpenCV YuNet and SFace ONNX model files if they are not already present.

Because of this, the first launch requires:

* An active internet connection
* Permission to create the `data/models` folder beside the executable
* A connected webcam for the calibration wizard

The initial download only happens once. After the models have been downloaded, the application can be used offline.

If the calibration wizard cannot start, verify that:

* A webcam is connected and accessible by Windows.
* The application has permission to write to the installation folder.
* The initial model download completed successfully.

## Build from source

Requirements:

* Python 3.9+
* Windows 10/11
* Connected webcam
* Internet connection on first run for automatic model download

```bash
git clone https://github.com/your-username/spot-vision-suite.git
cd spot-vision-suite
python main.py
```

## Building the executable

Install PyInstaller:

```bash
pip install pyinstaller
```

Build a single executable:

```bash
pyinstaller --noconfirm --onefile --windowed --name "SPOT_Suite" main.py
```

The executable will be generated in the `dist` folder.

## Development History and Iterations

This project went through six major iterations to reach this release. Because I was working on this entirely offline without an internet connection at the time, intermediate commits could not be pushed to GitHub as development progressed.

Earlier versions gave the operator manual control over individual computer vision parameters, tracking subroutines, and detection thresholds. While that gave granular control, it introduced too much friction when operating a physical robot dog.

This current version reduces those manual controls to keep the system streamlined while retaining an interactive AR calibration process so the operator still feels connected to the robot's setup. The recognition backend was also upgraded from basic geometric landmark matching to 128-dimensional deep feature embeddings (YuNet and SFace) for more dependable human recognition.

## Project Layout

* `main.py` — application source code and GUI
* `data/models/` — downloaded ONNX models and enrolled identity profiles
