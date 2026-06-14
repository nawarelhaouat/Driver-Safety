# Driver Safety Monitoring System

Real-time driver monitoring system for detecting **drowsiness**, **yawning**, and **driver distraction** using computer vision.

This project was developed as part of the **Image Processing** module at **ENSA Tanger** during the academic year **2025–2026**.

## Academic Context

* **Project title:** Driver Safety Monitoring System
* **Topic:** Real-time detection of driver drowsiness and distraction using computer vision
* **Developed by:** Nawar El Haouat & Nada El Oukili
* **Supervised by:** Prof. Abdelmonaime LACHKAR
* **Institution:** ENSA Tanger
* **Academic year:** 2025–2026

## Project Overview

Road safety is a major challenge, and many accidents are caused by human factors such as fatigue, drowsiness, and distraction. This project proposes a low-cost computer vision solution that monitors the driver through a webcam and triggers alerts when dangerous behaviors are detected.

The system is composed of two complementary modules:

1. **Drowsiness Detection**

   * Detects eye closure in real time.
   * Uses the Eye Aspect Ratio (EAR).
   * Combines geometric analysis with a CNN model trained using PyTorch.
   * Triggers an alarm when the eyes remain closed for several consecutive frames.

2. **Distraction and Yawning Detection**

   * Detects yawning using the Mouth Aspect Ratio (MAR).
   * Detects head distraction using Head Pose Estimation.
   * Uses OpenCV `solvePnP` to estimate head orientation.
   * Triggers alerts when the driver looks away from the road for too long.

## Main Features

* Real-time webcam processing
* Face landmark detection using MediaPipe Face Mesh
* Eye closure detection using EAR
* CNN-based eye state classification
* Yawning detection using MAR
* Head pose estimation using yaw, pitch, and roll angles
* Visual status display on the video frame
* Sound alert system
* CPU-compatible implementation
* Modular Python project structure

## Technologies Used

| Technology   | Role                                                  |
| ------------ | ----------------------------------------------------- |
| Python       | Main programming language                             |
| OpenCV       | Webcam capture, image processing, display, `solvePnP` |
| MediaPipe    | Face landmark detection                               |
| PyTorch      | CNN training and inference                            |
| NumPy        | Numerical calculations                                |
| pygame-ce    | Alarm sound management                                |
| scikit-learn | Dataset splitting and evaluation metrics              |

## Project Structure

```text
DrowsyDriverSafetyAlertSystem/
├── main.py                      # Main entry point and real-time loop
├── collect_data.py              # Collects training images
├── train_model.py               # Trains the CNN model
├── evaluate_model.py            # Evaluates the trained model
├── generate_alarm.py            # Generates the alarm audio file
├── assets/
│   └── alarm.wav                # Alarm sound
├── data/
│   ├── awake/                   # Images of open eyes
│   └── drowsy/                  # Images of closed eyes
├── models/
│   └── eye_cnn.pth              # Trained CNN model
├── config/
│   └── settings.py              # Project constants and thresholds
└── src/
    ├── face_detector.py         # Face and landmarks detection
    ├── eye_detector.py          # Eye extraction and EAR calculation
    ├── drowsiness_detector.py   # Drowsiness decision logic
    ├── alarm_manager.py         # Alarm sound control
    └── utils.py                 # Display and helper functions
```

## How the System Works

### 1. Drowsiness Detection

For each webcam frame:

1. The frame is captured using OpenCV.
2. MediaPipe Face Mesh detects facial landmarks.
3. Six landmarks are selected for each eye.
4. The Eye Aspect Ratio is calculated.
5. The eye region is extracted and passed to a CNN model.
6. If both eyes are detected as closed for several consecutive frames, the system triggers a drowsiness alert.

The EAR is calculated as:

```text
EAR = ( ||P2 - P6|| + ||P3 - P5|| ) / ( 2 × ||P1 - P4|| )
```

### 2. Yawning Detection

The system calculates the Mouth Aspect Ratio (MAR) using selected mouth landmarks.

```text
MAR = ( ||M2 - M6|| + ||M3 - M5|| ) / ( 2 × ||M1 - M4|| )
```

A high MAR value over several frames indicates that the mouth is widely open, which may correspond to yawning.

### 3. Distraction Detection

The distraction module estimates the head orientation using OpenCV `solvePnP`.

The system analyzes:

* **Yaw:** left/right head rotation
* **Pitch:** up/down head movement
* **Roll:** lateral head tilt

If the yaw or pitch angle exceeds the configured threshold for several consecutive frames, the system displays a **DISTRACTED** alert.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nawarelhaouat/Driver-Safety.git
cd Driver-Safety
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available:

```bash
pip install opencv-python mediapipe torch torchvision numpy pygame-ce scikit-learn
```

## Usage

Run the real-time monitoring system:

```bash
python main.py
```

## Keyboard Controls

| Key | Action                       |
| --- | ---------------------------- |
| `Q` | Quit the application         |
| `R` | Reset the drowsiness counter |

## Training the CNN Model

### 1. Collect training data

```bash
python collect_data.py
```

Collect images for both classes:

* `awake`: eyes open
* `drowsy`: eyes closed

### 2. Train the model

```bash
python train_model.py
```

The trained model will be saved in:

```text
models/eye_cnn.pth
```

### 3. Evaluate the model

```bash
python evaluate_model.py
```

## Configuration

Adjust thresholds and paths in:

```text
config/settings.py
```

Typical configurable parameters include:

* EAR threshold
* MAR threshold
* Number of consecutive frames before alert
* Webcam index
* Alarm file path
* Head pose thresholds

## Results

The system was tested in real time using a standard laptop without a dedicated GPU. The drowsiness module successfully detected eye closure and triggered the alarm after the eyes remained closed for several consecutive frames.

During training, the CNN achieved very high accuracy on the collected dataset. However, the dataset was limited because it was collected from a small number of conditions. Therefore, the model is suitable for demonstration but requires more diverse data before real deployment.

## Limitations

* The CNN was trained on a limited dataset.
* Performance may decrease in low-light conditions.
* Strong head rotation can affect landmark detection.
* Sunglasses may reduce eye and mouth landmark accuracy.
* Static thresholds may not work equally well for all users.
* The system is currently a prototype and not a certified safety device.

## Future Improvements

* Add automatic calibration for EAR, MAR, yaw, and pitch thresholds.
* Train the CNN on public datasets such as CEW or NTHU.
* Use a larger and more diverse dataset with different users, lighting conditions, and face angles.
* Merge both modules into a single optimized pipeline.
* Deploy the system on an embedded device such as Raspberry Pi.
* Test the system inside a real vehicle environment.

## Repository

```text
https://github.com/nawarelhaouat/Driver-Safety
```

## Authors

* Nawar El Haouat
* Nada El Oukili

## Supervisor

* Prof. Abdelmonaim LACHKAR

## Institution

* École Nationale des Sciences Appliquées de Tanger — ENSA Tanger

## License

This project is intended for academic and educational use.
