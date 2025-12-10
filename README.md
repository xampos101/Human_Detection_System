👁️ Human Detection & Tracking SystemAdvanced detection and tracking using YOLOv12, ByteTrack, and persistent Re-Identification.This project implements a robust computer vision system capable of detecting humans in real-time or via video files. It leverages the state-of-the-art YOLOv12 for detection and an enhanced ByteTrack algorithm with memory-based Re-Identification (Re-ID) to maintain identity persistence even when subjects leave and re-enter the frame.🚀 Key FeaturesCore Technology✅ YOLOv12 Detection: Utilizes the latest iteration of YOLO for high-speed, high-accuracy detection.✅ Robust Tracking: Implements ByteTrack for smooth frame-to-frame association.✅ Smart Re-Identification: Maintains specific user IDs even after occlusion or re-entry into the camera frame.✅ CPU/GPU Acceleration: Supports hardware acceleration with automatic or manual device selection (CUDA support).User Interface & Experience✅ GUI Control Panel: User-friendly interface built with Tkinter.✅ Dual Modes: Support for Real-time Webcam and Video File Upload.✅ Analytics: Live counters for current people in frame and total unique people counted.✅ Video Recording: Built-in recording functionality for camera mode (Hotkeys: R to record, S to stop).✅ Status Display: On-screen display (OSD) for active device info and recording status.🛠️ Installation1. Set up the EnvironmentCreate and activate a virtual environment to keep dependencies isolated.Bash# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Linux/MacOS:
source .venv/bin/activate
2. Install DependenciesBashpip install -r requirements.txt
3. GPU Support (Optional but Recommended)For significantly faster performance, install PyTorch with CUDA support:Bash# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
(Default pip install torch installs the CPU-only version).💻 UsageLaunch the application using the main entry point:Bashpython src/demo.py
GUI ControlsUpon launching, you can select your input source:Camera: Starts real-time detection using the default webcam.Upload Video: Opens a file dialog to process a pre-recorded video file.Device Selection: Choose between Auto, CPU, or GPU before starting.Hotkeys (Camera Mode)R: Start RecordingS: Stop RecordingQ: Quit Application📂 Project StructureThe project follows a clean architecture pattern for scalability.PlaintextHuman Detection/
├── .venv/                 # Virtual environment
├── data/                  # Input assets
│   └── people_walking.mp4
├── models/                # Weights for YOLOv12
├── outputs/               # Processed videos and recordings
├── src/                   # Source code
│   ├── detection/
│   │   ├── detect.py      # YOLOv12 detector implementation
│   ├── tracking/
│   │   ├── tracker.py     # ByteTrack with Re-ID logic
│   ├── utils/
│   │   ├── helpers.py     # Utility functions
│   └── demo.py            # Main entry point & GUI
├── README.md
└── requirements.txt
⚙️ Technical DetailsDevice Selection LogicThe system optimizes performance based on available hardware:Auto-detect: Prioritizes GPU (CUDA) if available; falls back to CPU.CPU: Forces the model to run on the processor.GPU: Forces CUDA usage (requires NVIDIA GPU + Drivers).How Re-Identification WorksTo solve the "identity switch" problem when a person leaves the frame, the system uses a 4-step process:ByteTrack: Handles immediate frame-to-frame tracking.Feature Memory: Stores visual feature vectors for every tracked ID.Similarity Matching: When a "new" detection occurs, it compares features against the history of lost IDs.Threshold Logic: If similarity > 0.7, the system restores the old ID instead of assigning a new one.Configuration & TuningYou can fine-tune tracking parameters in src/tracking/tracker.py:ParameterDefaultDescriptionreid_threshold0.7Minimum similarity score required to restore a lost ID.max_time_lost120Number of frames to keep an ID in memory after it disappears.track_buffer30Buffer size for the ByteTrack algorithm.
