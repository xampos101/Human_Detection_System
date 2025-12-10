# Human Detection System 

A human detection and tracking system using YOLOv12, improved re-ID, and camera recording capabilities.

## Features

- ✅ YOLOv12 detection (latest model)
- ✅ ByteTrack for robust tracking
- ✅ Re-identification: maintains ID even when person leaves and re-enters the frame
- ✅ Real-time camera or video upload
- ✅ GUI with Tkinter
- ✅ **CPU/GPU Selection**: Choose device (Auto/CPU/GPU) for better performance
- ✅ Counters: current people & total people who passed
- ✅ Recording in camera mode with hotkeys (R/S)
- ✅ Device info display in GUI and video output
- ✅ Clean architecture

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python demo.py
```

From the GUI you can select:
- **Camera**: Real-time detection from webcam
- **Upload Video**: Upload video file
- **Open Records**: Open the outputs folder

## Project Structure

```
Human Detection/
├── .venv/                    # Virtual environment
├── data/                     # Video files
│   └── people_walking.mp4
├── models/                   # Saved models
├── outputs/                  # Output videos with detections
├── src/
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detect.py        # YOLOv12 detector
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py       # ByteTrack tracker with re-ID
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py       # Helper functions
│   └── demo.py              # GUI & main application
├── README.md
└── requirements.txt
```

## Device Selection (CPU/GPU)

The system supports automatic detection and device selection:

- **Auto-detect**: Automatic selection (GPU if available, otherwise CPU)
- **CPU**: Force CPU usage
- **GPU (CUDA)**: Force GPU usage (if available)

Device selection is done from the GUI before starting detection. The device is displayed in the video output.

## How Re-identification Works

The system uses:
1. **ByteTrack**: For frame-to-frame tracking
2. **Feature Memory**: Stores features for each ID
3. **Similarity Matching**: When a new ID appears, it compares with old IDs
4. **Threshold-based Re-ID**: If similarity > 0.7, reuses the old ID

## Tuning Parameters

In `tracker.py` you can adjust:
- `reid_threshold`: Threshold for re-identification (default: 0.7)
- `max_time_lost`: How many frames to remember an ID (default: 120)
- `track_buffer`: Buffer for ByteTrack (default: 30)

## Requirements

- Python 3.8+
- PyTorch (required, supports both CPU and GPU)
- CUDA (optional, for GPU acceleration - installed via PyTorch)
- Webcam (for real-time mode)

### Installing PyTorch with GPU Support

```bash
# CPU only (default)
pip install torch torchvision

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Troubleshooting

**Issue**: YOLOv12 doesn't download
- Solution: Manually download `yolo12n.pt` to the `models/` folder

**Issue**: Slow performance
- Solution: Select GPU from GUI (if available) or use a smaller model (nano: yolo12n.pt)
- Note: GPU can provide 3-10x better performance than CPU

**Issue**: Camera doesn't open
- Solution: Check if the camera is being used by another application
