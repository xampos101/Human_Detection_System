"""
YOLOv12 Human Detector - Improved Version
Χρησιμοποιεί το πιο πρόσφατο YOLOv12 model για person detection
"""

from ultralytics import YOLO
import numpy as np
from pathlib import Path
import cv2


def get_available_device(device_preference=None):
    """
    Εντοπισμός διαθέσιμου device (CPU/GPU)
    
    Args:
        device_preference: 'cuda', 'cpu', ή None για auto-detection
        
    Returns:
        device string ('cuda', 'cpu', 'mps', etc.)
    """
    try:
        import torch
        
        if device_preference == 'cpu':
            return 'cpu'
        elif device_preference == 'cuda':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                print("⚠️ CUDA requested but not available, falling back to CPU")
                return 'cpu'
        else:
            # Auto-detection
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'  # Apple Silicon
            else:
                return 'cpu'
    except ImportError:
        print("⚠️ PyTorch not installed, using CPU")
        return 'cpu'


class HumanDetector:
    """Detector για ανθρώπους χρησιμοποιώντας YOLOv12"""

    def __init__(self, model_path="models/yolo12n.pt", confidence=0.3, iou_threshold=0.45, device=None):
        """
        Args:
            model_path: Path για το YOLOv12 model
            confidence: Minimum confidence threshold (μειωμένο για καλύτερο detection)
            iou_threshold: IoU threshold για NMS
            device: 'cuda', 'cpu', ή None για auto-detection
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.model_path = Path(model_path)
        
        # Device selection
        self.device = get_available_device(device)
        self.device_name = self._get_device_name()

        # Φόρτωση YOLOv12 model
        print(f"📦 Φόρτωση YOLOv12 model από {model_path}...")
        print(f"🖥️  Device: {self.device_name}")
        self.model = YOLO(model_path)
        
        # Set device για το model
        if self.device != 'cpu':
            try:
                self.model.to(self.device)
            except Exception as e:
                print(f"⚠️ Could not set device to {self.device}: {e}")
                print("   Falling back to CPU")
                self.device = 'cpu'
                self.device_name = "CPU"
        
        print("✅ Model φορτώθηκε επιτυχώς!")

        # COCO dataset: class 0 = person
        self.person_class_id = 0
    
    def _get_device_name(self):
        """Επιστροφή user-friendly device name"""
        try:
            import torch
            if self.device == 'cuda':
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA"
                return f"GPU ({gpu_name})"
            elif self.device == 'mps':
                return "Apple Silicon (MPS)"
            else:
                return "CPU"
        except (ImportError, AttributeError, RuntimeError):
            return "CPU"

    def detect(self, frame):
        """
        Ανίχνευση ανθρώπων σε ένα frame

        Args:
            frame: Input frame (numpy array)

        Returns:
            detections: numpy array of [x1, y1, x2, y2, confidence]
        """
        # Τρέχουμε το YOLOv12 model με βελτιωμένες παραμέτρους
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
            classes=[0],  # Μόνο person class
            device=self.device  # Explicit device specification
        )

        detections = []

        # Εξαγωγή detections μόνο για ανθρώπους (class 0)
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Έλεγχος αν είναι person
                if int(box.cls) == self.person_class_id:
                    # Συντεταγμένες bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])

                    # Φιλτράρισμα πολύ μικρών boxes (πιθανά false positives)
                    width = x2 - x1
                    height = y2 - y1

                    if width > 20 and height > 40:  # Minimum ανθρώπινο μέγεθος
                        detections.append([x1, y1, x2, y2, conf])

        return np.array(detections) if detections else np.empty((0, 5))

    def extract_appearance_features(self, frame, box):
        """
        Εξαγωγή appearance features από bounding box
        Χρησιμοποιεί color histogram & spatial features

        Args:
            frame: Input frame
            box: Bounding box [x1, y1, x2, y2, conf]

        Returns:
            feature_vector: numpy array
        """
        x1, y1, x2, y2 = map(int, box[:4])

        # Crop το άτομο
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return np.zeros(128)  # Empty feature

        # Resize για consistency
        try:
            person_crop = cv2.resize(person_crop, (64, 128))
        except (cv2.error, ValueError, AttributeError):
            return np.zeros(128)

        # Color histogram features (HSV)
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)

        # Χωρισμός σε πάνω/κάτω μέρος (torso/legs)
        h = person_crop.shape[0]
        top_half = hsv[:h//2, :]
        bottom_half = hsv[h//2:, :]

        # Histograms
        hist_top_h = cv2.calcHist([top_half], [0], None, [16], [0, 180])
        hist_top_s = cv2.calcHist([top_half], [1], None, [8], [0, 256])
        hist_bottom_h = cv2.calcHist([bottom_half], [0], None, [16], [0, 180])
        hist_bottom_s = cv2.calcHist([bottom_half], [1], None, [8], [0, 256])

        # Normalize
        hist_top_h = cv2.normalize(hist_top_h, hist_top_h).flatten()
        hist_top_s = cv2.normalize(hist_top_s, hist_top_s).flatten()
        hist_bottom_h = cv2.normalize(hist_bottom_h, hist_bottom_h).flatten()
        hist_bottom_s = cv2.normalize(hist_bottom_s, hist_bottom_s).flatten()

        # Aspect ratio feature
        aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0

        # Concatenate όλα τα features
        features = np.concatenate([
            hist_top_h,      # 16
            hist_top_s,      # 8
            hist_bottom_h,   # 16
            hist_bottom_s,   # 8
            [aspect_ratio]   # 1
        ])

        # Pad to 128 dimensions
        if len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)))

        return features[:128]

    def detect_batch(self, frames):
        """
        Batch detection για πολλαπλά frames (πιο γρήγορο)

        Args:
            frames: List of frames

        Returns:
            List of detections για κάθε frame
        """
        results = self.model(
            frames,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
            classes=[0],
            device=self.device  # Explicit device specification
        )

        all_detections = []
        for result in results:
            detections = []
            boxes = result.boxes

            for box in boxes:
                if int(box.cls) == self.person_class_id:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])

                    # Φιλτράρισμα μικρών boxes
                    width = x2 - x1
                    height = y2 - y1

                    if width > 20 and height > 40:
                        detections.append([x1, y1, x2, y2, conf])

            all_detections.append(
                np.array(detections) if detections else np.empty((0, 5))
            )

        return all_detections