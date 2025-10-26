"""
Helper functions για visualization και utilities
"""

import cv2
import numpy as np
from datetime import datetime
import random

# Χρώματα για visualization
COLORS=[(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for i in range(10)]



def get_color_for_id(track_id):
    """Επιστροφή consistent χρώματος για κάθε track ID"""
    return COLORS[track_id % len(COLORS)]


def draw_tracks(frame, tracks, stats):
    """
    Σχεδιασμός bounding boxes και IDs στο frame

    Args:
        frame: Input frame
        tracks: List of (track_id, box) tuples
        stats: Dictionary με statistics

    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()

    # Σχεδιασμός κάθε track
    for track_id, box in tracks:
        x1, y1, x2, y2 = map(int, box[:4])
        color = get_color_for_id(track_id)

        # Bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # ID label
        label = f"ID: {track_id}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Background για το label
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 10, y1),
            color,
            -1
        )

        # Label text
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # Στατιστικά panel
    panel_height = 100
    panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    # Current people
    cv2.putText(
        panel,
        f"Current People: {stats['current_people']}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    # Total people
    cv2.putText(
        panel,
        f"Total People: {stats['total_people']}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 0),
        2
    )

    # FPS (προαιρετικό, μπορείς να το προσθέσεις αργότερα)

    # Συνδυασμός frame με panel
    result = np.vstack([annotated_frame, panel])

    return result


def resize_frame(frame, max_width=1280, max_height=720):
    """
    Resize frame διατηρώντας το aspect ratio

    Args:
        frame: Input frame
        max_width: Maximum width
        max_height: Maximum height

    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]

    # Υπολογισμός scaling factor
    scale = min(max_width / w, max_height / h, 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return frame


def create_output_path(input_path, output_dir="outputs"):
    """
    Δημιουργία output path για processed video

    Args:
        input_path: Path του input video
        output_dir: Directory για outputs

    Returns:
        Output path
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_name = Path(input_path).stem

    output_path = output_dir / f"{input_name}_tracked_{timestamp}.mp4"

    return str(output_path)


def format_time(seconds):
    """
    Μετατροπή seconds σε readable format

    Args:
        seconds: Αριθμός δευτερολέπτων

    Returns:
        Formatted string (π.χ. "2:30")
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def calculate_fps(frame_count, elapsed_time):
    """
    Υπολογισμός FPS

    Args:
        frame_count: Αριθμός frames
        elapsed_time: Χρόνος σε δευτερόλεπτα

    Returns:
        FPS
    """
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0.0