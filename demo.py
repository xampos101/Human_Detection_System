"""
GUI Application για Human Detection & Tracking - Improved Version
Με REC button για camera recording
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from pathlib import Path
import threading
import time
from datetime import datetime

from src.detection.detect import HumanDetector
from src.tracking.tracker import HumanTracker
from src.utils.helpers import draw_tracks, resize_frame, create_output_path


class HumanDetectionApp:
    """Κύρια εφαρμογή με GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("Human Detection & Tracking System")
        self.root.geometry("550x550")
        self.root.resizable(False, False)

        # State
        self.is_running = False
        self.is_recording = False
        self.video_writer = None
        self.detector = None
        self.tracker = None
        self.device_preference = tk.StringVar(value="auto")  # auto, cpu, cuda

        self._setup_ui()

    def _setup_ui(self):
        """Δημιουργία UI elements"""
        # Title
        title_label = tk.Label(
            self.root,
            text="🎥 Human Detection & Tracking",
            font=("Arial", 20, "bold"),
            fg="#2c3e50"
        )
        title_label.pack(pady=30)

        # Subtitle
        subtitle = tk.Label(
            self.root,
            text="Powered by YOLOv12 | Real-time Tracking with Re-ID",
            font=("Arial", 10),
            fg="#7f8c8d"
        )
        subtitle.pack()

        # Device selection frame
        device_frame = tk.Frame(self.root)
        device_frame.pack(pady=20)
        
        device_label = tk.Label(
            device_frame,
            text="Device:",
            font=("Arial", 11, "bold"),
            fg="#2c3e50"
        )
        device_label.pack(side=tk.LEFT, padx=5)
        
        # Device options
        device_options = [("Auto-detect", "auto"), ("CPU", "cpu"), ("GPU (CUDA)", "cuda")]
        
        for text, value in device_options:
            rb = tk.Radiobutton(
                device_frame,
                text=text,
                variable=self.device_preference,
                value=value,
                font=("Arial", 10),
                fg="#34495e",
                activebackground="#ecf0f1",
                command=self._update_device_info
            )
            rb.pack(side=tk.LEFT, padx=5)
        
        # Device info label
        self.device_info_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 9),
            fg="#95a5a6"
        )
        self.device_info_label.pack(pady=5)
        self._update_device_info()

        # Frame για buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=30)

        # Camera button
        camera_btn = tk.Button(
            button_frame,
            text="📹 Real-time Camera",
            font=("Arial", 14),
            bg="#3498db",
            fg="white",
            width=20,
            height=2,
            command=self._start_camera,
            cursor="hand2"
        )
        camera_btn.pack(pady=10)

        # Upload video button
        upload_btn = tk.Button(
            button_frame,
            text="📁 Upload Video",
            font=("Arial", 14),
            bg="#2ecc71",
            fg="white",
            width=20,
            height=2,
            command=self._upload_video,
            cursor="hand2"
        )
        upload_btn.pack(pady=10)

        # Open Records button
        records_btn = tk.Button(
            button_frame,
            text="📂 Open Records",
            font=("Arial", 14),
            bg="#9b59b6",
            fg="white",
            width=20,
            height=2,
            command=self._open_outputs_folder,
            cursor="hand2"
        )
        records_btn.pack(pady=10)

        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Έτοιμο",
            font=("Arial", 10),
            fg="#95a5a6"
        )
        self.status_label.pack(side=tk.BOTTOM, pady=20)

        # Info
        info_label = tk.Label(
            self.root,
            text="Camera Mode: Πάτα 'R' για REC, 'S' για Stop REC, ESC για έξοδο",
            font=("Arial", 9),
            fg="#95a5a6"
        )
        info_label.pack(side=tk.BOTTOM, pady=5)
    
    def _update_device_info(self):
        """Ενημέρωση device info label"""
        try:
            from src.detection.detect import get_available_device
            device = get_available_device(self.device_preference.get())
            
            if device == 'cuda':
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        self.device_info_label.config(
                            text=f"✓ GPU Available: {gpu_name}",
                            fg="#27ae60"
                        )
                    else:
                        self.device_info_label.config(
                            text="⚠ GPU requested but not available",
                            fg="#e74c3c"
                        )
                except:
                    self.device_info_label.config(
                        text="⚠ GPU requested but PyTorch not available",
                        fg="#e74c3c"
                    )
            elif device == 'mps':
                self.device_info_label.config(
                    text="✓ Apple Silicon (MPS) Available",
                    fg="#27ae60"
                )
            else:
                self.device_info_label.config(
                    text="ℹ Using CPU",
                    fg="#95a5a6"
                )
        except Exception as e:
            self.device_info_label.config(
                text="⚠ Could not detect device",
                fg="#e74c3c"
            )

    def _open_outputs_folder(self):
        """Άνοιγμα του φακέλου outputs"""
        import subprocess
        import platform
        import os

        output_dir = Path("outputs")
        
        # Δημιουργία φακέλου αν δεν υπάρχει
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir.absolute()

        try:
            if platform.system() == "Windows":
                # Windows: χρήση explorer
                os.startfile(str(output_path))
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", str(output_path)])
            else:  # Linux
                subprocess.Popen(["xdg-open", str(output_path)])
            
            self.status_label.config(text=f"Άνοιξε φάκελος: {output_path}")
        except Exception as e:
            messagebox.showerror(
                "Σφάλμα",
                f"Δεν μπόρεσε να ανοίξει ο φάκελος:\n{str(e)}\n\n"
                f"Path: {output_path}"
            )
            self.status_label.config(text="Σφάλμα ανοίγματος φακέλου")

    def _initialize_models(self):
        """Αρχικοποίηση detector και tracker"""
        # Αν υπάρχει ήδη detector με διαφορετικό device, reset
        device_pref = self.device_preference.get()
        if self.detector is not None:
            # Check if device changed
            current_device = self.detector.device
            new_device = device_pref if device_pref != "auto" else None
            if new_device is None:
                # Auto mode - check what would be selected
                from src.detection.detect import get_available_device
                new_device = get_available_device(None)
            
            if current_device != new_device:
                self.detector = None
        
        if self.detector is None:
            self.status_label.config(text="Φόρτωση μοντέλου...")
            self.root.update()

            # Δημιουργία models directory αν δεν υπάρχει
            Path("models").mkdir(exist_ok=True)

            try:
                # Device selection
                device = None if device_pref == "auto" else device_pref
                
                # Χαμηλότερο confidence για καλύτερο detection
                self.detector = HumanDetector(
                    model_path="models/yolo12n.pt",
                    confidence=0.3,  # Μειωμένο από 0.5
                    device=device
                )
                self.tracker = HumanTracker(
                    max_time_lost=90,  # 3 seconds @ 30fps
                    reid_threshold=0.75,  # Πιο strict για ακριβέστερο re-ID
                    iou_threshold=0.3
                )
                device_status = f"Μοντέλο φορτώθηκε! ({self.detector.device_name})"
                self.status_label.config(text=device_status)
            except Exception as e:
                messagebox.showerror(
                    "Σφάλμα",
                    f"Σφάλμα φόρτωσης μοντέλου:\n{str(e)}\n\n"
                    "Το YOLOv12 θα κατέβει αυτόματα στην πρώτη εκτέλεση."
                )
                self.status_label.config(text="Σφάλμα φόρτωσης")
                return False

        return True

    def _start_camera(self):
        """Έναρξη real-time camera detection"""
        if not self._initialize_models():
            return

        self.is_running = True
        self.is_recording = False

        # Κλείσιμο του main window
        self.root.withdraw()

        # Εκτέλεση σε νέο thread
        thread = threading.Thread(target=self._process_camera, daemon=True)
        thread.start()

    def _upload_video(self):
        """Upload και επεξεργασία video"""
        if not self._initialize_models():
            return

        # Διάλογος επιλογής αρχείου
        file_path = filedialog.askopenfilename(
            title="Επιλογή Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ],
            initialdir="data"
        )

        if not file_path:
            return

        self.is_running = True

        # Κλείσιμο του main window
        self.root.withdraw()

        # Εκτέλεση σε νέο thread
        thread = threading.Thread(
            target=self._process_video,
            args=(file_path,),
            daemon=True
        )
        thread.start()

    def _start_recording(self, frame_shape, fps):
        """Έναρξη recording"""
        if self.is_recording:
            return

        # Δημιουργία output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / f"camera_recording_{timestamp}.mp4"

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frame_shape[:2]

        self.video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        self.is_recording = True
        self.recording_path = str(output_path)

        print(f"\n🔴 RECORDING STARTED: {output_path}")

    def _stop_recording(self):
        """Διακοπή recording"""
        if not self.is_recording:
            return

        self.is_recording = False

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        print(f"\n⏹️  RECORDING STOPPED")

        # Εμφάνιση dialog με το path
        def show_save_dialog():
            result = messagebox.askyesno(
                "Recording Αποθηκεύτηκε",
                f"Το video αποθηκεύτηκε στο:\n{self.recording_path}\n\n"
                "Θέλεις να ανοίξεις τον φάκελο outputs;"
            )

            if result:
                import subprocess
                import platform

                output_dir = Path("outputs").absolute()

                if platform.system() == "Windows":
                    subprocess.Popen(f'explorer "{output_dir}"')
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen(["open", str(output_dir)])
                else:  # Linux
                    subprocess.Popen(["xdg-open", str(output_dir)])

        # Run dialog in main thread
        self.root.after(0, show_save_dialog)

    def _process_camera(self):
        """Επεξεργασία real-time camera feed με REC capability"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            messagebox.showerror(
                "Σφάλμα",
                "Δεν μπόρεσε να ανοίξει η κάμερα!"
            )
            self.root.deiconify()
            return

        # Camera properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30  # Default

        # Reset tracker
        self.tracker.reset()

        frame_count = 0
        start_time = time.time()

        cv2.namedWindow("Human Detection - Camera", cv2.WINDOW_NORMAL)

        print("\n🎥 Έναρξη real-time detection...")
        print("Controls:")
        print("  R - Start/Resume Recording")
        print("  S - Stop Recording")
        print("  ESC - Exit\n")

        while self.is_running:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Detection με improved parameters
            detections = self.detector.detect(frame)

            # Tracking με appearance features
            tracks = self.tracker.update(detections, frame, frame_count, self.detector)
            stats = self.tracker.get_stats()

            # Visualization
            output_frame = draw_tracks(frame, tracks, stats)

            # FPS calculation
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            # FPS indicator
            cv2.putText(
                output_frame,
                f"FPS: {current_fps:.1f}",
                (output_frame.shape[1] - 150, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Device indicator
            device_text = self.detector.device_name if self.detector else "N/A"
            cv2.putText(
                output_frame,
                f"Device: {device_text}",
                (output_frame.shape[1] - 200, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            # REC indicator
            if self.is_recording:
                # Flashing REC indicator
                if frame_count % 20 < 10:
                    cv2.circle(output_frame, (30, 30), 15, (0, 0, 255), -1)
                cv2.putText(
                    output_frame,
                    "REC",
                    (55, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                # Write frame
                if self.video_writer is not None:
                    self.video_writer.write(output_frame)

            # Display
            display_frame = resize_frame(output_frame, max_width=1280)
            cv2.imshow("Human Detection - Camera", display_frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC - Exit
                if self.is_recording:
                    self._stop_recording()
                break
            elif key == ord('r') or key == ord('R'):  # R - Start recording
                if not self.is_recording:
                    self._start_recording(output_frame.shape, fps)
            elif key == ord('s') or key == ord('S'):  # S - Stop recording
                if self.is_recording:
                    self._stop_recording()

        # Cleanup
        if self.is_recording:
            self._stop_recording()

        cap.release()
        cv2.destroyAllWindows()

        # Επιστροφή στο main window
        self.root.deiconify()
        self.status_label.config(text="Έτοιμο")

        # Εμφάνιση τελικών στατιστικών
        messagebox.showinfo(
            "Στατιστικά Session",
            f"📊 Session Ολοκληρώθηκε\n\n"
            f"Frames: {frame_count}\n"
            f"Συνολικοί άνθρωποι: {stats['total_people']}\n"
            f"Μέσο FPS: {current_fps:.1f}\n"
            f"Διάρκεια: {elapsed:.1f}s"
        )

    def _process_video(self, video_path):
        """Επεξεργασία uploaded video"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            messagebox.showerror(
                "Σφάλμα",
                f"Δεν μπόρεσε να ανοίξει το video:\n{video_path}"
            )
            self.root.deiconify()
            return

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video path
        output_path = create_output_path(video_path)

        # Video writer (με panel height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 100))

        # Reset tracker
        self.tracker.reset()

        frame_count = 0
        start_time = time.time()

        cv2.namedWindow("Human Detection - Video", cv2.WINDOW_NORMAL)

        print(f"\n🎬 Επεξεργασία video: {Path(video_path).name}")
        print(f"Frames: {total_frames} | FPS: {fps}")
        print("Πάτα ESC για ακύρωση\n")

        while self.is_running:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Detection με improved parameters
            detections = self.detector.detect(frame)

            # Tracking με appearance features
            tracks = self.tracker.update(detections, frame, frame_count, self.detector)
            stats = self.tracker.get_stats()

            # Visualization
            output_frame = draw_tracks(frame, tracks, stats)

            # Progress
            progress = (frame_count / total_frames) * 100
            cv2.putText(
                output_frame,
                f"Progress: {progress:.1f}%",
                (output_frame.shape[1] - 220, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # Αποθήκευση frame
            out.write(output_frame)

            # Display (κάθε 2 frames για ταχύτητα)
            if frame_count % 2 == 0:
                display_frame = resize_frame(output_frame, max_width=1280)
                cv2.imshow("Human Detection - Video", display_frame)

            # ESC για ακύρωση
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n❌ Ακυρώθηκε από τον χρήστη")
                break

            # Print progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_count}/{total_frames} | "
                      f"{progress:.1f}% | FPS: {fps_processing:.1f} | "
                      f"People: {stats['current_people']}/{stats['total_people']}")

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Επιστροφή στο main window
        self.root.deiconify()
        self.status_label.config(text="Έτοιμο")

        # Στατιστικά
        elapsed = time.time() - start_time
        stats = self.tracker.get_stats()

        if frame_count == total_frames:
            messagebox.showinfo(
                "Ολοκληρώθηκε!",
                f"✅ Το video αποθηκεύτηκε:\n{output_path}\n\n"
                f"📊 Στατιστικά:\n"
                f"Frames: {frame_count}\n"
                f"Συνολικοί άνθρωποι: {stats['total_people']}\n"
                f"Χρόνος: {elapsed:.1f}s\n"
                f"Μέσο FPS: {frame_count/elapsed:.1f}"
            )
            print(f"\n✅ Ολοκληρώθηκε!")
            print(f"Output: {output_path}")
            print(f"Συνολικοί άνθρωποι: {stats['total_people']}")
        else:
            print(f"\n⚠️ Διακόπηκε στο frame {frame_count}/{total_frames}")


def main():
    """Entry point"""
    # Δημιουργία απαραίτητων directories
    for directory in ["models", "data", "outputs"]:
        Path(directory).mkdir(exist_ok=True)

    print("=" * 60)
    print("🎥 Human Detection & Tracking System")
    print("=" * 60)
    print("\n📋 Features:")
    print("  ✅ YOLOv12 Detection (improved confidence)")
    print("  ✅ Appearance-based Re-identification")
    print("  ✅ Real-time Camera with Recording")
    print("  ✅ Video Upload Processing")
    print("\n🎮 Camera Controls:")
    print("  R - Start Recording")
    print("  S - Stop Recording")
    print("  ESC - Exit")
    print("\n" + "=" * 60 + "\n")

    # Εκτέλεση GUI
    root = tk.Tk()
    app = HumanDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()