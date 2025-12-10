"""
Human Tracker με Improved Re-identification
Χρησιμοποιεί appearance features για καλύτερο re-ID
"""

import numpy as np
from collections import defaultdict, deque
from scipy.spatial.distance import cosine


class HumanTracker:
    """
    Tracker με improved re-identification capabilities
    Χρησιμοποιεί appearance features για ακριβέστερο matching
    """

    def __init__(self, max_time_lost=90, reid_threshold=0.75, iou_threshold=0.3):
        """
        Args:
            max_time_lost: Πόσα frames να θυμάται ένα lost track
            reid_threshold: Threshold για re-identification (0-1, higher = stricter)
            iou_threshold: Minimum IoU για frame-to-frame matching
        """
        self.max_time_lost = max_time_lost
        self.reid_threshold = reid_threshold
        self.iou_threshold = iou_threshold

        # Tracking state
        self.tracks = {}  # active tracks: {track_id: track_info}
        self.lost_tracks = {}  # lost tracks για re-ID
        self.next_id = 1
        self.total_people = 0  # Σύνολο unique ανθρώπων

        # History για κάθε track
        self.track_history = defaultdict(lambda: deque(maxlen=30))

        # Appearance features για re-ID
        self.track_features = {}  # {track_id: list of feature vectors}

    def _compute_iou(self, box1, box2):
        """Υπολογισμός IoU μεταξύ δύο boxes"""
        x1_min, y1_min, x1_max, y1_max = box1[:4]
        x2_min, y2_min, x2_max, y2_max = box2[:4]

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _compute_appearance_similarity(self, features1, features2):
        """
        Υπολογισμός ομοιότητας βάσει appearance features
        Χρησιμοποιεί cosine similarity

        Args:
            features1, features2: Feature vectors

        Returns:
            similarity: 0-1 (1 = identical)
        """
        if features1 is None or features2 is None:
            return 0.0

        try:
            # Cosine similarity (1 - cosine distance)
            similarity = 1.0 - cosine(features1, features2)
            return max(0.0, similarity)  # Clamp to [0, 1]
        except:
            return 0.0

    def _get_average_feature(self, track_id):
        """Επιστροφή μέσου feature vector για ένα track"""
        if track_id not in self.track_features or len(self.track_features[track_id]) == 0:
            return None

        # Average των τελευταίων features
        features = self.track_features[track_id]
        return np.mean(features, axis=0)

    def _match_detections_to_tracks(self, detections):
        """Matching detections με existing tracks χρησιμοποιώντας IoU"""
        if len(detections) == 0:
            return [], list(self.tracks.keys()), []

        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))

        # Υπολογισμός IoU matrix
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]['box'] for tid in track_ids]

        iou_matrix = np.zeros((len(track_boxes), len(detections)))
        for i, tbox in enumerate(track_boxes):
            for j, dbox in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(tbox, dbox)

        # Greedy matching με minimum IoU threshold
        matches = []
        unmatched_tracks = list(range(len(track_ids)))
        unmatched_detections = list(range(len(detections)))

        while len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)

            if iou_matrix[i, j] < self.iou_threshold:
                break

            matches.append((track_ids[i], j))

            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
            unmatched_tracks.remove(i)
            unmatched_detections.remove(j)

        # Επιστροφή unmatched track IDs
        unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]

        return matches, unmatched_track_ids, unmatched_detections

    def _try_reidentify(self, detection_feature):
        """
        Προσπάθεια re-identification με lost tracks
        Χρησιμοποιεί appearance features

        Args:
            detection_feature: Feature vector του νέου detection

        Returns:
            best_match_id: ID του matched track ή None
        """
        if len(self.lost_tracks) == 0 or detection_feature is None:
            return None

        best_match_id = None
        best_similarity = 0

        for track_id, lost_info in self.lost_tracks.items():
            # Σύγκριση με το average feature του track
            track_feature = self._get_average_feature(track_id)

            if track_feature is None:
                continue

            # Appearance similarity
            app_similarity = self._compute_appearance_similarity(
                track_feature,
                detection_feature
            )

            # Spatial proximity (bonus αν είναι κοντά στην τελευταία θέση)
            spatial_bonus = 0.0
            if 'last_position' in lost_info:
                last_x = (lost_info['box'][0] + lost_info['box'][2]) / 2
                last_y = (lost_info['box'][1] + lost_info['box'][3]) / 2

                # Θα χρησιμοποιήσουμε αυτό αργότερα με το detection box
                spatial_bonus = 0.0  # Placeholder

            # Combined similarity
            similarity = app_similarity * 0.9 + spatial_bonus * 0.1

            if similarity > best_similarity and similarity > self.reid_threshold:
                best_similarity = similarity
                best_match_id = track_id

        return best_match_id

    def update(self, detections, frame, frame_id, detector=None):
        """
        Update tracker με νέα detections

        Args:
            detections: numpy array [x1, y1, x2, y2, conf]
            frame: Current frame (για feature extraction)
            frame_id: Current frame number
            detector: HumanDetector instance (για feature extraction)

        Returns:
            List of (track_id, box) tuples
        """
        # Extract appearance features για όλα τα detections
        detection_features = []
        if detector is not None:
            for det in detections:
                feat = detector.extract_appearance_features(frame, det)
                detection_features.append(feat)
        else:
            detection_features = [None] * len(detections)

        # Matching με existing tracks
        matches, unmatched_tracks, unmatched_detections = \
            self._match_detections_to_tracks(detections)

        # Update matched tracks
        for track_id, det_idx in matches:
            self.tracks[track_id]['box'] = detections[det_idx]
            self.tracks[track_id]['last_seen'] = frame_id
            self.tracks[track_id]['hits'] += 1

            # Update features
            if detection_features[det_idx] is not None:
                if track_id not in self.track_features:
                    self.track_features[track_id] = []

                self.track_features[track_id].append(detection_features[det_idx])

                # Κράτα μόνο τα τελευταία 10 features
                if len(self.track_features[track_id]) > 10:
                    self.track_features[track_id] = self.track_features[track_id][-10:]

            # Update history
            self.track_history[track_id].append(
                ((detections[det_idx][0] + detections[det_idx][2]) / 2,
                 (detections[det_idx][1] + detections[det_idx][3]) / 2)
            )

        # Move unmatched tracks to lost
        for track_id in unmatched_tracks:
            # Μετακίνηση στα lost tracks (unmatched = not detected in this frame)
            if track_id in self.tracks:
                self.lost_tracks[track_id] = self.tracks[track_id].copy()
                self.lost_tracks[track_id]['last_position'] = (
                    (self.tracks[track_id]['box'][0] + self.tracks[track_id]['box'][2]) / 2,
                    (self.tracks[track_id]['box'][1] + self.tracks[track_id]['box'][3]) / 2
                )
                del self.tracks[track_id]

        # Handle unmatched detections (νέοι άνθρωποι ή re-ID)
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            detection_feat = detection_features[det_idx]

            # Προσπάθεια re-identification
            reidentified_id = self._try_reidentify(detection_feat)

            if reidentified_id is not None:
                # Re-identification επιτυχής!
                print(f"✅ Re-identified: ID {reidentified_id}")

                self.tracks[reidentified_id] = {
                    'box': detection,
                    'last_seen': frame_id,
                    'hits': self.lost_tracks[reidentified_id]['hits'] + 1
                }

                # Update features
                if detection_feat is not None:
                    if reidentified_id not in self.track_features:
                        self.track_features[reidentified_id] = []
                    self.track_features[reidentified_id].append(detection_feat)

                del self.lost_tracks[reidentified_id]
            else:
                # Νέο track
                new_id = self.next_id
                self.next_id += 1
                self.total_people += 1

                self.tracks[new_id] = {
                    'box': detection,
                    'last_seen': frame_id,
                    'hits': 1
                }

                # Initialize features
                if detection_feat is not None:
                    self.track_features[new_id] = [detection_feat]

                print(f"🆕 Νέο άτομο: ID {new_id}")

        # Καθαρισμός πολύ παλιών lost tracks
        to_remove = []
        for track_id, lost_info in self.lost_tracks.items():
            if frame_id - lost_info['last_seen'] > self.max_time_lost:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.lost_tracks[track_id]
            if track_id in self.track_features:
                del self.track_features[track_id]

        # Επιστροφή active tracks
        return [(tid, info['box']) for tid, info in self.tracks.items()]

    def get_stats(self):
        """Επιστροφή statistics"""
        return {
            'current_people': len(self.tracks),
            'total_people': self.total_people,
            'lost_tracks': len(self.lost_tracks)
        }

    def reset(self):
        """Reset του tracker"""
        self.tracks.clear()
        self.lost_tracks.clear()
        self.track_history.clear()
        self.track_features.clear()
        self.next_id = 1
        self.total_people = 0