# Human Detection System �

Σύστημα ανίχνευσης και tracking ανθρώπων με YOLOv12, improved re-ID και camera recording.

## Χαρακτηριστικά

- ✅ YOLOv12 detection (πιο πρόσφατο μοντέλο)
- ✅ ByteTrack για robust tracking
- ✅ Re-identification: διατηρεί το ID ακόμα κι αν βγει και ξαναμπεί στο πλάνο
- ✅ Real-time camera ή video upload
- ✅ GUI με Tkinter
- ✅ Μετρητές: τρέχοντες άνθρωποι & συνολικοί άνθρωποι που πέρασαν
- ✅ Recording στο camera mode με hotkeys (R/S)
- ✅ Καθαρή αρχιτεκτονική

## Εγκατάσταση

```bash
# Δημιουργία virtual environment
python -m venv .venv

# Ενεργοποίηση
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Εγκατάσταση dependencies
pip install -r requirements.txt
```

## Χρήση

```bash
python src/demo.py
```

Μέσα από το GUI μπορείς να επιλέξεις:
- **Camera**: Real-time detection από webcam
- **Upload Video**: Ανέβασμα video αρχείου

## Δομή Project

```
Human Detection/
├── .venv/                    # Virtual environment
├── data/                     # Video αρχεία
│   └── people_walking.mp4
├── models/                   # Αποθηκευμένα μοντέλα
├── outputs/                  # Output videos με detections
├── src/
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detect.py        # YOLOv12 detector
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py       # ByteTrack tracker με re-ID
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py       # Helper functions
│   └── demo.py              # GUI & main application
├── README.md
└── requirements.txt
```

## Πώς λειτουργεί το Re-identification

Το σύστημα χρησιμοποιεί:
1. **ByteTrack**: Για frame-to-frame tracking
2. **Feature Memory**: Αποθηκεύει χαρακτηριστικά κάθε ID
3. **Similarity Matching**: Όταν εμφανίζεται νέο ID, το συγκρίνει με παλιά IDs
4. **Threshold-based Re-ID**: Αν η ομοιότητα > 0.7, επαναχρησιμοποιεί το παλιό ID

## Παράμετροι Tuning

Στο `tracker.py` μπορείς να ρυθμίσεις:
- `reid_threshold`: Όριο για re-identification (default: 0.7)
- `max_time_lost`: Πόσα frames να θυμάται ένα ID (default: 120)
- `track_buffer`: Buffer για ByteTrack (default: 30)

## Requirements

- Python 3.8+
- CUDA (προαιρετικό, για GPU acceleration)
- Webcam (για real-time mode)

## Troubleshooting

**Πρόβλημα**: Το YOLOv12 δεν κατεβαίνει
- Λύση: Κατέβασε manually το `yolo12n.pt` στον φάκελο `models/`

**Πρόβλημα**: Αργή ταχύτητα
- Λύση: Χρησιμοποίησε GPU ή μικρότερο μοντέλο (nano: yolo12n.pt)

**Πρόβλημα**: Camera δεν ανοίγει
- Λύση: Έλεγξε αν η κάμερα χρησιμοποιείται από άλλη εφαρμογή
