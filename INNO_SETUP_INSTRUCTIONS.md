# Οδηγίες για Inno Setup Compiler

## Αρχεία που χρειάζονται

Στο Inno Setup Compiler, πρέπει να συμπεριλάβεις **ολόκληρο** τον φάκελο `dist\Human_Detection_System\`:

### Αρχεία που πρέπει να συμπεριλάβεις:

1. **Human_Detection_System.exe** - Το κύριο εκτελέσιμο αρχείο
2. **Ολόκληρος ο φάκελος `_internal\`** - Περιέχει όλα τα dependencies:
   - Python runtime
   - PyTorch libraries
   - OpenCV
   - Ultralytics
   - NumPy, SciPy
   - Tkinter (GUI)
   - Όλα τα άλλα dependencies

### Δομή αρχείων:

```
dist\Human_Detection_System\
├── Human_Detection_System.exe    ← ΚΥΡΙΟ ΕΚΤΕΛΕΣΙΜΟ
└── _internal\                    ← ΟΛΟΚΛΗΡΟΣ ΦΑΚΕΛΟΣ (RECURSIVE)
    ├── *.dll
    ├── *.pyd
    ├── torch\
    ├── ultralytics\
    ├── cv2\
    ├── models\
    └── ... (όλα τα dependencies)
```

## Οδηγίες για Inno Setup

### Μέθοδος 1: Χρήση του .iss Script (Προτεινόμενη)

1. Άνοιξε το **Inno Setup Compiler**
2. File → Open → Επέλεξε `Human_Detection_System_Setup.iss`
3. Build → Compile (ή F9)
4. Το installer θα δημιουργηθεί στο `installer\` folder

### Μέθοδος 2: Manual Setup

1. Άνοιξε το **Inno Setup Compiler**
2. File → New → Create a new script file using the Script Wizard
3. Συμπλήρωσε τα στοιχεία:
   - Application name: `Human Detection System`
   - Application version: `1.0`
   - Application publisher: (Το όνομα σου)
   - Application website: (Προαιρετικό)

4. **Στο Files section:**
   - Source: `C:\Users\xampo\PycharmProjects\TeLoII\Human_Detection_System\dist\Human_Detection_System\Human_Detection_System.exe`
   - Destination: `{app}\` (root installation folder)

5. **Προσθήκη του _internal folder:**
   - Source: `C:\Users\xampo\PycharmProjects\TeLoII\Human_Detection_System\dist\Human_Detection_System\_internal\*`
   - Destination: `{app}\_internal\`
   - **ΣΗΜΑΝΤΙΚΟ**: Ενεργοποίησε το checkbox **"Recurse subdirectories"** (recursesubdirs)
   - Ενεργοποίησε το checkbox **"Create all subdirectories"** (createallsubdirs)

6. **Icons:**
   - Desktop icon: `{app}\Human_Detection_System.exe`
   - Start Menu: `{app}\Human_Detection_System.exe`

7. **Run section (Προαιρετικό):**
   - Ενεργοποίησε "Run after installation" αν θέλεις να ανοίξει το app μετά την εγκατάσταση

## Σημαντικά Σημεία

### ✅ DO (Κάνε):
- Συμπερίλαβε **ΟΛΟΚΛΗΡΟ** τον φάκελο `_internal\` με **recursive** option
- Βεβαιώσου ότι το `Human_Detection_System.exe` είναι στο root του installation folder
- Το `_internal\` folder πρέπει να είναι **δίπλα** στο .exe (όχι μέσα σε άλλο φάκελο)

### ❌ DON'T (Μην κάνεις):
- Μην ξεχάσεις το `_internal\` folder - το .exe δεν θα τρέξει χωρίς αυτό
- Μην αλλάξεις τη δομή των folders - το .exe αναζητά τα dependencies σε συγκεκριμένα paths
- Μην προσθέσεις μόνο μερικά αρχεία από το `_internal\` - χρειάζονται όλα

## Δομή μετά την εγκατάσταση

Μετά την εγκατάσταση, η δομή θα είναι:

```
C:\Program Files\Human Detection System\
├── Human_Detection_System.exe
└── _internal\
    ├── (όλα τα dependencies)
    └── ...
```

## Testing

Μετά τη δημιουργία του installer:

1. Τρέξε το installer σε ένα test system
2. Βεβαιώσου ότι το .exe τρέχει χωρίς errors
3. Δοκίμασε όλες τις λειτουργίες (camera, video upload, κτλ)

## Μέγεθος Installer

Το installer θα είναι περίπου **1-2 GB** λόγω:
- PyTorch (~500MB)
- Ultralytics (~200MB)
- OpenCV (~100MB)
- SciPy, NumPy (~200MB)
- Python runtime (~100MB)
- Άλλα dependencies

Αυτό είναι **φυσιολογικό** για ML applications.

## Troubleshooting

**Πρόβλημα**: Το .exe δεν τρέχει μετά την εγκατάσταση
- **Λύση**: Βεβαιώσου ότι το `_internal\` folder συμπεριλήφθηκε με recursive option

**Πρόβλημα**: Missing DLL errors
- **Λύση**: Συμπερίλαβε όλα τα .dll files από το `_internal\` folder

**Πρόβλημα**: Το installer είναι πολύ μεγάλο
- **Λύση**: Χρησιμοποίησε compression (LZMA) - είναι ενεργοποιημένο στο .iss script

