# Git Commit Guide

## Αρχεία που πρέπει να κάνεις Commit:

### ✅ Source Code (Source Files)
```bash
git add demo.py
git add src/
git add requirements.txt
```

### ✅ Documentation
```bash
git add README.md
git add INNO_SETUP_INSTRUCTIONS.md
git add LICENSE
```

### ✅ Build Configuration Files
```bash
git add .gitignore
git add Human_Detection_System.spec
git add build_exe.bat
git add Human_Detection_System_Setup.iss
```

### ✅ Data (Optional - αν θέλεις να συμπεριλάβεις sample video)
```bash
# Προαιρετικό - το video είναι μεγάλο
# git add data/people_walking.mp4
```

## Αρχεία που ΔΕΝ πρέπει να κάνεις Commit:

❌ `build/` - Build artifacts (αποκλείεται από .gitignore)
❌ `dist/` - Compiled executables (αποκλείεται από .gitignore)
❌ `__pycache__/` - Python cache (αποκλείεται από .gitignore)
❌ `installer/` - Installer outputs (αποκλείεται από .gitignore)
❌ `models/*.pt` - Model files (μεγάλα, θα κατέβουν αυτόματα)
❌ `outputs/` - Output videos (αποκλείεται από .gitignore)
❌ `.idea/` - IDE settings (αποκλείεται από .gitignore)

## Quick Commit Commands:

```bash
# Προσθήκη όλων των σημαντικών αρχείων
git add .gitignore
git add demo.py
git add src/
git add requirements.txt
git add README.md
git add INNO_SETUP_INSTRUCTIONS.md
git add LICENSE
git add Human_Detection_System.spec
git add build_exe.bat
git add Human_Detection_System_Setup.iss

# Commit
git commit -m "Add build configuration and fix PermissionError handling

- Add PyInstaller spec file and build scripts
- Add Inno Setup installer configuration
- Fix PermissionError by using AppData fallback
- Add comprehensive .gitignore
- Update README with build instructions"

# Push
git push origin main
```

## Εναλλακτικά (αν θέλεις να προσθέσεις όλα τα tracked files):

```bash
# Προσθήκη όλων των modified και untracked files (με βάση το .gitignore)
git add .

# Commit
git commit -m "Add build configuration and fix PermissionError handling"

# Push
git push origin main
```

## Σημείωση:

Το `.gitignore` που δημιουργήθηκε θα αποκλείσει αυτόματα:
- Build artifacts (build/, dist/)
- Cache files (__pycache__/)
- Output files (outputs/)
- Model files (models/*.pt)
- Installer outputs (installer/)

Έτσι μπορείς να χρησιμοποιήσεις `git add .` με ασφάλεια!

