# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['demo.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models', 'models'),
        ('data', 'data'),
    ],
    hiddenimports=[
        'ultralytics',
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'PIL',
        'PIL._tkinter_finder',
        'scipy',
        'lap',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Human_Detection_System',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # False για GUI app (no console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Μπορείς να προσθέσεις .ico file εδώ
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Human_Detection_System',
)

