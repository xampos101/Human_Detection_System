@echo off
echo ========================================
echo Building Human Detection System to EXE
echo ========================================
echo.

REM Ελέγχος αν υπάρχει το spec file
if exist "Human_Detection_System.spec" (
    echo Using spec file...
    py -m PyInstaller Human_Detection_System.spec
) else (
    echo Creating build with PyInstaller...
    py -m PyInstaller --name="Human_Detection_System" ^
        --onedir ^
        --windowed ^
        --add-data "models;models" ^
        --add-data "data;data" ^
        --hidden-import=ultralytics ^
        --hidden-import=torch ^
        --hidden-import=torchvision ^
        --hidden-import=cv2 ^
        --hidden-import=numpy ^
        --hidden-import=PIL ^
        --hidden-import=scipy ^
        --hidden-import=lap ^
        --hidden-import=tkinter ^
        --collect-all=ultralytics ^
        --collect-all=torch ^
        demo.py
)

echo.
echo ========================================
echo Build completed!
echo ========================================
echo.
echo The executable can be found in: dist\Human_Detection_System\
echo.
pause

