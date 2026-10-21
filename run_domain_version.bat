@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if exist "%VENV_PY%" goto install

if exist "%LOCALAPPDATA%\Programs\TruthLens\Python313\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\TruthLens\Python313\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('where.exe python 2^>nul') do (
        echo %%P | findstr /i /c:"\WindowsApps\" >nul
        if errorlevel 1 (
            "%%P" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,13) else 1)" >nul 2>&1
            if not errorlevel 1 if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
        )
    )
)

if not defined PYTHON_EXE (
    echo Python 3.13 was not found. Install Python 3.13 and try again.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m venv .venv
if errorlevel 1 (
    echo Could not create the project virtual environment.
    pause
    exit /b 1
)

:install
".venv\Scripts\python.exe" -m pip install -r domain_version\requirements.txt
if errorlevel 1 (
    echo Could not install the domain-version requirements.
    pause
    exit /b 1
)

if not exist "domain_version\models\model.joblib" (
    echo The model is missing. Run train_model.py from this folder first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" domain_version\app.py
pause
