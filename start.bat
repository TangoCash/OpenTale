@echo off
REM ===========================================
REM Start script for OpenTale
REM ===========================================

REM Check if the virtual environment exists, create if not.
if not exist ".venv\Scripts\activate" (
    echo Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b %errorlevel%
    )
)

REM Install/update dependencies.
echo Installing dependencies...
call .venv\Scripts\python -m pip install --upgrade pip -q
call .venv\Scripts\pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b %errorlevel%
)

REM Activate the virtual environment.
echo Activating virtual environment...
call .venv\Scripts\activate

REM Run the main Python script.
echo Running main.py...
python web_app.py

pause