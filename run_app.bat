@echo off
rem Starts the DeepResearch Agent: Streamlit runs the Python backend and the web page in one process
cd /d "%~dp0"
set PORT=8501

rem when started by itself with "open_browser", only wait for the server and open the website
if "%~1"=="open_browser" goto :open_browser

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating the virtual environment and installing packages, this takes a few minutes...
    python -m venv .venv || goto :failed
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :failed
)

if not exist ".env" (
    echo [error] .env is missing. Copy .env.example to .env and paste your GEMINI_API_KEY.
    goto :failed
)

if not exist "app.py" (
    echo [error] app.py does not exist yet. The web UI is built in milestone 6, see PLAN.md.
    goto :failed
)

rem open the website in the background as soon as the server answers
start "" /min "%~f0" open_browser

echo Starting DeepResearch Agent on http://localhost:%PORT%  (close this window or press Ctrl+C to stop)
".venv\Scripts\python.exe" -m streamlit run app.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false
goto :eof

:open_browser
rem check the Streamlit health page once per second for up to 60 seconds
for /l %%i in (1,1,60) do (
    curl -sf -o nul http://localhost:%PORT%/_stcore/health && goto :browser_ready
    timeout /t 1 /nobreak >nul
)
:browser_ready
start "" http://localhost:%PORT%
exit

:failed
pause
exit /b 1
