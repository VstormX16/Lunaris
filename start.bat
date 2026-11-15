@echo off
rem Starts the analysis server (port 8000) and the web app (port 5173), each in its own window.
cd /d "%~dp0"
start "Lunar analysis server" cmd /k python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
cd frontend
start "Lunar web app" cmd /k npm run dev
echo Open http://localhost:5173 once both windows say they are ready.
