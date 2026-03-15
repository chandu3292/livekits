@echo off
setlocal enabledelayedexpansion

:: ─────────────────────────────────────────────────────────
:: LiveKit Agent Platform - Start / Restart Script (Windows)
:: Starts: LiveKit Server, FastAPI Server, MCP Voice Agent
:: Logs overwritten on each restart
:: ─────────────────────────────────────────────────────────

cd /d "%~dp0"
set "DIR=%~dp0"
set "LOG_DIR=%DIR%logs"
set "PYTHONIOENCODING=utf-8"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: Route to command
if "%1"=="" goto restart
if /i "%1"=="start" goto start_all
if /i "%1"=="stop" goto stop_all
if /i "%1"=="restart" goto restart
if /i "%1"=="status" goto status
goto usage

:: ── STOP ──
:stop_all
echo.
echo   Stopping existing processes...

:: Kill by port - LiveKit (7880)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7880.*LISTENING" 2^>nul') do (
    echo   Stopped LiveKit Server (PID %%a)
    taskkill /F /PID %%a >nul 2>&1
)

:: Kill by port - FastAPI (8005)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8005.*LISTENING" 2^>nul') do (
    echo   Stopped FastAPI Server (PID %%a)
    taskkill /F /PID %%a >nul 2>&1
)

:: Kill by window title
taskkill /FI "WINDOWTITLE eq livekit-srv" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq fastapi-srv" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq mcp-agent" /F >nul 2>&1

timeout /t 2 /nobreak >nul
echo   Done.
echo.
if /i "%1"=="stop" (
    echo   All services stopped.
    goto end
)
goto :eof

:: ── START ──
:start_all
echo.
echo   LiveKit Agent Platform - Starting...
echo.

:: Activate virtual environment
if exist "%DIR%.venv\Scripts\activate.bat" (
    echo   Activating .venv...
    call "%DIR%.venv\Scripts\activate.bat"
    echo         Done
) else (
    echo   WARNING: .venv not found, using system Python
)
echo.

:: 1. LiveKit Server
if exist "%DIR%livekit-server.exe" (
    set "LK_BIN=%DIR%livekit-server.exe"
) else (
    where livekit-server >nul 2>&1
    if !errorlevel! equ 0 (
        set "LK_BIN=livekit-server"
    ) else (
        echo   ERROR: livekit-server.exe not found!
        goto end
    )
)

echo   [1/3] LiveKit Server (port 7880)
start "livekit-srv" /B cmd /c "!LK_BIN! --config "%DIR%livekit.yaml" > "%LOG_DIR%\livekit.log" 2>&1"

set "ready=0"
for /l %%i in (1,1,15) do (
    if !ready!==0 (
        curl -s http://localhost:7880 >nul 2>&1
        if !errorlevel! equ 0 (
            echo         Ready
            set "ready=1"
        ) else (
            timeout /t 1 /nobreak >nul
        )
    )
)
if !ready!==0 echo         Timeout

:: 2. FastAPI Server
echo   [2/3] FastAPI Server (port 8005)
start "fastapi-srv" /B cmd /c "python "%DIR%server.py" > "%LOG_DIR%\server.log" 2>&1"

set "ready=0"
for /l %%i in (1,1,45) do (
    if !ready!==0 (
        curl -s http://localhost:8005 >nul 2>&1
        if !errorlevel! equ 0 (
            echo         Ready
            set "ready=1"
        ) else (
            timeout /t 1 /nobreak >nul
        )
    )
)
if !ready!==0 echo         Timeout (model may still be loading)

:: 3. MCP Voice Agent
echo   [3/3] MCP Voice Agent
start "mcp-agent" /B cmd /c "python "%DIR%mcp-agent.py" start > "%LOG_DIR%\agent.log" 2>&1"
echo         Started
echo.

echo   All services running!
echo   UI: http://localhost:8005
echo   Logs: logs\livekit.log, logs\server.log, logs\agent.log
echo.
goto end

:: ── RESTART ──
:restart
call :stop_all
goto start_all

:: ── STATUS ──
:status
echo.
set "found=0"
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7880.*LISTENING" 2^>nul') do (
    echo   [RUNNING]  LiveKit Server  (PID %%a)
    set "found=1"
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8005.*LISTENING" 2^>nul') do (
    echo   [RUNNING]  FastAPI Server  (PID %%a)
    set "found=1"
)
if !found!==0 echo   No services running
echo.
goto end

:usage
echo   Usage: start.bat [start^|stop^|restart^|status]
echo   Default: restart

:end
endlocal
