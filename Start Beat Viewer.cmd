@echo off
rem NeoTex Beat Viewer - double-click, or drag an ECG CSV onto this file.
rem Uses your global Python; missing packages are installed into it on the first run.
cd /d "%~dp0"
where py >nul 2>nul && ( py -3 beat_viewer.py %* ) || ( python beat_viewer.py %* )
if errorlevel 1 pause
