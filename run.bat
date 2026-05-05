@echo off
where nvidia-smi >nul 2>nul

if %ERRORLEVEL%==0 (
    echo NVIDIA GPU detected. Running Rust Base Builder with CUDA backend...
    mvn -Pcuda clean javafx:run
) else (
    echo NVIDIA GPU was not detected. Running Rust Base Builder with CPU backend...
    mvn -Pcpu clean javafx:run
)

pause
