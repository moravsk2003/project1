@echo off
setlocal

cd /d "%~dp0"

call :find_java
call :find_maven
if errorlevel 1 goto :end

where nvidia-smi >nul 2>nul
if "%ERRORLEVEL%"=="0" (
    set "PROFILE=cuda"
    echo NVIDIA GPU detected. Running Rust Base Builder with CUDA backend...
) else (
    set "PROFILE=cpu"
    echo NVIDIA GPU was not detected. Running Rust Base Builder with CPU backend...
)

echo Using Maven: %MVN_CMD%
if defined JAVA_HOME echo Using JAVA_HOME: %JAVA_HOME%
echo.

"%MVN_CMD%" -P%PROFILE% clean javafx:run
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Rust Base Builder did not start. Maven exited with code %EXIT_CODE%.
)

goto :pause_and_exit

:find_java
if defined JAVA_HOME if exist "%JAVA_HOME%\bin\java.exe" exit /b 0

for %%J in (
    "%USERPROFILE%\.antigravity\extensions\redhat.java-1.55.2026042408-win32-x64\jre\21.0.10-win32-x86_64"
    "C:\Program Files\JetBrains\IntelliJ IDEA 2026.1.1\jbr"
    "C:\Program Files\JetBrains\IntelliJ IDEA Community Edition 2025.2.5\jbr"
    "C:\Program Files\Java\jdk-23"
    "C:\Program Files\Eclipse Adoptium\jdk-25.0.1.8-hotspot"
) do (
    if exist "%%~J\bin\java.exe" (
        set "JAVA_HOME=%%~J"
        set "PATH=%%~J\bin;%PATH%"
        exit /b 0
    )
)

exit /b 0

:find_maven
if exist "%~dp0mvnw.cmd" (
    set "MVN_CMD=%~dp0mvnw.cmd"
    exit /b 0
)

for %%M in (
    "mvn.cmd"
    "mvn.bat"
    "mvn.exe"
    "C:\Program Files\JetBrains\IntelliJ IDEA 2026.1.1\plugins\maven\lib\maven3\bin\mvn.cmd"
    "C:\Program Files\JetBrains\IntelliJ IDEA Community Edition 2025.2.5\plugins\maven\lib\maven3\bin\mvn.cmd"
) do (
    where %%~M >nul 2>nul
    if not errorlevel 1 (
        set "MVN_CMD=%%~M"
        exit /b 0
    )
    if exist "%%~M" (
        set "MVN_CMD=%%~M"
        exit /b 0
    )
)

echo Maven was not found.
echo Install Maven, add it to PATH, or install IntelliJ IDEA with the bundled Maven plugin.
exit /b 1

:pause_and_exit
pause
exit /b %EXIT_CODE%

:end
set "EXIT_CODE=%ERRORLEVEL%"
goto :pause_and_exit
