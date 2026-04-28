@echo off
echo Running Rust Base Builder with bundled JDK...

set JAVA_EXE=%USERPROFILE%\.antigravity\extensions\redhat.java-1.49.0-win32-x64\jre\21.0.9-win32-x86_64\bin\java.exe
set JFX_PATH=%USERPROFILE%\.m2\repository\org\openjfx

if not exist "%JAVA_EXE%" (
    echo Error: Java executable not found at %JAVA_EXE%
    pause
    exit /b
)

"%JAVA_EXE%" ^
  --module-path "%JFX_PATH%\javafx-controls\21\javafx-controls-21-win.jar;%JFX_PATH%\javafx-graphics\21\javafx-graphics-21-win.jar;%JFX_PATH%\javafx-base\21\javafx-base-21-win.jar;%JFX_PATH%\javafx-fxml\21\javafx-fxml-21-win.jar" ^
  --add-modules javafx.controls,javafx.fxml ^
  -cp "target\classes" ^
  com.rustbuilder.Launcher

pause
