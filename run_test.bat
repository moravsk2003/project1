@echo off
set JAVA_EXE=%USERPROFILE%\.antigravity\extensions\redhat.java-1.53.2026021110-win32-x64\jre\21.0.10-win32-x86_64\bin\java.exe
set JAVAC_EXE=%USERPROFILE%\.antigravity\extensions\redhat.java-1.53.2026021110-win32-x64\jre\21.0.10-win32-x86_64\bin\javac.exe
set JFX_PATH=%USERPROFILE%\.m2\repository\org\openjfx

"%JAVAC_EXE%" --module-path "%JFX_PATH%\javafx-controls\21\javafx-controls-21-win.jar;%JFX_PATH%\javafx-graphics\21\javafx-graphics-21-win.jar;%JFX_PATH%\javafx-base\21\javafx-base-21-win.jar;%JFX_PATH%\javafx-fxml\21\javafx-fxml-21-win.jar" --add-modules javafx.controls,javafx.fxml -cp "target\classes" "src\main\java\com\rustbuilder\service\TestAI.java" -d "target\classes"
if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed.
    exit /b %ERRORLEVEL%
)

"%JAVA_EXE%" --module-path "%JFX_PATH%\javafx-controls\21\javafx-controls-21-win.jar;%JFX_PATH%\javafx-graphics\21\javafx-graphics-21-win.jar;%JFX_PATH%\javafx-base\21\javafx-base-21-win.jar;%JFX_PATH%\javafx-fxml\21\javafx-fxml-21-win.jar" --add-modules javafx.controls,javafx.fxml -cp "target\classes" "com.rustbuilder.service.TestAI"

