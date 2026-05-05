@echo off
echo Running Rust Base Builder with CPU backend...
mvn -Pcpu clean javafx:run
pause
