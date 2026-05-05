@echo off
echo Running Rust Base Builder with CUDA backend...
mvn -Pcuda clean javafx:run
pause
