# AI backend selection

The project uses DeepLearning4J/ND4J for reinforcement-learning models.

## Default: CPU

The Maven project uses the CPU backend by default. This is important for IDE
runs, because `com.rustbuilder.Launcher` receives the classpath that IntelliJ
IDEA or VS Code builds before the JVM starts. Keeping CUDA off the default
classpath avoids startup crashes on Windows machines where the NVIDIA driver is
present but the CUDA native dependencies required by ND4J cannot be loaded.

After changing this file or switching branches, reload the Maven project in the
IDE so the launcher classpath is rebuilt.

Expected startup log for the default IDE run:

```text
Loaded [CpuBackend] backend
```

Expected startup log for GPU:

```text
Loaded [JCublasBackend] backend
Backend used: [CUDA]
```

If the log says `Loaded [CpuBackend] backend` while you intended to use CUDA,
the IDE is still running with the CPU dependency on its classpath. Activate the
`cuda` Maven profile and reload the Maven project.

If the log says:

```text
Skipped [JCublasBackend] backend (unavailable)
jnicudart.dll: Can't find dependent libraries
```

then the CUDA dependency is on the classpath, but Windows cannot load one of
the required native CUDA/NVIDIA/Visual C++ DLLs. Check that `nvidia-smi` works,
the NVIDIA driver is installed, and the Microsoft Visual C++ Redistributable is
available. To start the app without fixing the CUDA installation first, switch
back to the `cpu` profile.

For command-line launch, use:

```bat
run.bat
```

`run.bat` uses the CPU backend by default. To opt in to CUDA from the same
script, use:

```bat
run.bat gpu
```

You can also set:

```bat
set RUSTBUILDER_BACKEND=cuda
run.bat
```

## NVIDIA CUDA

Use CUDA explicitly from the command line on a machine with a compatible NVIDIA
GPU driver/CUDA runtime.

```bat
mvn -Pcuda clean javafx:run
```

or:

```bat
run_gpu.bat
```

The `cuda` profile uses:

```xml
org.nd4j:nd4j-cuda-11.6-platform
org.nd4j:nd4j-native-platform
```

## CPU fallback

Use the CPU backend on a machine without a compatible NVIDIA CUDA setup.

```bat
mvn -Pcpu clean javafx:run
```

or:

```bat
run_cpu.bat
```

The `cpu` profile uses:

```xml
org.nd4j:nd4j-native-platform
```
