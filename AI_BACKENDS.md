# AI backend selection

The project uses DeepLearning4J/ND4J for reinforcement-learning models.

## Default: NVIDIA CUDA

The Maven project uses the NVIDIA CUDA backend by default. This is important
for IDE runs, because `com.rustbuilder.Launcher` receives the classpath that
IntelliJ IDEA or VS Code builds before the JVM starts.

After changing this file or switching branches, reload the Maven project in the
IDE so the launcher classpath is rebuilt.

Expected startup log for GPU:

```text
Loaded [JCublasBackend] backend
Backend used: [CUDA]
```

If the log says `Loaded [CpuBackend] backend`, the IDE is still running with the
CPU dependency on its classpath.

If the log says:

```text
Skipped [JCublasBackend] backend (unavailable)
jnicudart.dll: Can't find dependent libraries
```

then the CUDA dependency is on the classpath, but Windows cannot load one of
the required native CUDA/NVIDIA/Visual C++ DLLs. Check that `nvidia-smi` works,
the NVIDIA driver is installed, and the Microsoft Visual C++ Redistributable is
available. The `cuda` profile also includes `nd4j-native-platform` as a fallback
so the application can still start on CPU if CUDA native loading fails.

For command-line launch, use:

```bat
run.bat
```

`run.bat` checks whether `nvidia-smi` is available:

- If an NVIDIA GPU/driver is detected, it runs the CUDA backend.
- If no NVIDIA GPU/driver is detected, it runs the CPU backend.

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
