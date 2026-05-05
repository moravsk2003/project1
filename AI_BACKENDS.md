# AI backend selection

The project uses DeepLearning4J/ND4J for reinforcement-learning models.

## Default: auto-select GPU when available

The Maven project auto-selects the backend when it is imported by IntelliJ IDEA
or run from the command line:

- If `${env.SystemRoot}/System32/nvidia-smi.exe` exists, Maven activates the
  `cuda` profile.
- Otherwise Maven keeps the default `cpu` profile active.

After changing GPU drivers or switching machines, reload the Maven project in
IntelliJ IDEA so the IDE rebuilds the launcher classpath.

For command-line launch, use:

```bat
run.bat
```

`run.bat` checks whether `nvidia-smi` is available:

- If an NVIDIA GPU/driver is detected, it runs the CUDA backend.
- If no NVIDIA GPU/driver is detected, it runs the CPU backend.

When launching `com.rustbuilder.Launcher` directly from IntelliJ IDEA, the
selected Maven profile decides which ND4J backend is on the classpath.

## NVIDIA CUDA

Use CUDA explicitly on a machine with a compatible NVIDIA GPU driver/CUDA
runtime.

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
