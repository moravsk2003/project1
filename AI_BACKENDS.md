# AI Backend Selection

The project uses DeepLearning4J and ND4J (version `1.0.0-M2.1`) for reinforcement-learning models.

ND4J selects its execution backend dynamically at runtime using Java's `ServiceLoader` mechanism, loading whichever backend platform implementation is present on the classpath.

---

## Classpath Entry Points

The project has two distinct entry points, depending on how the application is executed:

1. **`com.rustbuilder.Launcher`**: A standard Java launcher class (not extending JavaFX `Application`). Running this class is the recommended way to start the app from an IDE (like IntelliJ IDEA or VS Code). It receives the classpath built by the IDE and forwards execution to `MainApp`, bypassing JavaFX restrictions on running application subclasses directly from the classpath.
2. **`com.rustbuilder.MainApp`**: The main JavaFX application class. This entry point is used directly by the Maven plugin (`javafx:run`) configured in [pom.xml](file:///d:/project/pom.xml#L119), which handles modular dependencies and JVM parameters automatically.

---

## Maven Profiles & Dependencies

The backend implementation is determined by which Maven profile is active during compile/run:

### 1. Default Profile: `cpu`
- **Active by default**: Keeps the classpath lightweight and stable on machines without NVIDIA GPUs.
- **Dependencies**: Includes `nd4j-native-platform`.
- **Declaration** in [pom.xml](file:///d:/project/pom.xml#L131-L148):
  ```xml
  <dependency>
      <groupId>org.nd4j</groupId>
      <artifactId>nd4j-native-platform</artifactId>
      <version>${dl4j.version}</version>
  </dependency>
  ```

### 2. CUDA Profile: `cuda`
- **Opt-in**: Used on machines equipped with compatible NVIDIA GPU drivers and a CUDA toolkit.
- **Dependencies**: Includes `nd4j-cuda-11.6-platform` and `nd4j-native-platform` (as fallback).
- **Declaration** in [pom.xml](file:///d:/project/pom.xml#L150-L171):
  ```xml
  <dependency>
      <groupId>org.nd4j</groupId>
      <artifactId>nd4j-cuda-11.6-platform</artifactId>
      <version>${dl4j.version}</version>
  </dependency>
  <dependency>
      <groupId>org.nd4j</groupId>
      <artifactId>nd4j-native-platform</artifactId>
      <version>${dl4j.version}</version>
  </dependency>
  ```

---

## Log Verification

Check the SLF4J/Logback logs during application startup to confirm which ND4J backend has loaded.

### Expected logs for CPU (`cpu` profile):
```text
[main] INFO org.nd4j.linalg.factory.Nd4jBackend - Loaded [CpuBackend] backend
[main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Backend used: [CPU]; OS: [Windows 10]
```

### Expected logs for GPU (`cuda` profile):
```text
[main] INFO org.nd4j.linalg.factory.Nd4jBackend - Loaded [JCublasBackend] backend
[main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Backend used: [CUDA]; OS: [Windows 10]
```

### Troubleshooting CUDA initialization:
If the `cuda` profile is active but ND4J fails to load it, you will see a warning in the logs:
```text
[main] WARN org.nd4j.linalg.factory.Nd4jBackend - Skipped [JCublasBackend] backend (unavailable)
```
Followed by a stack trace indicating a linking failure (e.g., `java.lang.UnsatisfiedLinkError: no jnicudart in java.library.path` or similar missing native DLLs). 

If this happens:
- Confirm that `nvidia-smi` works on your command line.
- Verify that Microsoft Visual C++ Redistributable is installed.
- Ensure that the required CUDA toolkit version is installed and visible on the PATH.
- If you need to temporarily bypass CUDA errors and launch the application, switch back to the `cpu` profile.

---

## Command-Line Execution

To run the application from the command line, use the provided batch scripts in the project root:

- **`run.bat`**: The central launcher script. It automatically searches for a local JDK (including IDE-bundled runtimes) and detects Maven (checking for `mvnw.cmd` or local/IDE Maven installations).
  - Runs the CPU profile by default:
    ```cmd
    run.bat
    ```
  - Force the CPU profile explicitly:
    ```cmd
    run.bat cpu
    ```
  - Opt-in to the CUDA backend:
    ```cmd
    run.bat gpu
    ```
    or
    ```cmd
    run.bat cuda
    ```
  - Override via environment variables:
    ```cmd
    set RUSTBUILDER_BACKEND=cuda
    run.bat
    ```
    *(Note: explicit command-line arguments like `cpu` override the `RUSTBUILDER_BACKEND` environment variable)*

- **`run_cpu.bat`**: A convenience script that delegates to `run.bat cpu` to run with the CPU backend.
- **`run_gpu.bat`**: A convenience script that delegates to `run.bat gpu` to run with the CUDA GPU backend.

---

## Manual Maven Commands

If Maven is configured globally in your command line, you can start the application directly:

### Run on CPU
```cmd
mvn -Pcpu clean javafx:run
```

### Run on GPU (CUDA)
```cmd
mvn -Pcuda clean javafx:run
```
