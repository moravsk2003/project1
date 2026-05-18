# Build

The canonical local verification path is Maven with the CPU profile:

```bash
mvn -Pcpu clean test
```

The CPU profile is active by default and uses `nd4j-native-platform`, so it is the
portable path for machines without CUDA.

CUDA is opt-in:

```bash
mvn -Pcuda clean test
```

Use the CUDA profile only on machines with a compatible NVIDIA CUDA setup.
