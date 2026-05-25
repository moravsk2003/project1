package com.rustbuilder.architecture;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

class ArchitectureBoundaryTest {

    private static final Path MAIN_SOURCE = Paths.get("src", "main", "java", "com", "rustbuilder");

    @Test
    void modelModuleDoesNotReachIntoApplicationModules() throws IOException {
        assertNoReferences("model",
                "com.rustbuilder.service.",
                "com.rustbuilder.ai.",
                "com.rustbuilder.ui.",
                "com.rustbuilder.controller.");
    }

    @Test
    void serviceModuleDoesNotDependOnAiOrPresentation() throws IOException {
        assertNoReferences("service",
                "com.rustbuilder.ai.",
                "com.rustbuilder.ui.",
                "com.rustbuilder.controller.");
    }

    @Test
    void aiModuleDoesNotDependOnPresentation() throws IOException {
        assertNoReferences("ai",
                "com.rustbuilder.ui.",
                "com.rustbuilder.controller.");
    }

    @Test
    void generationStrategiesStaySeparated() throws IOException {
        assertNoReferences(Paths.get("ai", "rl"), "com.rustbuilder.ai.ea.");
        assertNoReferences(Paths.get("ai", "ea"), "com.rustbuilder.ai.rl.");
    }

    @Test
    void llmProviderAdaptersDoNotOrchestrateTrainingRuntime() throws IOException {
        assertNoReferences(Paths.get("ai", "rl", "supervisor", "provider", "gemini"),
                "com.rustbuilder.ai.rl.application.RLTrainingService",
                "com.rustbuilder.ai.rl.infrastructure.RLModelManager",
                "com.rustbuilder.ai.rl.supervisor.application.LlmOrchestrator");
    }

    @Test
    void graphModuleDoesNotDependOnEvaluatorModule() throws IOException {
        assertNoReferences(Paths.get("service", "graph"), "com.rustbuilder.service.evaluator.");
    }

    @Test
    void llmOrchestratorDelegatesHistoricalLogAccess() throws IOException {
        String source = read("src/main/java/com/rustbuilder/ai/rl/supervisor/application/LlmOrchestrator.java");

        assertFalse(source.contains("RLModelManager"));
        assertFalse(source.contains("Files."));
        assertFalse(source.contains(".split("));
        assertFalse(source.contains("new SupervisorObservation("));
        assertFalse(source.contains("new Thread("));
        assertFalse(source.contains("Thread.sleep("));
    }

    @Test
    void multiDiscreteDqnAgentDoesNotInspectGameObjectTypes() throws IOException {
        String source = read("src/main/java/com/rustbuilder/ai/rl/policy/multidiscrete/MultiDiscreteDQNAgent.java");

        assertFalse(source.contains("BuildingType.TC"));
        assertFalse(source.contains("BuildingType.LOOT_ROOM"));
        assertFalse(source.contains("com.rustbuilder.model.core.BuildingType"));
    }

    @Test
    void gridModelDelegatesCollisionPlacementAndSpatialDetails() throws IOException {
        String source = read("src/main/java/com/rustbuilder/model/GridModel.java");

        assertFalse(source.contains("private static final class LongBlockListMap"));
        assertFalse(source.contains("SocketCompatibilityUtils"));
        assertFalse(source.contains("BuildingType.TC"));
        assertFalse(source.contains("BuildingType.WORKBENCH"));
        assertFalse(source.contains("BuildingType.LOOT_ROOM"));
    }

    private static void assertNoReferences(String modulePath, String... forbiddenReferences) throws IOException {
        assertNoReferences(Paths.get(modulePath), forbiddenReferences);
    }

    private static void assertNoReferences(Path modulePath, String... forbiddenReferences) throws IOException {
        Path root = MAIN_SOURCE.resolve(modulePath);
        List<String> violations = new ArrayList<>();

        try (Stream<Path> files = Files.walk(root)) {
            for (Path file : files.filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .toList()) {
                String content = Files.readString(file, StandardCharsets.UTF_8);
                for (String forbidden : forbiddenReferences) {
                    if (content.contains(forbidden)) {
                        violations.add(root.relativize(file) + " references " + forbidden);
                    }
                }
            }
        }

        assertTrue(violations.isEmpty(), () -> String.join(System.lineSeparator(), violations));
    }

    private static String read(String path) throws IOException {
        return Files.readString(Path.of(path), StandardCharsets.UTF_8);
    }
}
