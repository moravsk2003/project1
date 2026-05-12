package com.rustbuilder.architecture;

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
    void graphModuleDoesNotDependOnEvaluatorModule() throws IOException {
        assertNoReferences(Paths.get("service", "graph"), "com.rustbuilder.service.evaluator.");
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
}
