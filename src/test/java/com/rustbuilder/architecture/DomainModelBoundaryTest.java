package com.rustbuilder.architecture;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class DomainModelBoundaryTest {

    @Test
    void placementServiceReturnsImmutablePlacementResult() throws IOException {
        String service = read("src/main/java/com/rustbuilder/service/physics/PlacementService.java");
        String result = read("src/main/java/com/rustbuilder/service/physics/PlacementResult.java");

        assertTrue(service.contains("public static PlacementResult calculatePlacement"));
        assertFalse(service.contains("public static class Placement"));
        assertFalse(service.contains("public boolean valid"));
        assertTrue(result.contains("public sealed interface PlacementResult"));
        assertTrue(result.contains("record Valid("));
        assertTrue(result.contains("record Invalid("));
    }

    private String read(String path) throws IOException {
        return Files.readString(Path.of(path), StandardCharsets.UTF_8);
    }
}
