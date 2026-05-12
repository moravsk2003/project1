package com.rustbuilder.ai.rl.supervisor;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.rustbuilder.ai.rl.RLModelManager;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class SupervisorDecisionLogWriterTest {
    private static final String MODEL_NAME = "codex_supervisor_log_writer_test";
    private static final Path MODEL_DIR = RLModelManager.getModelDirectory(MODEL_NAME);
    private static final Path CSV_PATH = MODEL_DIR.resolve(MODEL_NAME + "_supervisor_decisions.csv");
    private static final Path JSONL_PATH = MODEL_DIR.resolve(MODEL_NAME + "_supervisor_decisions.jsonl");

    @AfterEach
    void cleanup() throws Exception {
        Files.deleteIfExists(CSV_PATH);
        Files.deleteIfExists(JSONL_PATH);
        Files.deleteIfExists(MODEL_DIR);
    }

    @Test
    void writesSupervisorDecisionCsvWithEscapedReason() throws Exception {
        SupervisorDecisionLogWriter writer = new SupervisorDecisionLogWriter();

        writer.write(MODEL_NAME, observation(), SupervisorDecision.keepGoing("keep, \"learning\""), false);

        String csv = Files.readString(CSV_PATH);
        assertTrue(csv.contains("timestamp,model_name,branch_id,episode,action,applied,reason"));
        assertTrue(csv.contains("\"codex_supervisor_log_writer_test\",\"candidate\",10,\"KEEP_GOING\",false,\"keep, \"\"learning\"\"\""));
    }

    private SupervisorObservation observation() {
        return new SupervisorObservation(
            "candidate",
            10,
            -1.0,
            0.1,
            0.2,
            1.0,
            -0.5,
            2.0,
            0.0,
            0,
            10,
            0.99,
            0.1,
            32,
            5,
            true,
            1,
            6,
            true,
            1,
            6,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            100,
            "MAX_STEPS",
            "step",
            "final",
            "2026-05-12T00:00:00Z",
            "2026-05-12T00:01:00Z",
            "2026-05-12T01:00:00Z",
            60_000L,
            3_540_000L,
            true,
            false,
            java.util.Map.of(),
            java.util.Map.of(),
            null,
            null);
    }
}
