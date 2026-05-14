package com.rustbuilder.ai.rl.supervisor;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.rustbuilder.ai.rl.RLModelManager;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class SupervisorDecisionLogWriterTest {
    private static final String MODEL_NAME = "codex_supervisor_log_writer_test";
    private static final Path MODEL_DIR = RLModelManager.getModelDirectory(MODEL_NAME);
    private static final Path LLM_DIR = RLModelManager.getModelLlmDirectory(MODEL_NAME);
    private static final Path CSV_PATH = LLM_DIR.resolve(MODEL_NAME + "_supervisor_decisions.csv");
    private static final Path JSONL_PATH = LLM_DIR.resolve(MODEL_NAME + "_supervisor_decisions.jsonl");
    private static final Path DEBUG_PATH = LLM_DIR.resolve(MODEL_NAME + "_supervisor_debug.jsonl");

    @AfterEach
    void cleanup() throws Exception {
        Files.deleteIfExists(CSV_PATH);
        Files.deleteIfExists(JSONL_PATH);
        Files.deleteIfExists(DEBUG_PATH);
        Files.deleteIfExists(LLM_DIR);
        Files.deleteIfExists(MODEL_DIR);
    }

    @Test
    void writesSupervisorDecisionCsvWithEscapedReason() throws Exception {
        SupervisorDecisionLogWriter writer = new SupervisorDecisionLogWriter();

        writer.write(MODEL_NAME, observation(), SupervisorDecision.keepGoing("keep, \"learning\""), false);

        String csv = Files.readString(CSV_PATH);
        assertTrue(csv.contains("timestamp,model_name,branch_id,episode,action,applied,reason"));
        assertTrue(csv.contains("confidence,risk_level,change_magnitude,requires_branch_test,expected_effect"));
        assertTrue(csv.contains("\"codex_supervisor_log_writer_test\",\"candidate\",10,\"KEEP_GOING\",false,\"keep, \"\"learning\"\"\""));
    }

    @Test
    void writesSupervisorDebugJsonWhenDiagnosticsProvided() throws Exception {
        SupervisorDecisionLogWriter writer = new SupervisorDecisionLogWriter();

        writer.write(
            MODEL_NAME,
            observation(),
            SupervisorDecision.keepGoing("keep learning"),
            true,
            java.util.Map.of(
                "provider", "builtin:gemini",
                "stages", java.util.List.of(java.util.Map.of("selectedModel", "gemma-4-31b-it"))));

        String debug = Files.readString(DEBUG_PATH);
        assertTrue(debug.contains("\"diagnostics\""));
        assertTrue(debug.contains("\"provider\":\"builtin:gemini\""));
        assertTrue(debug.contains("\"selectedModel\":\"gemma-4-31b-it\""));
    }

    @Test
    void upgradesExistingSupervisorCsvHeaderBeforeAppending() throws Exception {
        Files.createDirectories(LLM_DIR);
        Files.write(CSV_PATH, List.of(
            "timestamp,model_name,branch_id,episode,action,applied,reason,call_frequency,proposed_epsilon,has_reward_config,epsilon,best_score,avg_eval_score,invalid_rate,current_total_reward,training_elapsed_seconds,training_remaining_seconds,training_deadline",
            "\"2026-05-12 00:00:00\",\"codex_supervisor_log_writer_test\",\"candidate\",10,\"KEEP_GOING\",false,\"keep, \"\"old\"\"\",\"MEDIUM\",,false,0.100000,1.200000,0.300000,0.000000,1.500000,60.000,,\"2026-05-12T01:00:00Z\""));

        SupervisorDecisionLogWriter writer = new SupervisorDecisionLogWriter();

        writer.write(MODEL_NAME, observation(), SupervisorDecision.keepGoing("keep learning"), true);

        List<String> lines = Files.readAllLines(CSV_PATH);
        assertEquals(3, lines.size());
        assertTrue(lines.get(0).contains("confidence,risk_level,change_magnitude,requires_branch_test,expected_effect"));
        assertTrue(lines.get(1).contains("\"keep, \"\"old\"\"\""));
        assertTrue(lines.get(2).contains("\"KEEP_GOING\""));
        assertEquals(parseCsvLine(lines.get(0)).size(), parseCsvLine(lines.get(1)).size());
        assertEquals(parseCsvLine(lines.get(0)).size(), parseCsvLine(lines.get(2)).size());
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

    private static List<String> parseCsvLine(String line) {
        List<String> values = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean quoted = false;
        for (int i = 0; i < line.length(); i++) {
            char ch = line.charAt(i);
            if (ch == '"') {
                if (quoted && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else {
                    quoted = !quoted;
                }
            } else if (ch == ',' && !quoted) {
                values.add(current.toString());
                current.setLength(0);
            } else {
                current.append(ch);
            }
        }
        values.add(current.toString());
        return values;
    }
}
