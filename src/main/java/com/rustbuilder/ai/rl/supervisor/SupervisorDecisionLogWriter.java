package com.rustbuilder.ai.rl.supervisor;

import com.rustbuilder.ai.rl.RLModelManager;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Appends supervisor observations and accepted decisions as JSON lines and CSV.
 */
public class SupervisorDecisionLogWriter {
    private static final DateTimeFormatter TS_FMT =
        DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private static final String CSV_HEADER = String.join(",",
        "timestamp",
        "model_name",
        "branch_id",
        "episode",
        "action",
        "applied",
        "reason",
        "call_frequency",
        "proposed_epsilon",
        "has_reward_config",
        "epsilon",
        "best_score",
        "avg_eval_score",
        "invalid_rate",
        "current_total_reward",
        "confidence",
        "risk_level",
        "change_magnitude",
        "requires_branch_test",
        "expected_effect",
        "training_elapsed_seconds",
        "training_remaining_seconds",
        "training_deadline");

    public void write(String modelName, SupervisorObservation observation, SupervisorDecision decision) {
        write(modelName, observation, decision, true);
    }

    public void write(String modelName, SupervisorObservation observation, SupervisorDecision decision, boolean applied) {
        write(modelName, observation, decision, applied, Map.of());
    }

    public void write(String modelName,
                      SupervisorObservation observation,
                      SupervisorDecision decision,
                      boolean applied,
                      Map<String, Object> diagnostics) {
        if (modelName == null || modelName.isBlank() || observation == null || decision == null) {
            return;
        }

        Path dir = RLModelManager.getModelLlmDirectory(modelName);
        Path jsonlPath = dir.resolve(modelName + "_supervisor_decisions.jsonl");
        Path csvPath = dir.resolve(modelName + "_supervisor_decisions.csv");
        Path debugPath = dir.resolve(modelName + "_supervisor_debug.jsonl");
        try {
            Files.createDirectories(dir);
            Files.writeString(jsonlPath, toJsonLine(modelName, observation, decision, applied) + System.lineSeparator(),
                StandardCharsets.UTF_8,
                StandardOpenOption.WRITE,
                Files.exists(jsonlPath)
                    ? StandardOpenOption.APPEND
                    : StandardOpenOption.CREATE);
            appendCsvLine(csvPath, modelName, observation, decision, applied);
            if (diagnostics != null && !diagnostics.isEmpty()) {
                Files.writeString(debugPath, toDebugJsonLine(modelName, observation, decision, applied, diagnostics) + System.lineSeparator(),
                    StandardCharsets.UTF_8,
                    StandardOpenOption.WRITE,
                    Files.exists(debugPath)
                        ? StandardOpenOption.APPEND
                        : StandardOpenOption.CREATE);
            }
        } catch (IOException ignored) {
            // Supervisor logs must never break training.
        }
    }

    private void appendCsvLine(Path path,
                               String modelName,
                               SupervisorObservation obs,
                               SupervisorDecision decision,
                               boolean applied) throws IOException {
        boolean needsHeader = !Files.exists(path) || Files.size(path) == 0;
        StringBuilder sb = new StringBuilder();
        if (needsHeader) {
            sb.append(CSV_HEADER).append(System.lineSeparator());
        }
        sb.append(toCsvLine(modelName, obs, decision, applied)).append(System.lineSeparator());
        Files.writeString(path, sb.toString(),
            StandardCharsets.UTF_8,
            StandardOpenOption.WRITE,
            Files.exists(path)
                ? StandardOpenOption.APPEND
                : StandardOpenOption.CREATE);
    }

    private String toJsonLine(String modelName, SupervisorObservation obs, SupervisorDecision decision, boolean applied) {
        Map<String, Object> map = baseDecisionMap(modelName, obs, decision, applied);
        return SimpleJson.stringify(map);
    }

    private String toDebugJsonLine(String modelName,
                                   SupervisorObservation obs,
                                   SupervisorDecision decision,
                                   boolean applied,
                                   Map<String, Object> diagnostics) {
        Map<String, Object> map = baseDecisionMap(modelName, obs, decision, applied);
        map.put("diagnostics", diagnostics);
        return SimpleJson.stringify(map);
    }

    private Map<String, Object> baseDecisionMap(String modelName,
                                                SupervisorObservation obs,
                                                SupervisorDecision decision,
                                                boolean applied) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("timestamp", LocalDateTime.now().format(TS_FMT));
        map.put("model_name", modelName);
        map.put("branch_id", obs.branchId);
        map.put("episode", obs.totalEpisodesTrained);
        map.put("action", decision.getAction().name());
        map.put("applied", applied);
        map.put("reason", decision.getReason());
        map.put("call_frequency", callFrequencyName(decision));
        if (decision.getProposedEpsilon() != null) {
            map.put("proposed_epsilon", decision.getProposedEpsilon());
        }
        map.put("has_reward_config", decision.getProposedRewardConfig() != null);
        map.put("epsilon", obs.epsilon);
        map.put("best_score", obs.bestScore);
        map.put("avg_eval_score", obs.avgEvalScore);
        map.put("invalid_rate", obs.invalidActionRate);
        map.put("current_total_reward", obs.currentEpisodeStepReward + obs.currentEpisodeFinalReward);
        if (decision.getConfidence() != null) {
            map.put("confidence", decision.getConfidence());
        }
        map.put("risk_level", decision.getRiskLevel());
        map.put("expected_effect", decision.getExpectedEffect());
        map.put("rollback_plan", decision.getRollbackPlan());
        map.put("change_magnitude", decision.getChangeMagnitude());
        if (decision.getRequiresBranchTest() != null) {
            map.put("requires_branch_test", decision.getRequiresBranchTest());
        }
        map.put("training_elapsed_seconds", obs.trainingElapsedMs / 1000.0);
        map.put("training_remaining_seconds", obs.trainingRemainingMs >= 0 ? obs.trainingRemainingMs / 1000.0 : null);
        map.put("training_deadline", obs.trainingDeadlineIso);
        return map;
    }

    private String toCsvLine(String modelName, SupervisorObservation obs, SupervisorDecision decision, boolean applied) {
        return String.join(",",
            csv(LocalDateTime.now().format(TS_FMT)),
            csv(modelName),
            csv(obs.branchId),
            String.valueOf(obs.totalEpisodesTrained),
            csv(decision.getAction().name()),
            String.valueOf(applied),
            csv(decision.getReason()),
            csv(callFrequencyName(decision)),
            decision.getProposedEpsilon() != null
                ? String.format(Locale.US, "%.6f", decision.getProposedEpsilon())
                : "",
            String.valueOf(decision.getProposedRewardConfig() != null),
            String.format(Locale.US, "%.6f", obs.epsilon),
            String.format(Locale.US, "%.6f", obs.bestScore),
            String.format(Locale.US, "%.6f", obs.avgEvalScore),
            String.format(Locale.US, "%.6f", obs.invalidActionRate),
            String.format(Locale.US, "%.6f", obs.currentEpisodeStepReward + obs.currentEpisodeFinalReward),
            decision.getConfidence() != null
                ? String.format(Locale.US, "%.3f", decision.getConfidence())
                : "",
            csv(decision.getRiskLevel()),
            csv(decision.getChangeMagnitude()),
            decision.getRequiresBranchTest() != null ? String.valueOf(decision.getRequiresBranchTest()) : "",
            csv(decision.getExpectedEffect()),
            String.format(Locale.US, "%.3f", obs.trainingElapsedMs / 1000.0),
            obs.trainingRemainingMs >= 0
                ? String.format(Locale.US, "%.3f", obs.trainingRemainingMs / 1000.0)
                : "",
            csv(obs.trainingDeadlineIso));
    }

    private static String callFrequencyName(SupervisorDecision decision) {
        return decision.getProposedCallFrequency() != null
            ? decision.getProposedCallFrequency().name()
            : "";
    }

    private static String csv(String value) {
        if (value == null) {
            return "";
        }
        return "\"" + value
            .replace("\"", "\"\"")
            .replace("\r", " ")
            .replace("\n", " ") + "\"";
    }
}
