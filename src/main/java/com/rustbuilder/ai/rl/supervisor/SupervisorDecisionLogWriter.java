package com.rustbuilder.ai.rl.supervisor;

import com.rustbuilder.ai.rl.RLModelManager;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;

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
        "proposed_epsilon",
        "has_reward_config",
        "epsilon",
        "best_score",
        "avg_eval_score",
        "invalid_rate",
        "current_total_reward",
        "training_elapsed_seconds",
        "training_remaining_seconds",
        "training_deadline");

    public void write(String modelName, SupervisorObservation observation, SupervisorDecision decision) {
        write(modelName, observation, decision, true);
    }

    public void write(String modelName, SupervisorObservation observation, SupervisorDecision decision, boolean applied) {
        if (modelName == null || modelName.isBlank() || observation == null || decision == null) {
            return;
        }

        Path dir = RLModelManager.getModelDirectory(modelName);
        Path jsonlPath = dir.resolve(modelName + "_supervisor_decisions.jsonl");
        Path csvPath = dir.resolve(modelName + "_supervisor_decisions.csv");
        try {
            Files.createDirectories(dir);
            Files.writeString(jsonlPath, toJsonLine(modelName, observation, decision, applied) + System.lineSeparator(),
                StandardCharsets.UTF_8,
                StandardOpenOption.WRITE,
                Files.exists(jsonlPath)
                    ? StandardOpenOption.APPEND
                    : StandardOpenOption.CREATE);
            appendCsvLine(csvPath, modelName, observation, decision, applied);
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
        return String.format(Locale.US,
            "{\"timestamp\":\"%s\",\"model_name\":\"%s\",\"branch_id\":\"%s\","
                + "\"episode\":%d,\"action\":\"%s\",\"applied\":%b,\"reason\":\"%s\","
                + "\"epsilon\":%.6f,\"best_score\":%.6f,\"avg_eval_score\":%.6f,"
                + "\"invalid_rate\":%.6f,\"current_total_reward\":%.6f}",
            escape(LocalDateTime.now().format(TS_FMT)),
            escape(modelName),
            escape(obs.branchId),
            obs.totalEpisodesTrained,
            decision.getAction().name(),
            applied,
            escape(decision.getReason()),
            obs.epsilon,
            obs.bestScore,
            obs.avgEvalScore,
            obs.invalidActionRate,
            obs.currentEpisodeStepReward + obs.currentEpisodeFinalReward);
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
            decision.getProposedEpsilon() != null
                ? String.format(Locale.US, "%.6f", decision.getProposedEpsilon())
                : "",
            String.valueOf(decision.getProposedRewardConfig() != null),
            String.format(Locale.US, "%.6f", obs.epsilon),
            String.format(Locale.US, "%.6f", obs.bestScore),
            String.format(Locale.US, "%.6f", obs.avgEvalScore),
            String.format(Locale.US, "%.6f", obs.invalidActionRate),
            String.format(Locale.US, "%.6f", obs.currentEpisodeStepReward + obs.currentEpisodeFinalReward),
            String.format(Locale.US, "%.3f", obs.trainingElapsedMs / 1000.0),
            obs.trainingRemainingMs >= 0
                ? String.format(Locale.US, "%.3f", obs.trainingRemainingMs / 1000.0)
                : "",
            csv(obs.trainingDeadlineIso));
    }

    private static String escape(String value) {
        if (value == null) {
            return "";
        }
        return value
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\r", "\\r")
            .replace("\n", "\\n");
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
