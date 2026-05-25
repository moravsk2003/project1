package com.rustbuilder.ai.rl.supervisor.application;

import com.rustbuilder.ai.rl.infrastructure.RLModelManager;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class HistoricalTrainingReportService {

    public boolean modelMetadataExists(String modelName) {
        if (modelName == null || modelName.isBlank()) {
            return false;
        }
        Path metaPath = RLModelManager.findExistingModelFile(modelName, modelName + ".rmeta");
        return Files.exists(metaPath);
    }

    public String generateHistoricalReport(String modelName, Integer startEpoch, Integer endEpoch) {
        if (modelName == null || modelName.isBlank()) {
            return "Error: Model name not provided.";
        }

        Path logPath = findTrainingLog(modelName);
        if (logPath == null) {
            return "Error: Could not find training logs for model: " + modelName;
        }

        try {
            return summarize(modelName, startEpoch, endEpoch, Files.readAllLines(logPath, StandardCharsets.UTF_8));
        } catch (IOException e) {
            return "Error: Failed to read training logs: " + e.getMessage();
        }
    }

    private Path findTrainingLog(String modelName) {
        Path legacyLog = RLModelManager.findExistingModelFile(modelName, modelName + "_legacy_training.csv");
        if (Files.exists(legacyLog)) {
            return legacyLog;
        }

        Path multiDiscreteLog = RLModelManager.findExistingModelFile(modelName, modelName + "_multi_discrete_training.csv");
        return Files.exists(multiDiscreteLog) ? multiDiscreteLog : null;
    }

    private String summarize(String modelName, Integer startEpoch, Integer endEpoch, List<String> lines) {
        if (lines.size() <= 1) {
            return "Error: Log file is empty.";
        }

        List<String> headers = parseCsvLine(lines.get(0));
        int epochIdx = headers.indexOf("epoch");
        int avgTotalRewardIdx = headers.indexOf("avg_total_reward");
        int epsilonIdx = headers.indexOf("epsilon");
        int invalidRateIdx = headers.indexOf("invalid_rate_pct");

        if (epochIdx == -1 || avgTotalRewardIdx == -1) {
            return "Error: Invalid CSV format.";
        }

        int safeStartEpoch = startEpoch != null ? startEpoch : 0;
        int safeEndEpoch = endEpoch != null ? endEpoch : Integer.MAX_VALUE;

        int count = 0;
        double startReward = 0.0;
        double endReward = 0.0;
        double minReward = Double.MAX_VALUE;
        double maxReward = -Double.MAX_VALUE;

        StringBuilder summary = new StringBuilder();
        summary.append("Historical Report for ")
            .append(modelName)
            .append(" (Epochs ")
            .append(safeStartEpoch)
            .append("-")
            .append(safeEndEpoch == Integer.MAX_VALUE ? "End" : safeEndEpoch)
            .append("):\n");

        for (int i = 1; i < lines.size(); i++) {
            List<String> parts = parseCsvLine(lines.get(i));
            if (parts.size() <= avgTotalRewardIdx || parts.size() <= epochIdx) {
                continue;
            }

            try {
                int epoch = Integer.parseInt(parts.get(epochIdx));
                if (epoch < safeStartEpoch || epoch > safeEndEpoch) {
                    continue;
                }

                double reward = Double.parseDouble(parts.get(avgTotalRewardIdx));
                if (count == 0) {
                    startReward = reward;
                }
                endReward = reward;
                minReward = Math.min(minReward, reward);
                maxReward = Math.max(maxReward, reward);
                count++;

                if (count <= 5 || count % 10 == 0 || i == lines.size() - 1) {
                    String epsilon = valueAt(parts, epsilonIdx, "N/A");
                    String invalidRate = valueAt(parts, invalidRateIdx, "N/A");
                    summary.append(String.format(
                        "Epoch %d: Avg Reward %.4f, Epsilon %s, Invalid %s%%\n",
                        epoch,
                        reward,
                        epsilon,
                        invalidRate));
                }
            } catch (NumberFormatException ignored) {
                // Skip malformed rows; partial logs should still produce a useful report.
            }
        }

        if (count == 0) {
            return "No data found in the specified epoch range.";
        }

        summary.append(String.format("\nSummary over %d epochs:\n", count));
        summary.append(String.format("Start Reward: %.4f | End Reward: %.4f\n", startReward, endReward));
        summary.append(String.format("Min Reward: %.4f | Max Reward: %.4f\n", minReward, maxReward));
        summary.append(endReward > startReward ? "Trend: IMPROVING\n" : "Trend: STAGNANT/DEGRADING\n");
        return summary.toString();
    }

    private static String valueAt(List<String> parts, int index, String fallback) {
        return index >= 0 && index < parts.size() ? parts.get(index) : fallback;
    }

    private static List<String> parseCsvLine(String line) {
        List<String> values = new ArrayList<>();
        if (line == null) {
            return values;
        }

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
