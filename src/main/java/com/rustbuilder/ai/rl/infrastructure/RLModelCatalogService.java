package com.rustbuilder.ai.rl.infrastructure;

import com.rustbuilder.ai.rl.environment.spec.EncodingRuntimeConfig;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Builds compact, LLM-safe summaries of saved RL models.
 */
public final class RLModelCatalogService {
    private static final int MAX_MODELS = Integer.getInteger("rustbuilder.rl.llmCatalogMaxModels", 16);

    private RLModelCatalogService() {
    }

    public static List<Map<String, Object>> availableModelSummaries(EncodingRuntimeConfig currentConfig) {
        List<Map<String, Object>> summaries = new ArrayList<>();
        for (String modelName : RLModelManager.listModels()) {
            try {
                summaries.add(describeModel(modelName, currentConfig));
            } catch (Exception ignored) {
                // A corrupt or half-written model should not block LLM supervision.
            }
        }
        summaries.sort((a, b) -> {
            int compatible = Boolean.compare(booleanValue(b.get("compatible")), booleanValue(a.get("compatible")));
            if (compatible != 0) {
                return compatible;
            }
            return Double.compare(doubleValue(b.get("bestScore")), doubleValue(a.get("bestScore")));
        });
        if (summaries.size() > MAX_MODELS) {
            return List.copyOf(summaries.subList(0, MAX_MODELS));
        }
        return List.copyOf(summaries);
    }

    public static Map<String, Object> describeModel(String modelName,
                                                    EncodingRuntimeConfig currentConfig)
            throws IOException, ClassNotFoundException {
        RLModelManager.RLModel model = RLModelManager.loadMetadata(modelName);
        Path metaPath = RLModelManager.findExistingModelFile(modelName, modelName + ".rmeta");
        Path netPath = RLModelManager.findExistingModelFile(modelName, modelName + ".rnet");
        boolean hasNetworkWeights = Files.isRegularFile(netPath);
        String compatibilityIssue = compatibilityIssue(model, currentConfig);
        boolean compatible = compatibilityIssue.isBlank() && hasNetworkWeights;
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("name", model.name);
        map.put("metadataPath", metaPath.toString());
        map.put("hasMetadata", Files.isRegularFile(metaPath));
        map.put("hasNetworkWeights", hasNetworkWeights);
        map.put("compatible", compatible);
        map.put("compatibilityIssue", !hasNetworkWeights
            ? "missing_network_weights"
            : compatibilityIssue);
        map.put("episodesTrained", model.episodesTrained);
        map.put("bestScore", finiteOrZero(model.bestScore));
        map.put("savedEpsilon", finiteOrZero(model.epsilon));
        map.put("stateEncoderName", model.stateEncoderName);
        map.put("stateEncoderVersion", model.stateEncoderVersion);
        map.put("voxelChannels", model.voxelChannels);
        map.put("grid", model.gridWidth + "x" + model.gridHeight + "x" + model.gridFloors);
        map.put("actionShape", actionShape(model));
        map.put("tileIndexingMode", model.tileIndexingMode);
        map.put("hasGlobalVector", model.hasGlobalVector);
        map.put("globalFeatureCount", model.globalFeatureCount);
        map.put("use2dCnn", model.use2dCnn);
        map.put("bestBase", bestBaseSummary(model));
        map.put("recentTrainingLog", latestTrainingLogSummary(modelName));
        map.put("recommendationHint", recommendationHint(compatible, hasNetworkWeights, model));
        return map;
    }

    private static String compatibilityIssue(RLModelManager.RLModel model,
                                             EncodingRuntimeConfig currentConfig) {
        EncodingRuntimeConfig expectedConfig = expectedConfigFor(model, currentConfig);
        if (model == null || expectedConfig == null) {
            return "missing_runtime_config";
        }
        if (!equalsSafe(model.stateEncoderName, expectedConfig.stateEncodingSpec.encoderName)) {
            return "encoder_name_mismatch";
        }
        if (!equalsSafe(model.stateEncoderVersion, expectedConfig.stateEncodingSpec.encoderVersion)) {
            return "encoder_version_mismatch";
        }
        if (model.voxelChannels != expectedConfig.stateEncodingSpec.voxelChannels) {
            return "voxel_channels_mismatch";
        }
        if (model.gridWidth != expectedConfig.gridSpec.width
                || model.gridHeight != expectedConfig.gridSpec.height
                || model.gridFloors != expectedConfig.gridSpec.floors) {
            return "grid_shape_mismatch";
        }
        if (model.actionTypeCount != expectedConfig.actionSpaceSpec.typeCount
                || model.actionFloorCount != expectedConfig.actionSpaceSpec.floorCount
                || model.actionTileCount != expectedConfig.actionSpaceSpec.tileCount
                || model.actionRotationCount != expectedConfig.actionSpaceSpec.rotationCount
                || model.actionAimCount != expectedConfig.actionSpaceSpec.aimCount) {
            return "action_shape_mismatch";
        }
        if (!equalsSafe(model.tileIndexingMode, expectedConfig.actionSpaceSpec.tileIndexingMode.name())) {
            return "tile_indexing_mismatch";
        }
        if (model.hasGlobalVector != expectedConfig.stateEncodingSpec.hasGlobalVector
                || model.globalFeatureCount != expectedConfig.stateEncodingSpec.globalFeatureCount
                || model.hasObjectTable != expectedConfig.stateEncodingSpec.hasObjectTable
                || model.hasGraphState != expectedConfig.stateEncodingSpec.hasGraphState) {
            return "state_feature_mismatch";
        }
        return "";
    }

    private static EncodingRuntimeConfig expectedConfigFor(RLModelManager.RLModel model,
                                                           EncodingRuntimeConfig currentConfig) {
        if (model == null || model.stateEncoderVersion == null) {
            return currentConfig;
        }
        if ("v3".equalsIgnoreCase(model.stateEncoderVersion)) {
            return EncodingRuntimeConfig.createHybridV3Config();
        }
        if ("v2".equalsIgnoreCase(model.stateEncoderVersion)) {
            return EncodingRuntimeConfig.createVoxelV2Config();
        }
        if ("v1".equalsIgnoreCase(model.stateEncoderVersion)) {
            return EncodingRuntimeConfig.createVoxelV1Config();
        }
        return currentConfig;
    }

    private static Map<String, Object> latestTrainingLogSummary(String modelName) {
        Path logPath = RLModelManager.findExistingModelFile(modelName, modelName + "_multi_discrete_training.csv");
        if (!Files.isRegularFile(logPath)) {
            logPath = RLModelManager.findExistingModelFile(modelName, modelName + "_legacy_training.csv");
        }
        if (!Files.isRegularFile(logPath)) {
            return Map.of("present", false);
        }
        try {
            List<String> lines = Files.readAllLines(logPath, StandardCharsets.UTF_8);
            if (lines.size() <= 1) {
                return Map.of("present", true, "rows", 0);
            }
            String[] headers = splitCsv(lines.get(0));
            String[] last = splitCsv(lines.get(lines.size() - 1));
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("present", true);
            summary.put("path", logPath.toString());
            summary.put("rows", Math.max(0, lines.size() - 1));
            putColumn(summary, "lastEpoch", headers, last, "epoch");
            putColumn(summary, "lastEpisodesInEpoch", headers, last, "episodes");
            putColumn(summary, "lastAvgTotalReward", headers, last, "avg_total_reward");
            putColumn(summary, "lastBestRewardAllTime", headers, last, "best_reward_all_time");
            putColumn(summary, "lastBestEpisodeReward", headers, last, "best_ep_reward");
            putColumn(summary, "lastEpsilon", headers, last, "epsilon");
            putColumn(summary, "lastInvalidRatePct", headers, last, "invalid_rate_pct");
            putColumn(summary, "lastBestBaseBlocks", headers, last, "best_base_blocks");
            return summary;
        } catch (IOException e) {
            return Map.of("present", true, "error", e.getMessage());
        }
    }

    private static void putColumn(Map<String, Object> summary,
                                  String outputKey,
                                  String[] headers,
                                  String[] row,
                                  String column) {
        int idx = indexOf(headers, column);
        if (idx < 0 || idx >= row.length) {
            return;
        }
        String raw = row[idx] != null ? row[idx].trim() : "";
        if (raw.isBlank()) {
            return;
        }
        try {
            if (raw.contains(".")) {
                summary.put(outputKey, Double.parseDouble(raw));
            } else {
                summary.put(outputKey, Integer.parseInt(raw));
            }
        } catch (NumberFormatException e) {
            summary.put(outputKey, raw);
        }
    }

    private static int indexOf(String[] headers, String column) {
        if (headers == null || column == null) {
            return -1;
        }
        for (int i = 0; i < headers.length; i++) {
            if (column.equals(headers[i])) {
                return i;
            }
        }
        return -1;
    }

    private static String[] splitCsv(String line) {
        return line != null ? line.split(",", -1) : new String[0];
    }

    private static Map<String, Object> bestBaseSummary(RLModelManager.RLModel model) {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("hasEvalSnapshot", model.bestBaseEvalJson != null && !model.bestBaseEvalJson.isBlank());
        summary.put("hasRewardSnapshot", model.bestBaseRewardJson != null && !model.bestBaseRewardJson.isBlank());
        return summary;
    }

    private static String actionShape(RLModelManager.RLModel model) {
        return String.format(Locale.ROOT, "%d/%d/%d/%d/%d",
            model.actionTypeCount,
            model.actionFloorCount,
            model.actionTileCount,
            model.actionRotationCount,
            model.actionAimCount);
    }

    private static String recommendationHint(boolean compatible,
                                             boolean hasNetworkWeights,
                                             RLModelManager.RLModel model) {
        if (!hasNetworkWeights) {
            return "metadata_only_do_not_resume";
        }
        if (!compatible) {
            return "incompatible_do_not_resume";
        }
        if (model.episodesTrained <= 0) {
            return "compatible_but_untrained";
        }
        if (model.bestScore > 0.0) {
            return "strong_resume_candidate";
        }
        return "compatible_resume_candidate";
    }

    private static boolean equalsSafe(String a, String b) {
        return a != null ? a.equals(b) : b == null;
    }

    private static double finiteOrZero(double value) {
        return Double.isFinite(value) ? value : 0.0;
    }

    private static boolean booleanValue(Object value) {
        return value instanceof Boolean b && b;
    }

    private static double doubleValue(Object value) {
        return value instanceof Number number ? number.doubleValue() : 0.0;
    }
}
