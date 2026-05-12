package com.rustbuilder.ai.rl.supervisor;

import com.rustbuilder.ai.core.TrainingMetrics;
import com.rustbuilder.ai.rl.EpisodeResult;
import java.time.Instant;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Converts rich training internals into a compact observation for LLM review.
 */
public class RLTrainingAnalyzer {
    public SupervisorObservation summarize(String branchId, TrainingMetrics metrics, EpisodeResult result, com.rustbuilder.ai.rl.RLRewardConfig currentConfig) {
        return summarize(branchId, metrics, result, currentConfig, Collections.emptyMap(), Collections.emptyMap());
    }

    public SupervisorObservation summarize(String branchId,
                                           TrainingMetrics metrics,
                                           EpisodeResult result,
                                           com.rustbuilder.ai.rl.RLRewardConfig currentConfig,
                                           Map<String, Object> trainingContext,
                                           Map<String, Object> trendMetrics) {
        if (metrics == null) {
            throw new IllegalArgumentException("metrics must not be null");
        }

        return new SupervisorObservation(
            branchId,
            metrics.totalEpisodesTrained,
            metrics.bestScore,
            metrics.avgEvalScore,
            metrics.currentEpisodeEvalScore,
            metrics.currentEpisodeStepReward,
            metrics.currentEpisodeFinalReward,
            metrics.bestTotalReward,
            metrics.invalidActionRate,
            metrics.lastEpisodeInvalidActions,
            metrics.lastEpisodeTotalActions,
            metrics.epsilon,
            metrics.lastTrainLoss,
            metrics.memorySize,
            metrics.bestBaseBlocks,
            metrics.bestBaseHasTC,
            metrics.bestBaseDoors,
            result != null ? result.blocksPlaced : 0,
            result != null && result.hasTC,
            result != null ? result.componentCount : 0,
            result != null ? result.mainComponentBlocks : 0,
            result != null ? result.evalLogisticsScore : 0.0,
            result != null ? result.evalCostScore : 0.0,
            result != null ? result.evalRaidScore : 0.0,
            result != null ? result.evalWorkingAreaScore : 0.0,
            result != null ? result.evalSafeZoneScore : 0.0,
            result != null ? result.raidSulfurToTC : -1,
            result != null && result.stopReason != null ? result.stopReason.name() : "UNKNOWN",
            metrics.currentEpisodeStepRewardBreakdown,
            metrics.currentEpisodeFinalRewardBreakdown,
            toIso(metrics.trainingStartEpochMs),
            toIso(metrics.currentTimeEpochMs),
            metrics.trainingTimeLimitEnabled ? toIso(metrics.trainingDeadlineEpochMs) : "",
            metrics.trainingElapsedMs,
            metrics.trainingRemainingMs,
            metrics.trainingTimeLimitEnabled,
            metrics.trainingTimeLimitReached,
            summarizeEnumMap(result != null ? result.errorStats : null),
            summarizeEnumMap(result != null ? result.typeStats : null),
            currentConfig != null ? currentConfig.toMap() : null,
            trainingContext,
            trendMetrics,
            null
        );
    }

    private String toIso(long epochMs) {
        if (epochMs <= 0) {
            return "";
        }
        return Instant.ofEpochMilli(epochMs).toString();
    }

    private Map<String, Integer> summarizeEnumMap(Map<?, Integer> source) {
        if (source == null || source.isEmpty()) {
            return Map.of();
        }
        return source.entrySet().stream()
            .filter(entry -> entry.getValue() != null && entry.getValue() > 0)
            .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
            .limit(8)
            .collect(Collectors.toMap(
                entry -> String.valueOf(entry.getKey()),
                Map.Entry::getValue,
                (a, b) -> a,
                LinkedHashMap::new));
    }
}
