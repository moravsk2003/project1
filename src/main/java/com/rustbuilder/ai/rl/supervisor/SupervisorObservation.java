package com.rustbuilder.ai.rl.supervisor;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Compact, LLM-safe snapshot of one training branch.
 */
public final class SupervisorObservation {
    public final String branchId;
    public final int totalEpisodesTrained;
    public final double bestScore;
    public final double avgEvalScore;
    public final double currentEpisodeEvalScore;
    public final double currentEpisodeStepReward;
    public final double currentEpisodeFinalReward;
    public final double bestTotalReward;
    public final double invalidActionRate;
    public final int lastEpisodeInvalidActions;
    public final int lastEpisodeTotalActions;
    public final double epsilon;
    public final double lastTrainLoss;
    public final int memorySize;
    public final int bestBaseBlocks;
    public final boolean bestBaseHasTC;
    public final int bestBaseDoors;
    public final int episodeBlocksPlaced;
    public final boolean episodeHasTC;
    public final int componentCount;
    public final int mainComponentBlocks;
    public final double evalLogisticsScore;
    public final double evalCostScore;
    public final double evalRaidScore;
    public final double evalWorkingAreaScore;
    public final double evalSafeZoneScore;
    public final int raidSulfurToTC;
    public final String stopReason;
    public final String stepRewardBreakdown;
    public final String finalRewardBreakdown;
    public final String trainingStartTimeIso;
    public final String currentTimeIso;
    public final String trainingDeadlineIso;
    public final long trainingElapsedMs;
    public final long trainingRemainingMs;
    public final boolean trainingTimeLimitEnabled;
    public final boolean trainingTimeLimitReached;
    public final Map<String, Integer> invalidActionReasons;
    public final Map<String, Integer> actionTypeCounts;
    public final Map<String, Double> currentRewardConfig;
    public final Map<String, Object> trainingContext;
    public final Map<String, Object> trendMetrics;
    public final String historicalReport;

    public SupervisorObservation(String branchId,
                                 int totalEpisodesTrained,
                                 double bestScore,
                                 double avgEvalScore,
                                 double currentEpisodeEvalScore,
                                 double currentEpisodeStepReward,
                                 double currentEpisodeFinalReward,
                                 double bestTotalReward,
                                 double invalidActionRate,
                                 int lastEpisodeInvalidActions,
                                 int lastEpisodeTotalActions,
                                 double epsilon,
                                 double lastTrainLoss,
                                 int memorySize,
                                 int bestBaseBlocks,
                                 boolean bestBaseHasTC,
                                 int bestBaseDoors,
                                 int episodeBlocksPlaced,
                                 boolean episodeHasTC,
                                 int componentCount,
                                 int mainComponentBlocks,
                                 double evalLogisticsScore,
                                 double evalCostScore,
                                 double evalRaidScore,
                                 double evalWorkingAreaScore,
                                 double evalSafeZoneScore,
                                 int raidSulfurToTC,
                                 String stopReason,
                                 String stepRewardBreakdown,
                                 String finalRewardBreakdown,
                                 String trainingStartTimeIso,
                                 String currentTimeIso,
                                 String trainingDeadlineIso,
                                 long trainingElapsedMs,
                                 long trainingRemainingMs,
                                 boolean trainingTimeLimitEnabled,
                                 boolean trainingTimeLimitReached,
                                 Map<String, Integer> invalidActionReasons,
                                 Map<String, Integer> actionTypeCounts,
                                 Map<String, Double> currentRewardConfig,
                                 String historicalReport) {
        this(branchId, totalEpisodesTrained, bestScore, avgEvalScore, currentEpisodeEvalScore,
            currentEpisodeStepReward, currentEpisodeFinalReward, bestTotalReward,
            invalidActionRate, lastEpisodeInvalidActions, lastEpisodeTotalActions,
            epsilon, lastTrainLoss, memorySize, bestBaseBlocks, bestBaseHasTC,
            bestBaseDoors, episodeBlocksPlaced, episodeHasTC, componentCount,
            mainComponentBlocks, evalLogisticsScore, evalCostScore, evalRaidScore,
            evalWorkingAreaScore, evalSafeZoneScore, raidSulfurToTC, stopReason,
            stepRewardBreakdown, finalRewardBreakdown, trainingStartTimeIso,
            currentTimeIso, trainingDeadlineIso, trainingElapsedMs,
            trainingRemainingMs, trainingTimeLimitEnabled, trainingTimeLimitReached,
            invalidActionReasons, actionTypeCounts, currentRewardConfig,
            Collections.emptyMap(), Collections.emptyMap(), historicalReport);
    }

    public SupervisorObservation(String branchId,
                                 int totalEpisodesTrained,
                                 double bestScore,
                                 double avgEvalScore,
                                 double currentEpisodeEvalScore,
                                 double currentEpisodeStepReward,
                                 double currentEpisodeFinalReward,
                                 double bestTotalReward,
                                 double invalidActionRate,
                                 int lastEpisodeInvalidActions,
                                 int lastEpisodeTotalActions,
                                 double epsilon,
                                 double lastTrainLoss,
                                 int memorySize,
                                 int bestBaseBlocks,
                                 boolean bestBaseHasTC,
                                 int bestBaseDoors,
                                 int episodeBlocksPlaced,
                                 boolean episodeHasTC,
                                 int componentCount,
                                 int mainComponentBlocks,
                                 double evalLogisticsScore,
                                 double evalCostScore,
                                 double evalRaidScore,
                                 double evalWorkingAreaScore,
                                 double evalSafeZoneScore,
                                 int raidSulfurToTC,
                                 String stopReason,
                                 String stepRewardBreakdown,
                                 String finalRewardBreakdown,
                                 String trainingStartTimeIso,
                                 String currentTimeIso,
                                 String trainingDeadlineIso,
                                 long trainingElapsedMs,
                                 long trainingRemainingMs,
                                 boolean trainingTimeLimitEnabled,
                                 boolean trainingTimeLimitReached,
                                 Map<String, Integer> invalidActionReasons,
                                 Map<String, Integer> actionTypeCounts,
                                 Map<String, Double> currentRewardConfig,
                                 Map<String, Object> trainingContext,
                                 Map<String, Object> trendMetrics,
                                 String historicalReport) {
        this.branchId = branchId != null ? branchId : "candidate";
        this.totalEpisodesTrained = totalEpisodesTrained;
        this.bestScore = bestScore;
        this.avgEvalScore = avgEvalScore;
        this.currentEpisodeEvalScore = currentEpisodeEvalScore;
        this.currentEpisodeStepReward = currentEpisodeStepReward;
        this.currentEpisodeFinalReward = currentEpisodeFinalReward;
        this.bestTotalReward = bestTotalReward;
        this.invalidActionRate = invalidActionRate;
        this.lastEpisodeInvalidActions = lastEpisodeInvalidActions;
        this.lastEpisodeTotalActions = lastEpisodeTotalActions;
        this.epsilon = epsilon;
        this.lastTrainLoss = lastTrainLoss;
        this.memorySize = memorySize;
        this.bestBaseBlocks = bestBaseBlocks;
        this.bestBaseHasTC = bestBaseHasTC;
        this.bestBaseDoors = bestBaseDoors;
        this.episodeBlocksPlaced = episodeBlocksPlaced;
        this.episodeHasTC = episodeHasTC;
        this.componentCount = componentCount;
        this.mainComponentBlocks = mainComponentBlocks;
        this.evalLogisticsScore = evalLogisticsScore;
        this.evalCostScore = evalCostScore;
        this.evalRaidScore = evalRaidScore;
        this.evalWorkingAreaScore = evalWorkingAreaScore;
        this.evalSafeZoneScore = evalSafeZoneScore;
        this.raidSulfurToTC = raidSulfurToTC;
        this.stopReason = stopReason != null ? stopReason : "UNKNOWN";
        this.stepRewardBreakdown = stepRewardBreakdown != null ? stepRewardBreakdown : "";
        this.finalRewardBreakdown = finalRewardBreakdown != null ? finalRewardBreakdown : "";
        this.trainingStartTimeIso = trainingStartTimeIso != null ? trainingStartTimeIso : "";
        this.currentTimeIso = currentTimeIso != null ? currentTimeIso : "";
        this.trainingDeadlineIso = trainingDeadlineIso != null ? trainingDeadlineIso : "";
        this.trainingElapsedMs = Math.max(0L, trainingElapsedMs);
        this.trainingRemainingMs = trainingRemainingMs;
        this.trainingTimeLimitEnabled = trainingTimeLimitEnabled;
        this.trainingTimeLimitReached = trainingTimeLimitReached;
        this.invalidActionReasons = copyMap(invalidActionReasons);
        this.actionTypeCounts = copyMap(actionTypeCounts);
        this.currentRewardConfig = copyDoubleMap(currentRewardConfig);
        this.trainingContext = copyObjectMap(trainingContext);
        this.trendMetrics = copyObjectMap(trendMetrics);
        this.historicalReport = historicalReport != null ? historicalReport : "";
    }

    private static Map<String, Integer> copyMap(Map<String, Integer> source) {
        if (source == null || source.isEmpty()) {
            return Collections.emptyMap();
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(source));
    }

    private static Map<String, Double> copyDoubleMap(Map<String, Double> source) {
        if (source == null || source.isEmpty()) {
            return Collections.emptyMap();
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(source));
    }

    private static Map<String, Object> copyObjectMap(Map<String, Object> source) {
        if (source == null || source.isEmpty()) {
            return Collections.emptyMap();
        }
        return Collections.unmodifiableMap(new LinkedHashMap<>(source));
    }
}
