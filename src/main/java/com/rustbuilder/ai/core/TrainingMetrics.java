package com.rustbuilder.ai.core;

/**
 * An immutable snapshot of the RL Agent's training state and performance metrics.
 * Decouples the UI and reporting logic from the internal state of the RLTrainingService.
 */
public class TrainingMetrics {
    public final int currentEpoch;
    public final int totalEpochs;
    public final int currentEpisodeInEpoch;
    public final int totalEpisodesPerEpoch;
    
    public final int totalEpisodesTrained;
    public final double bestScore;
    public final double epsilon;
    public final double lastTrainLoss;
    
    public final double avgReward;
    public final double invalidActionRate; // 0.0 to 1.0
    public final int lastEpisodeInvalidActions;
    public final int lastEpisodeTotalActions;
    
    public final int bestBaseBlocks;
    public final boolean bestBaseHasTC;
    public final int bestBaseDoors;
    
    public final double avgEvalScore;
    public final double currentEpisodeEvalScore;
    public final double currentEpisodeStepReward;
    public final double currentEpisodeFinalReward;
    public final String currentEpisodeStepRewardBreakdown;
    public final String currentEpisodeFinalRewardBreakdown;
    public final double bestTotalReward;
    public final double bestBaseStepReward;
    public final double bestBaseFinalReward;
    public final double bestBaseTotalReward;
    
    public final int memorySize;

    public TrainingMetrics(
            int currentEpoch, int totalEpochs, int currentEpisodeInEpoch, int totalEpisodesPerEpoch,
            int totalEpisodesTrained, double bestScore, double epsilon, double lastTrainLoss,
            double avgReward, double invalidActionRate, int lastEpisodeInvalidActions, int lastEpisodeTotalActions,
            int bestBaseBlocks, boolean bestBaseHasTC, int bestBaseDoors,
            double avgEvalScore, double currentEpisodeEvalScore, double currentEpisodeStepReward, double currentEpisodeFinalReward,
            int memorySize) {
        this(currentEpoch, totalEpochs, currentEpisodeInEpoch, totalEpisodesPerEpoch,
            totalEpisodesTrained, bestScore, epsilon, lastTrainLoss,
            avgReward, invalidActionRate, lastEpisodeInvalidActions, lastEpisodeTotalActions,
            bestBaseBlocks, bestBaseHasTC, bestBaseDoors,
            avgEvalScore, currentEpisodeEvalScore, currentEpisodeStepReward, currentEpisodeFinalReward,
            "", "", 0.0, 0.0, bestScore, bestScore, memorySize);
    }

    public TrainingMetrics(
            int currentEpoch, int totalEpochs, int currentEpisodeInEpoch, int totalEpisodesPerEpoch,
            int totalEpisodesTrained, double bestScore, double epsilon, double lastTrainLoss,
            double avgReward, double invalidActionRate, int lastEpisodeInvalidActions, int lastEpisodeTotalActions,
            int bestBaseBlocks, boolean bestBaseHasTC, int bestBaseDoors,
            double avgEvalScore, double currentEpisodeEvalScore, double currentEpisodeStepReward, double currentEpisodeFinalReward,
            String currentEpisodeFinalRewardBreakdown, int memorySize) {
        this(currentEpoch, totalEpochs, currentEpisodeInEpoch, totalEpisodesPerEpoch,
            totalEpisodesTrained, bestScore, epsilon, lastTrainLoss,
            avgReward, invalidActionRate, lastEpisodeInvalidActions, lastEpisodeTotalActions,
            bestBaseBlocks, bestBaseHasTC, bestBaseDoors,
            avgEvalScore, currentEpisodeEvalScore, currentEpisodeStepReward, currentEpisodeFinalReward,
            "", currentEpisodeFinalRewardBreakdown, 0.0, 0.0, bestScore, bestScore, memorySize);
    }

    public TrainingMetrics(
            int currentEpoch, int totalEpochs, int currentEpisodeInEpoch, int totalEpisodesPerEpoch,
            int totalEpisodesTrained, double bestScore, double epsilon, double lastTrainLoss,
            double avgReward, double invalidActionRate, int lastEpisodeInvalidActions, int lastEpisodeTotalActions,
            int bestBaseBlocks, boolean bestBaseHasTC, int bestBaseDoors,
            double avgEvalScore, double currentEpisodeEvalScore, double currentEpisodeStepReward, double currentEpisodeFinalReward,
            String currentEpisodeStepRewardBreakdown, String currentEpisodeFinalRewardBreakdown,
            double bestTotalReward, double bestBaseStepReward, double bestBaseFinalReward, double bestBaseTotalReward,
            int memorySize) {
        
        this.currentEpoch = currentEpoch;
        this.totalEpochs = totalEpochs;
        this.currentEpisodeInEpoch = currentEpisodeInEpoch;
        this.totalEpisodesPerEpoch = totalEpisodesPerEpoch;
        
        this.totalEpisodesTrained = totalEpisodesTrained;
        this.bestScore = bestScore;
        this.epsilon = epsilon;
        this.lastTrainLoss = lastTrainLoss;
        
        this.avgReward = avgReward;
        this.invalidActionRate = invalidActionRate;
        this.lastEpisodeInvalidActions = lastEpisodeInvalidActions;
        this.lastEpisodeTotalActions = lastEpisodeTotalActions;
        
        this.bestBaseBlocks = bestBaseBlocks;
        this.bestBaseHasTC = bestBaseHasTC;
        this.bestBaseDoors = bestBaseDoors;
        
        this.avgEvalScore = avgEvalScore;
        this.currentEpisodeEvalScore = currentEpisodeEvalScore;
        this.currentEpisodeStepReward = currentEpisodeStepReward;
        this.currentEpisodeFinalReward = currentEpisodeFinalReward;
        this.currentEpisodeStepRewardBreakdown = currentEpisodeStepRewardBreakdown != null ? currentEpisodeStepRewardBreakdown : "";
        this.currentEpisodeFinalRewardBreakdown = currentEpisodeFinalRewardBreakdown != null ? currentEpisodeFinalRewardBreakdown : "";
        this.bestTotalReward = bestTotalReward;
        this.bestBaseStepReward = bestBaseStepReward;
        this.bestBaseFinalReward = bestBaseFinalReward;
        this.bestBaseTotalReward = bestBaseTotalReward;
        
        this.memorySize = memorySize;
    }
}
