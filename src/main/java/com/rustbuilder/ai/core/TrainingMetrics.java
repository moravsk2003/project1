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
    
    public final int memorySize;

    public TrainingMetrics(
            int currentEpoch, int totalEpochs, int currentEpisodeInEpoch, int totalEpisodesPerEpoch,
            int totalEpisodesTrained, double bestScore, double epsilon, double lastTrainLoss,
            double avgReward, double invalidActionRate, int lastEpisodeInvalidActions, int lastEpisodeTotalActions,
            int bestBaseBlocks, boolean bestBaseHasTC, int bestBaseDoors,
            double avgEvalScore, double currentEpisodeEvalScore, double currentEpisodeStepReward, double currentEpisodeFinalReward,
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
        
        this.memorySize = memorySize;
    }
}
