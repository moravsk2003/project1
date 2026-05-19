package com.rustbuilder.ai.rl.application;

import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import java.nio.file.Path;

/**
 * Immutable training-run configuration used by RLTrainingService.
 */
public final class RLTrainingConfig {
    private final String modelName;
    private final int episodesPerEpoch;
    private final int maxStepsPerEpisode;
    private final int epochs;
    private final double logisticsWeight;
    private final double costWeight;
    private final double raidWeight;
    private final double workingAreaWeight;
    private final double safeZoneWeight;
    private final LlmSupervisorConfig supervisorConfig;
    private final long trainingDurationMs;
    private final boolean use2dCnn;
    private final Path outputDirectory;

    public RLTrainingConfig(String modelName,
                            int episodesPerEpoch,
                            int maxStepsPerEpisode,
                            double logisticsWeight,
                            double costWeight,
                            double raidWeight,
                            double workingAreaWeight,
                            double safeZoneWeight,
                            int epochs,
                            LlmSupervisorConfig supervisorConfig) {
        this(modelName, episodesPerEpoch, maxStepsPerEpisode, logisticsWeight, costWeight,
            raidWeight, workingAreaWeight, safeZoneWeight, epochs, supervisorConfig, 0L, false);
    }

    public RLTrainingConfig(String modelName,
                            int episodesPerEpoch,
                            int maxStepsPerEpisode,
                            double logisticsWeight,
                            double costWeight,
                            double raidWeight,
                            double workingAreaWeight,
                            double safeZoneWeight,
                            int epochs,
                            LlmSupervisorConfig supervisorConfig,
                            long trainingDurationMs,
                            boolean use2dCnn) {
        this(modelName, episodesPerEpoch, maxStepsPerEpisode, logisticsWeight, costWeight,
            raidWeight, workingAreaWeight, safeZoneWeight, epochs, supervisorConfig,
            trainingDurationMs, use2dCnn, null);
    }

    public RLTrainingConfig(String modelName,
                            int episodesPerEpoch,
                            int maxStepsPerEpisode,
                            double logisticsWeight,
                            double costWeight,
                            double raidWeight,
                            double workingAreaWeight,
                            double safeZoneWeight,
                            int epochs,
                            LlmSupervisorConfig supervisorConfig,
                            long trainingDurationMs,
                            boolean use2dCnn,
                            Path outputDirectory) {
        this.modelName = modelName;
        this.episodesPerEpoch = episodesPerEpoch;
        this.maxStepsPerEpisode = maxStepsPerEpisode;
        this.logisticsWeight = logisticsWeight;
        this.costWeight = costWeight;
        this.raidWeight = raidWeight;
        this.workingAreaWeight = workingAreaWeight;
        this.safeZoneWeight = safeZoneWeight;
        this.epochs = epochs;
        this.supervisorConfig = supervisorConfig != null
            ? supervisorConfig.clone()
            : LlmSupervisorConfig.disabled();
        this.trainingDurationMs = Math.max(0L, trainingDurationMs);
        this.use2dCnn = use2dCnn;
        this.outputDirectory = outputDirectory;
    }

    public String getModelName() {
        return modelName;
    }

    public int getEpisodesPerEpoch() {
        return episodesPerEpoch;
    }

    public int getMaxStepsPerEpisode() {
        return maxStepsPerEpisode;
    }

    public int getEpochs() {
        return epochs;
    }

    public double getLogisticsWeight() {
        return logisticsWeight;
    }

    public double getCostWeight() {
        return costWeight;
    }

    public double getRaidWeight() {
        return raidWeight;
    }

    public double getWorkingAreaWeight() {
        return workingAreaWeight;
    }

    public double getSafeZoneWeight() {
        return safeZoneWeight;
    }

    public LlmSupervisorConfig getSupervisorConfig() {
        return supervisorConfig.clone();
    }

    public long getTrainingDurationMs() {
        return trainingDurationMs;
    }
    
    public boolean isUse2dCnn() {
        return use2dCnn;
    }

    public Path getOutputDirectory() {
        return outputDirectory;
    }
}
