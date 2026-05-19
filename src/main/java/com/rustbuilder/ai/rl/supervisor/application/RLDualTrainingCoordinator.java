package com.rustbuilder.ai.rl.supervisor.application;


import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import com.rustbuilder.ai.core.TrainingMetrics;
import com.rustbuilder.ai.rl.application.RLTrainingConfig;
import com.rustbuilder.ai.rl.application.RLTrainingService;
import com.rustbuilder.ai.rl.application.RLTrainingServiceFactory;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;

/**
 * Runs baseline and LLM-candidate training branches in isolated services.
 */
public class RLDualTrainingCoordinator {
    private final RLTrainingServiceFactory trainingServiceFactory;

    public RLDualTrainingCoordinator() {
        this(RLTrainingService::new);
    }

    public RLDualTrainingCoordinator(RLTrainingServiceFactory trainingServiceFactory) {
        this.trainingServiceFactory = Objects.requireNonNull(trainingServiceFactory, "trainingServiceFactory");
    }

    public DualTrainingResult trainInParallel(RLTrainingConfig baselineConfig,
                                              RLTrainingConfig candidateConfig,
                                              Consumer<BranchProgress> progressCallback) throws Exception {
        return trainInParallel(trainingServiceFactory.create(), trainingServiceFactory.create(),
            baselineConfig, candidateConfig, progressCallback);
    }

    public DualTrainingResult trainInParallel(RLTrainingService baselineService,
                                              RLTrainingService candidateService,
                                              RLTrainingConfig baselineConfig,
                                              RLTrainingConfig candidateConfig,
                                              Consumer<BranchProgress> progressCallback) throws Exception {
        if (baselineService == null || candidateService == null) {
            throw new IllegalArgumentException("branch services must not be null");
        }
        if (baselineConfig == null || candidateConfig == null) {
            throw new IllegalArgumentException("branch configs must not be null");
        }
        if (baselineConfig.getModelName().equals(candidateConfig.getModelName())) {
            throw new IllegalArgumentException("baseline and candidate model names must be different");
        }

        RLTrainingConfig safeBaselineConfig = withoutSupervisor(baselineConfig);
        candidateService.setSupervisorConfig(candidateConfig.getSupervisorConfig());

        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<?> baselineFuture = executor.submit(() ->
                baselineService.train(safeBaselineConfig,
                    metrics -> publish(progressCallback, "baseline", metrics),
                    null));

            Future<?> candidateFuture = executor.submit(() ->
                candidateService.train(candidateConfig,
                    metrics -> publish(progressCallback, candidateConfig.getSupervisorConfig().getBranchId(), metrics),
                    null));

            baselineFuture.get();
            candidateFuture.get();
            return new DualTrainingResult(baselineService, candidateService);
        } finally {
            executor.shutdownNow();
        }
    }

    private void publish(Consumer<BranchProgress> progressCallback, String branchId, TrainingMetrics metrics) {
        if (progressCallback != null) {
            progressCallback.accept(new BranchProgress(branchId, metrics));
        }
    }

    RLTrainingConfig withoutSupervisor(RLTrainingConfig config) {
        LlmSupervisorConfig disabled = LlmSupervisorConfig.disabled();
        disabled.setBranchId(config.getSupervisorConfig().getBranchId());
        return new RLTrainingConfig(
            config.getModelName(),
            config.getEpisodesPerEpoch(),
            config.getMaxStepsPerEpisode(),
            config.getLogisticsWeight(),
            config.getCostWeight(),
            config.getRaidWeight(),
            config.getWorkingAreaWeight(),
            config.getSafeZoneWeight(),
            config.getEpochs(),
            disabled,
            config.getTrainingDurationMs(),
            config.isUse2dCnn(),
            config.getOutputDirectory()
        );
    }

    public static final class BranchProgress {
        public final String branchId;
        public final TrainingMetrics metrics;

        public BranchProgress(String branchId, TrainingMetrics metrics) {
            this.branchId = branchId != null ? branchId : "branch";
            this.metrics = metrics;
        }
    }

    public static final class DualTrainingResult {
        public final RLTrainingService baselineService;
        public final RLTrainingService candidateService;

        public DualTrainingResult(RLTrainingService baselineService, RLTrainingService candidateService) {
            this.baselineService = baselineService;
            this.candidateService = candidateService;
        }
    }
}
