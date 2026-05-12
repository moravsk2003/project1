package com.rustbuilder.ai.rl;

import com.rustbuilder.ai.core.TrainingMetrics;
import java.util.LinkedList;
import java.util.Random;
import java.util.function.Consumer;
import com.rustbuilder.core.action.BuildAction;
import com.rustbuilder.core.placement.PlacementError;
import com.rustbuilder.ai.rl.multidiscrete.*;
import com.rustbuilder.ai.rl.env.state.StateRepresentationEncoder;
import com.rustbuilder.ai.rl.env.state.VoxelV1StateEncoder;
import com.rustbuilder.ai.rl.env.state.BucketedVoxelV2StateEncoder;
import com.rustbuilder.ai.rl.env.state.HybridV3StateEncoder;
import com.rustbuilder.ai.rl.env.spec.EncodingRuntimeConfig;
import com.rustbuilder.ai.rl.supervisor.LlmSupervisor;
import com.rustbuilder.ai.rl.supervisor.LlmSupervisorApplyMode;
import com.rustbuilder.ai.rl.supervisor.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.supervisor.LlmSupervisorRateLimiter;
import com.rustbuilder.ai.rl.supervisor.NoOpLlmSupervisor;
import com.rustbuilder.ai.rl.supervisor.RLTrainingAnalyzer;
import com.rustbuilder.ai.rl.supervisor.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.SupervisorDecisionLogWriter;
import com.rustbuilder.ai.rl.supervisor.SupervisorDecisionValidator;
import com.rustbuilder.ai.rl.supervisor.SupervisorJson;
import com.rustbuilder.ai.rl.supervisor.SupervisorObservation;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.structure.Door;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.service.evaluator.HouseEvaluator;
import com.rustbuilder.service.physics.PlacementService;

/**
 * Main service to train the RL Agent using a Deep Q-Network.
 */
public class RLTrainingService {

    public static class PlacementResult {
        public boolean inserted = false;
        public boolean survived = false;
        public BuildingBlock placedBlock = null;
        public String failReason = null;
        public PlacementError error = PlacementError.NONE;
        public double minDist = -1.0;
        public double socketDist = -1.0;
    }

    private final HouseEvaluator evaluator;
    private final Random random;
    private StateRepresentationEncoder stateEncoder;
    private EncodingRuntimeConfig activeConfig;
    public enum EncoderMode { V1, V2, V3 }

    public enum TrainingLoadProfile {
        LOW("Low (heavy apps)", 0.25, Thread.MIN_PRIORITY),
        MEDIUM("Medium (background)", 0.50, Math.max(Thread.MIN_PRIORITY, Thread.NORM_PRIORITY - 2)),
        HIGH("High (light use)", 0.75, Thread.NORM_PRIORITY),
        MAXIMUM("Maximum (night)", 1.0, Thread.MAX_PRIORITY);

        private final String displayName;
        private final double activeRatio;
        private final int threadPriority;

        TrainingLoadProfile(String displayName, double activeRatio, int threadPriority) {
            this.displayName = displayName;
            this.activeRatio = activeRatio;
            this.threadPriority = threadPriority;
        }

        public String getDisplayName() {
            return displayName;
        }

        public double getActiveRatio() {
            return activeRatio;
        }

        public int getTargetPercent() {
            return (int) Math.round(activeRatio * 100.0);
        }

        private int getThreadPriority() {
            return threadPriority;
        }

        @Override
        public String toString() {
            return displayName;
        }
    }

    private volatile TrainingLoadProfile trainingLoadProfile = TrainingLoadProfile.MAXIMUM;
    private EncoderMode encoderMode = EncoderMode.V3;

    /**
     * Experimental 5-phase multi-discrete action space integration.
     */
    private MultiDiscreteAction currentMultiAction;
    private boolean use2dCnn = false;

    // Hyperparameters
    private double epsilon = 1.0;
    private double epsilonDecay;
    private final double minEpsilon = 0.05;
    private final int targetUpdateFreq = 10;
    private static final int MEMORY_CAPACITY = 10000;

    // Runtime State
    private int episodesTrained = 0;
    private double bestScore = -1;
    private GridModel bestGridModel;
    private double bestTotalReward = -Double.MAX_VALUE;
    private GridModel bestRewardGridModel;
    private GridModel currentGridModel;
    private final Object bestGridLock = new Object();
    private double lastTrainLoss = 0;

    // Experimental 5-phase flow settings
    private MultiDiscretePhasePolicy multiDiscretePolicy;
    private NeuralMultiDiscreteDecisionProvider multiDiscreteNeuralProvider;
    private MultiDiscreteDQNAgent multiDiscreteAgent;
    private MultiDiscreteExperienceReplay multiDiscreteMemory;
    private MultiDiscreteStateObserver multiDiscreteObserver;
    // Extended stats
    private final LinkedList<Double> recentRewards = new LinkedList<>();
    private boolean useAimSectorLearning = false;
    private final LinkedList<Double> recentEvalScores = new LinkedList<>();
    private static final int AVG_WINDOW = 50;
    private double avgReward = 0;
    private double avgEvalScore = 0;
    private int lastEpisodeInvalidActions = 0;
    private int lastEpisodeTotalActions = 0;
    private int bestBaseBlocks = 0;
    private boolean bestBaseHasTC = false;
    private int bestBaseDoors = 0;
    private double bestBaseStepReward = 0;
    private double bestBaseFinalReward = 0;
    private double bestBaseTotalReward = 0;

    private RLRewardConfig rewardConfig;

    private volatile boolean stopRequested = false;
    private volatile boolean trainingRunning = false;
    private long trainingStartTime = 0;
    private long totalTrainingTimeMs = 0;
    private long currentTrainingDurationMs = 0;
    private long trainingDeadlineMs = 0;
    private boolean trainingTimeLimitAnnounced = false;
    private String currentRunId;

    // Training log (file I/O delegated to RLTrainingLogger)
    private final RLTrainingLogger logger = new RLTrainingLogger();
    private LlmSupervisorConfig supervisorConfig = LlmSupervisorConfig.disabled();
    private LlmSupervisor llmSupervisor = new NoOpLlmSupervisor();
    private final RLTrainingAnalyzer trainingAnalyzer = new RLTrainingAnalyzer();
    private final SupervisorDecisionValidator supervisorDecisionValidator = new SupervisorDecisionValidator();
    private final SupervisorDecisionLogWriter supervisorDecisionLogWriter = new SupervisorDecisionLogWriter();
    private final LlmSupervisorRateLimiter supervisorRateLimiter = new LlmSupervisorRateLimiter();
    private volatile String lastSupervisorDecisionSummary = "";
    private volatile SupervisorDecision pendingSupervisorDecision;
    private volatile String pendingSupervisorDecisionSummary = "";
    private volatile java.util.function.Consumer<String> supervisorLogCallback;
    private final com.rustbuilder.ai.rl.supervisor.LlmOrchestrator llmOrchestrator;

    /**
     * Called by LlmOrchestrator to surface its decision/error to the UI log.
     */
    public void setLastSupervisorDecision(SupervisorDecision decision, String summary) {
        this.lastSupervisorDecisionSummary = summary != null ? summary : "";
    }

    public void setSupervisorLogCallback(java.util.function.Consumer<String> supervisorLogCallback) {
        this.supervisorLogCallback = supervisorLogCallback;
    }

    private void emitSupervisorLog(String message) {
        java.util.function.Consumer<String> callback = supervisorLogCallback;
        if (callback != null && message != null && !message.isBlank()) {
            callback.accept(message);
        }
    }

    public RLTrainingService() {
        this.evaluator = new HouseEvaluator();
        this.bestGridModel = new GridModel();
        this.bestRewardGridModel = new GridModel();
        this.currentGridModel = new GridModel();
        this.random = new Random();
        this.currentRunId = "run_" + System.currentTimeMillis();

        this.activeConfig = EncodingRuntimeConfig.createHybridV3Config();
        this.stateEncoder = new HybridV3StateEncoder(this.activeConfig);

        // Neural Multi-Discrete Flow configuration
        this.rewardConfig = RLRewardConfig.createDefault();
        this.multiDiscreteMemory = new MultiDiscreteExperienceReplay(MEMORY_CAPACITY);
        this.multiDiscreteAgent = new MultiDiscreteDQNAgent(this.activeConfig.stateEncodingSpec, this.activeConfig.actionSpaceSpec, this.use2dCnn);
        this.multiDiscreteAgent.setRewardConfig(this.rewardConfig);
        this.multiDiscreteNeuralProvider = new NeuralMultiDiscreteDecisionProvider(this.multiDiscreteAgent, this.stateEncoder);

        // Default policy is Neural
        this.multiDiscretePolicy = new ProvidedPhaseMultiDiscretePolicy(this.multiDiscreteNeuralProvider);
        
        this.llmOrchestrator = new com.rustbuilder.ai.rl.supervisor.LlmOrchestrator(this);
        this.llmOrchestrator.start();
    }

    public com.rustbuilder.ai.rl.supervisor.LlmOrchestrator getLlmOrchestrator() {
        return llmOrchestrator;
    }

    public void setEncoderMode(EncoderMode mode) {
        if (this.encoderMode == mode) return;
        this.encoderMode = mode;
        reinitializeForEncoderSwitch();
    }

    public EncoderMode getEncoderMode() {
        return encoderMode;
    }

    private void reinitializeForEncoderSwitch() {
        switch (encoderMode) {
            case V3:
                this.activeConfig = EncodingRuntimeConfig.createHybridV3Config();
                this.stateEncoder = new HybridV3StateEncoder(this.activeConfig);
                break;
            case V2:
                this.activeConfig = EncodingRuntimeConfig.createVoxelV2Config();
                this.stateEncoder = new BucketedVoxelV2StateEncoder(this.activeConfig);
                break;
            case V1:
            default:
                this.activeConfig = EncodingRuntimeConfig.createVoxelV1Config();
                this.stateEncoder = new VoxelV1StateEncoder(this.activeConfig);
                break;
        }

        if (this.multiDiscreteMemory != null) {
            this.multiDiscreteMemory.clear();
        }

        // Isolate replay memory for the new version
        this.multiDiscreteMemory = new MultiDiscreteExperienceReplay(MEMORY_CAPACITY);

        // Create new agent matching new spec
        this.multiDiscreteAgent = new MultiDiscreteDQNAgent(activeConfig.stateEncodingSpec, activeConfig.actionSpaceSpec, this.use2dCnn);
        this.multiDiscreteAgent.setRewardConfig(this.rewardConfig);
        this.multiDiscreteNeuralProvider = new NeuralMultiDiscreteDecisionProvider(this.multiDiscreteAgent, this.stateEncoder);
        this.multiDiscreteNeuralProvider.setUseAimSectorLearning(this.useAimSectorLearning);
        this.multiDiscretePolicy = new ProvidedPhaseMultiDiscretePolicy(this.multiDiscreteNeuralProvider);

        // Reset statistics for fresh run
        this.episodesTrained = 0;
        this.bestScore = -Double.MAX_VALUE;
        this.bestTotalReward = -Double.MAX_VALUE;
        this.bestGridModel.clear();
        this.bestRewardGridModel.clear();
        this.currentGridModel.clear();
        this.epsilon = 1.0;
        this.epsilonDecay = 1.0;
        this.lastTrainLoss = 0;
        this.recentRewards.clear();
        this.recentEvalScores.clear();
        this.avgReward = 0;
        this.avgEvalScore = 0;
        this.lastEpisodeInvalidActions = 0;
        this.lastEpisodeTotalActions = 0;
        this.bestBaseBlocks = 0;
        this.bestBaseHasTC = false;
        this.bestBaseDoors = 0;
        this.bestBaseStepReward = 0;
        this.bestBaseFinalReward = 0;
        this.bestBaseTotalReward = 0;
        this.logger.init(); // Re-initialize log files for new encoder mode
    }

    /**
     * Train for a number of epochs, each epoch running 'episodes' training episodes.
     * Progress callback receives a TrainingMetrics object containing current status.
     */
    public void train(String modelName, int episodes, int maxStepsPerEpisode, double logW, double costW, double raidW, double workingAreaW, double safeZoneW,
                      int epochs, Consumer<TrainingMetrics> progressCallback, Runnable epochCompleteCallback) {
        train(new RLTrainingConfig(modelName, episodes, maxStepsPerEpisode, logW, costW, raidW,
            workingAreaW, safeZoneW, epochs, supervisorConfig), progressCallback, epochCompleteCallback);
    }

    public void train(RLTrainingConfig trainingConfig,
                      Consumer<TrainingMetrics> progressCallback,
                      Runnable epochCompleteCallback) {
        if (trainingConfig == null) {
            throw new IllegalArgumentException("trainingConfig must not be null");
        }

        String modelName = trainingConfig.getModelName();
        int episodes = trainingConfig.getEpisodesPerEpoch();
        int maxStepsPerEpisode = trainingConfig.getMaxStepsPerEpisode();
        int epochs = trainingConfig.getEpochs();
        this.supervisorConfig = trainingConfig.getSupervisorConfig();
        this.currentTrainingDurationMs = trainingConfig.getTrainingDurationMs();
        
        if (this.use2dCnn != trainingConfig.isUse2dCnn()) {
            this.use2dCnn = trainingConfig.isUse2dCnn();
            reinitializeForEncoderSwitch();
        }

        evaluator.setWeights(
            trainingConfig.getLogisticsWeight(),
            trainingConfig.getCostWeight(),
            trainingConfig.getRaidWeight(),
            trainingConfig.getWorkingAreaWeight(),
            trainingConfig.getSafeZoneWeight());

        int totalEpisodesToTrain = epochs * episodes;
        double exploreEpisodes = totalEpisodesToTrain * 0.8;

        if (exploreEpisodes > 0 && epsilon > (minEpsilon + 0.001)) {
            epsilonDecay = Math.pow(minEpsilon / epsilon, 1.0 / exploreEpisodes);
        } else {
            epsilonDecay = 1.0;
        }

        stopRequested = false;
        trainingRunning = true;
        this.trainingStartTime = System.currentTimeMillis();
        this.trainingDeadlineMs = currentTrainingDurationMs > 0
            ? trainingStartTime + currentTrainingDurationMs
            : 0L;
        this.trainingTimeLimitAnnounced = false;

        logger.setLogFile(modelName, true);
        logger.setRunContext(currentRunId,
            activeConfig.stateEncodingSpec.encoderVersion,
            activeConfig.stateEncodingSpec.voxelChannels,
            activeConfig.stateEncodingSpec.hasGlobalVector,
            activeConfig.stateEncodingSpec.globalFeatureCount);

        String modelRunDir = RLModelManager.getModelDirectory(modelName).toString();
        logger.writeRunMetadata(modelName, activeConfig,
            "default_config",
            "default_training",
            modelRunDir, modelRunDir);

        logger.init();

        try {
            TrainingLoadProfile lastAppliedLoadProfile = null;

            for (int epoch = 0; epoch < epochs; epoch++) {
                if (stopRequested || shouldStopForTrainingTimeLimit()) break;

                double epochTotalReward = 0;
                double epochTotalStepReward = 0;
                double epochTotalFinalReward = 0;
                int epochPositiveTotal = 0;
                int epochPositiveFinal = 0;
                int epochTotalInvalid = 0;
                int epochTotalActions = 0;
                int epochTotalBlocks = 0;
                int epochRejectedProx = 0;
                long epochEncoderTime = 0;
                long epochGlobalTime = 0;
                double epochBestEpScore = -1;
                long epochStartMs = System.currentTimeMillis();

            EpisodeEvaluator episodeEvaluator = new EpisodeEvaluator(evaluator, rewardConfig);
            EpisodeRunner runner = new EpisodeRunner(multiDiscreteAgent, multiDiscreteMemory, multiDiscretePolicy, multiDiscreteObserver, random, rewardConfig, logger, this, stateEncoder, evaluator);

            for (int ep = 0; ep < episodes; ep++) {
                if (stopRequested || shouldStopForTrainingTimeLimit()) break;
                TrainingLoadProfile activeLoadProfile = trainingLoadProfile;
                if (activeLoadProfile != lastAppliedLoadProfile) {
                    applyTrainingThreadPriority(activeLoadProfile);
                    lastAppliedLoadProfile = activeLoadProfile;
                }

                long episodeWorkStartNs = System.nanoTime();
                EpisodeResult result;
                // Set epsilon BEFORE the episode to ensure correct exploration rate
                if (multiDiscreteNeuralProvider != null) {
                    multiDiscreteNeuralProvider.setEpsilon(epsilon);
                }
                result = runner.runExperimentalMultiDiscreteEpisode(maxStepsPerEpisode, episodesTrained, epoch + 1, ep + 1);

                this.currentMultiAction = runner.getCurrentMultiAction();
                if (runner.getLastTrainLoss() > 0) {
                    this.lastTrainLoss = runner.getLastTrainLoss();
                }

                long finalEvalStartNs = System.nanoTime();
                episodeEvaluator.evaluate(result, maxStepsPerEpisode, epoch, ep, episodes, multiDiscreteMemory);
                result.perfFinalEvalNs += System.nanoTime() - finalEvalStartNs;

                // Track epoch stats
                epochTotalReward += (result.accStepReward + result.finalEvalReward);
                epochTotalStepReward += result.accStepReward;
                epochTotalFinalReward += result.finalEvalReward;
                if ((result.accStepReward + result.finalEvalReward) > 0) epochPositiveTotal++;
                if (result.finalEvalReward > 0) epochPositiveFinal++;
                epochTotalInvalid += result.invalidActions;
                epochTotalActions += result.totalActions;
                epochTotalBlocks += result.blocksPlaced;
                epochRejectedProx += HeuristicMaskingUtils.tilesRejectedByNearStructureRule;
                epochEncoderTime += result.totalEncoderTimeMs;
                epochGlobalTime += result.totalGlobalEncoderTimeMs;

                double epTotal = result.accStepReward + result.finalEvalReward;
                if (epTotal > epochBestEpScore) epochBestEpScore = epTotal;

                long episodeLogStartNs = System.nanoTime();
                logger.writeEpisodeLog(epoch + 1, ep + 1, episodesTrained + 1, result, epsilon);
                result.perfEpisodeLogNs += System.nanoTime() - episodeLogStartNs;

                long updateBestStartNs = System.nanoTime();
                updateBestBase(result);
                result.perfUpdateBestNs += System.nanoTime() - updateBestStartNs;

                logger.writePerformanceLog(epoch + 1, ep + 1, episodesTrained + 1, result);

                double episodeEvalScore = 0.0;
                if (result.evaluationResult != null) {
                    episodeEvalScore = result.evaluationResult.finalScore;
                }
                recentEvalScores.addLast(episodeEvalScore);
                if (recentEvalScores.size() > AVG_WINDOW) recentEvalScores.removeFirst();
                avgEvalScore = recentEvalScores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

                if (ep > 0 && ep % targetUpdateFreq == 0) {
                    if (multiDiscreteAgent != null) {
                        multiDiscreteAgent.updateTargetNetwork();
                    }
                }

                if (epsilon > minEpsilon) {
                    epsilon *= epsilonDecay;
                }

                episodesTrained++;

                this.lastEpisodeInvalidActions = result.invalidActions;
                this.lastEpisodeTotalActions = result.totalActions;

                recentRewards.addLast(result.accStepReward);
                if (recentRewards.size() > AVG_WINDOW) recentRewards.removeFirst();
                avgReward = recentRewards.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

                TrainingMetrics metrics = createTrainingMetrics(
                    epoch + 1, epochs, ep + 1, episodes, result, episodeEvalScore);
                maybeInvokeSupervisor(modelName, metrics, result);

                if (progressCallback != null && (ep % 5 == 0 || ep == episodes - 1)) {
                    progressCallback.accept(metrics);
                }

                if (result.finalEvalReward > epochBestEpScore) epochBestEpScore = result.finalEvalReward;
                throttleTrainingLoad(System.nanoTime() - episodeWorkStartNs, activeLoadProfile);
                if (shouldStopForTrainingTimeLimit()) break;
            }

                // End of epoch logging
                long epochMs = System.currentTimeMillis() - epochStartMs;
                logger.writeEpochLog(epoch + 1, episodes, epochTotalReward,
                    epochTotalStepReward, epochTotalFinalReward,
                    epochPositiveTotal, epochPositiveFinal,
                    epochTotalInvalid, epochTotalActions, epochTotalBlocks,
                    epochRejectedProx, epochEncoderTime, epochGlobalTime,
                    epochBestEpScore, epochMs,
                    bestScore, epsilon, lastTrainLoss,
                    bestBaseBlocks, bestBaseHasTC, bestBaseDoors);

            if (epochCompleteCallback != null) {
                epochCompleteCallback.run();
            }
        }
        } finally {
            logger.close();
            totalTrainingTimeMs += System.currentTimeMillis() - trainingStartTime;
            currentTrainingDurationMs = 0L;
            trainingDeadlineMs = 0L;
            trainingRunning = false;
        }
    }

    private TrainingMetrics createTrainingMetrics(int currentEpoch,
                                                  int totalEpochs,
                                                  int currentEpisodeInEpoch,
                                                  int totalEpisodesPerEpoch,
                                                  EpisodeResult result,
                                                  double episodeEvalScore) {
        double invalidRate = result.totalActions > 0
            ? (double) result.invalidActions / result.totalActions
            : 0.0;

        return new TrainingMetrics(
            currentEpoch, totalEpochs, currentEpisodeInEpoch, totalEpisodesPerEpoch,
            episodesTrained, bestScore, epsilon, lastTrainLoss,
            avgReward, invalidRate, result.invalidActions, result.totalActions,
            bestBaseBlocks, bestBaseHasTC, bestBaseDoors,
            avgEvalScore, episodeEvalScore, result.accStepReward, result.finalEvalReward,
            result.stepRewardBreakdownSummary(), result.finalRewardBreakdownSummary(),
            getBestTotalRewardForDisplay(), bestBaseStepReward, bestBaseFinalReward, bestBaseTotalReward,
            multiDiscreteMemory != null ? multiDiscreteMemory.size() : 0,
            trainingStartTime,
            System.currentTimeMillis(),
            trainingDeadlineMs,
            getCurrentTrainingElapsedMs(),
            getCurrentTrainingRemainingMs(),
            currentTrainingDurationMs > 0,
            isTrainingTimeLimitReached()
        );
    }

    private boolean shouldStopForTrainingTimeLimit() {
        if (!isTrainingTimeLimitReached()) {
            return false;
        }
        if (!trainingTimeLimitAnnounced) {
            trainingTimeLimitAnnounced = true;
            lastSupervisorDecisionSummary = "Training time limit reached; stopping after current episode.";
        }
        stopRequested = true;
        return true;
    }

    private boolean isTrainingTimeLimitReached() {
        return trainingDeadlineMs > 0 && System.currentTimeMillis() >= trainingDeadlineMs;
    }

    private long getCurrentTrainingElapsedMs() {
        return trainingStartTime > 0 ? Math.max(0L, System.currentTimeMillis() - trainingStartTime) : 0L;
    }

    private long getCurrentTrainingRemainingMs() {
        if (trainingDeadlineMs <= 0) {
            return -1L;
        }
        return Math.max(0L, trainingDeadlineMs - System.currentTimeMillis());
    }

    private void applyTrainingThreadPriority(TrainingLoadProfile profile) {
        TrainingLoadProfile safeProfile = profile != null ? profile : TrainingLoadProfile.MAXIMUM;
        try {
            Thread.currentThread().setPriority(safeProfile.getThreadPriority());
        } catch (SecurityException ignored) {
            // Some launchers may disallow thread priority changes; throttling still works.
        }
    }

    private void throttleTrainingLoad(long activeNs, TrainingLoadProfile profile) {
        TrainingLoadProfile safeProfile = profile != null ? profile : TrainingLoadProfile.MAXIMUM;
        double activeRatio = safeProfile.getActiveRatio();
        if (activeRatio >= 0.999 || activeNs <= 0L) {
            return;
        }

        long activeMs = Math.max(1L, java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(activeNs));
        long sleepMs = Math.round(activeMs * ((1.0 - activeRatio) / activeRatio));
        while (sleepMs > 0L && !stopRequested && trainingLoadProfile == safeProfile) {
            long chunkMs = Math.min(sleepMs, 250L);
            try {
                Thread.sleep(chunkMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                stopRequested = true;
                return;
            }
            sleepMs -= chunkMs;
        }
    }

    private void maybeInvokeSupervisor(String modelName, TrainingMetrics metrics, EpisodeResult result) {
        LlmSupervisorConfig config = supervisorConfig;
        if (config == null || !config.isEnabled() || metrics == null) {
            return;
        }

        int interval = config.getCallIntervalEpisodes();
        if (metrics.totalEpisodesTrained <= 0 || interval <= 0 || metrics.totalEpisodesTrained % interval != 0) {
            return;
        }

        SupervisorObservation observation = trainingAnalyzer.summarize(config.getBranchId(), metrics, result, getRewardConfig());
        emitSupervisorLog(String.format(
            "Supervisor check: branch=%s episode=%d epsilon=%.5f best=%.4f",
            observation.branchId,
            observation.totalEpisodesTrained,
            observation.epsilon,
            observation.bestScore));
        if (!waitForSupervisorRateLimit(observation)) {
            return;
        }

        SupervisorDecision rawDecision;
        try {
            rawDecision = llmSupervisor.review(observation);
        } catch (Exception e) {
            emitSupervisorLog("Supervisor call failed: " + e.getMessage());
            rawDecision = SupervisorDecision.keepGoing("Supervisor call failed: " + e.getMessage());
        }

        SupervisorDecision decision = supervisorDecisionValidator.validate(rawDecision, config, getRewardConfig());
        boolean applied = shouldApplySupervisorDecision(config, decision);
        if (applied) {
            applySupervisorDecision(decision);
            clearPendingSupervisorDecision();
        } else if (config.getApplyMode() == LlmSupervisorApplyMode.MANUAL_APPROVAL
                && isManuallyApplicable(decision)) {
            pendingSupervisorDecision = decision;
            pendingSupervisorDecisionSummary = formatSupervisorDecisionSummary(observation, decision, false, config);
            emitSupervisorLog("[PENDING] " + pendingSupervisorDecisionSummary);
        } else if (config.getApplyMode() == LlmSupervisorApplyMode.LOG_ONLY) {
            clearPendingSupervisorDecision();
        }
        lastSupervisorDecisionSummary = formatSupervisorDecisionSummary(observation, decision, applied, config);
        emitSupervisorLog(lastSupervisorDecisionSummary);
        supervisorDecisionLogWriter.write(modelName, observation, decision, applied);
    }

    private boolean waitForSupervisorRateLimit(SupervisorObservation observation) {
        int estimatedTokens = estimateSupervisorTokens(observation);
        java.time.Duration delay = supervisorRateLimiter.reserveDelay(estimatedTokens);
        if (!delay.isPositive()) {
            return true;
        }

        lastSupervisorDecisionSummary = String.format(
            "Supervisor rate limit pause: waiting %ds (limits: %d RPM, %d RPD, %d TPM).",
            Math.max(1, delay.toSeconds()),
            supervisorRateLimiter.getRequestsPerMinute(),
            supervisorRateLimiter.getRequestsPerDay(),
            supervisorRateLimiter.getTokensPerMinute());
        emitSupervisorLog(lastSupervisorDecisionSummary);

        long remainingMs = delay.toMillis();
        while (remainingMs > 0 && !stopRequested) {
            long sleepMs = Math.min(remainingMs, 1_000L);
            try {
                Thread.sleep(sleepMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
            remainingMs -= sleepMs;
        }
        if (stopRequested) {
            return false;
        }
        supervisorRateLimiter.reserveNowAfterDelay(estimatedTokens);
        return true;
    }

    private int estimateSupervisorTokens(SupervisorObservation observation) {
        String json = SupervisorJson.observationToJson(observation);
        return Math.max(1, (json.length() / 4) + 2_048);
    }

    private boolean shouldApplySupervisorDecision(LlmSupervisorConfig config, SupervisorDecision decision) {
        if (config == null || decision == null || decision.getAction() == null) {
            return false;
        }
        if (decision.getAction() == com.rustbuilder.ai.rl.supervisor.SupervisorAction.KEEP_GOING
            || decision.getAction() == com.rustbuilder.ai.rl.supervisor.SupervisorAction.REQUEST_PROMOTION_CHECK) {
            return false;
        }
        return config.getApplyMode() == LlmSupervisorApplyMode.AUTO_APPLY;
    }

    private boolean isManuallyApplicable(SupervisorDecision decision) {
        if (decision == null || decision.getAction() == null) {
            return false;
        }
        return decision.getAction() == com.rustbuilder.ai.rl.supervisor.SupervisorAction.SET_EPSILON
            || decision.getAction() == com.rustbuilder.ai.rl.supervisor.SupervisorAction.REPLACE_REWARD_CONFIG
            || decision.getAction() == com.rustbuilder.ai.rl.supervisor.SupervisorAction.STOP_TRAINING;
    }

    private void clearPendingSupervisorDecision() {
        pendingSupervisorDecision = null;
        pendingSupervisorDecisionSummary = "";
    }

    private String formatSupervisorDecisionSummary(SupervisorObservation observation,
                                                   SupervisorDecision decision,
                                                   boolean applied,
                                                   LlmSupervisorConfig config) {
        String reason = decision.getReason();
        if (reason == null || reason.isBlank()) {
            reason = "no reason";
        }
        return String.format("Supervisor ep %d [%s]: %s, applied=%s, reason=%s",
            observation.totalEpisodesTrained,
            config.getApplyMode(),
            decision.getAction(),
            applied ? "yes" : "no",
            reason);
    }

    private void applySupervisorDecision(SupervisorDecision decision) {
        if (decision == null) {
            return;
        }

        switch (decision.getAction()) {
            case SET_EPSILON:
                if (decision.getProposedEpsilon() != null) {
                    setEpsilon(decision.getProposedEpsilon());
                }
                break;
            case REPLACE_REWARD_CONFIG:
                if (decision.getProposedRewardConfig() != null) {
                    setRewardConfig(decision.getProposedRewardConfig());
                }
                break;
            case STOP_TRAINING:
                requestStop();
                break;
            case RESTART_TRAINING:
                requestStop();
                if (decision.getProposedRewardConfig() != null) {
                    setRewardConfig(decision.getProposedRewardConfig());
                }
                if (decision.getProposedUse2dCnn() != null) {
                    setUse2dCnn(decision.getProposedUse2dCnn());
                }
                // Stop current run, orchestrator will pick up and start a new run later or user can click start
                break;
            case START_NEW_RUN:
            case REQUEST_PROMOTION_CHECK:
            case KEEP_GOING:
            default:
                break;
        }
    }



    /**
     * Track best base based on evaluation score.
     */
    private void updateBestBase(EpisodeResult result) {
        synchronized (bestGridLock) {
            double totalReward = result.accStepReward + result.finalEvalReward;
            if (totalReward > bestTotalReward) {
                bestTotalReward = totalReward;
                bestRewardGridModel.clear();
                for (BuildingBlock b : result.grid.getAllBlocks()) {
                    bestRewardGridModel.addBlockSilent(cloneBlock(b));
                }
            }

            if (result.finalEvalReward > bestScore) {
                bestScore = result.finalEvalReward;

                bestGridModel.clear();
                for (BuildingBlock b : result.grid.getAllBlocks()) {
                    bestGridModel.addBlockSilent(cloneBlock(b));
                }

                bestBaseBlocks = result.blocksPlaced;
                bestBaseHasTC = result.hasTC;
                for (BuildingBlock bk : result.grid.getAllBlocks()) {
                    if (bk.getType() == BuildingType.TC) bestBaseHasTC = true;
                }
                bestBaseDoors = (int) result.grid.getAllBlocks().stream().filter(bl -> bl instanceof Door).count();
                bestBaseStepReward = result.accStepReward;
                bestBaseFinalReward = result.finalEvalReward;
                bestBaseTotalReward = totalReward;
            }
        }

        // Snapshot current episode grid for UI previews regardless of score
        currentGridModel.clear();
        for (BuildingBlock b : result.grid.getAllBlocks()) {
            currentGridModel.addBlockSilent(cloneBlock(b));
        }
    }




    private BuildingBlock cloneBlock(BuildingBlock b) {
        return com.rustbuilder.util.BlockFactory.clone(b);
    }

    private static BuildingTier tierFromInt(int t) {
        return com.rustbuilder.util.BlockFactory.tierFromInt(t);
    }

    private static DoorType doorTypeFromInt(int d) {
        return com.rustbuilder.util.BlockFactory.doorTypeFromInt(d);
    }

    private static Orientation orientFromInt(int o) {
        return com.rustbuilder.util.BlockFactory.orientFromInt(o);
    }

    public RLTrainingService.PlacementResult placeBlock(GridModel gridModel, BuildAction action) {
        RLTrainingService.PlacementResult res = new RLTrainingService.PlacementResult();
        PlacementService.Placement placement = PlacementService.calculatePlacement(gridModel, action);

        res.minDist = placement.minDist;
        res.socketDist = placement.socketDist;

        if (!placement.valid) {
            res.error = (placement.error != PlacementError.NONE) ? placement.error : PlacementError.UNKNOWN;
            res.failReason = "invalid_placement(" + res.error + "): " + action.actionType;
            return res;
        }

        double finalRotation = placement.rotation;
        Orientation finalOrientation = placement.orientation;
        int z = (action.actionType == BuildAction.ActionType.FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) ? 0 : action.floor;
        if (z < 0) z = 0;

        BuildingTier tier = com.rustbuilder.util.BlockFactory.tierFromInt(action.tier);
        DoorType doorType = com.rustbuilder.util.BlockFactory.doorTypeFromInt(action.doorType);
        BuildingBlock block = PlacementService.createRealBlock(action, placement, tier, doorType);

        if (block != null) {
            res.placedBlock = block;

            boolean isFurniture = action.actionType == BuildAction.ActionType.TC || action.actionType == BuildAction.ActionType.WORKBENCH || action.actionType == BuildAction.ActionType.LOOT_ROOM;

            if (isFurniture) {
                if (gridModel.canPlace(block)) {
                    // Extra: check no other deployable already at this exact spot
                    boolean occupied = false;
                    for (BuildingBlock b2 : gridModel.getAllBlocks()) {
                        if (com.rustbuilder.util.BuildingTypeUtils.isFurniture(b2.getType()) &&
                            b2.getZ() == block.getZ() &&
                            Math.abs(b2.getX() - block.getX()) < 1.0 &&
                            Math.abs(b2.getY() - block.getY()) < 1.0) {
                            occupied = true;
                            break;
                        }
                    }
                    if (!occupied) {
                        gridModel.addBlockSilent(block);
                        res.inserted = true;
                        res.survived = true;
                        res.error = PlacementError.NONE;
                        return res;
                    } else {
                        res.error = PlacementError.COLLISION;
                        res.failReason = "furniture_slot_occupied";
                    }
                } else {
                    res.error = PlacementError.NO_SUPPORT;
                    res.failReason = "furniture_canPlace_failed";
                }
            } else {
                // Foundations, walls, floors, doors etc: full canPlace (collision + support)
                if (gridModel.canPlace(block)) {
                    gridModel.addBlockSilent(block);
                    if (action.actionType == BuildAction.ActionType.DOORWAY) {
                        Door d = new Door(block.getX(), block.getY(), z, finalOrientation, doorType);
                        d.setTier(tier);
                        d.setRotation(finalRotation);
                        if (gridModel.canPlace(d)) {
                            gridModel.addBlockSilent(d);
                        }
                    }
                    res.inserted = true;
                    res.error = PlacementError.NONE;
                    return res;
                } else {
                    // Decide if it's NO_SUPPORT or COLLISION
                    res.error = isFoundation(block) ? PlacementError.COLLISION : PlacementError.NO_SUPPORT;
                    res.failReason = "structure_canPlace_failed";
                }
            }
        }

        return res;
    }

    public RLRewardConfig getRewardConfig() {
        return (rewardConfig != null) ? rewardConfig.clone() : null;
    }

    public void setRewardConfig(RLRewardConfig rewardConfig) {
        if (rewardConfig == null) {
            this.rewardConfig = RLRewardConfig.createDefault();
        } else if (this.rewardConfig == null) {
            this.rewardConfig = rewardConfig.clone();
        } else {
            this.rewardConfig.copyFrom(rewardConfig);
        }
        if (this.multiDiscreteAgent != null) {
            this.multiDiscreteAgent.setRewardConfig(this.rewardConfig);
        }
    }

    public LlmSupervisorConfig getSupervisorConfig() {
        return supervisorConfig != null ? supervisorConfig.clone() : LlmSupervisorConfig.disabled();
    }

    public void setSupervisorConfig(LlmSupervisorConfig supervisorConfig) {
        this.supervisorConfig = supervisorConfig != null
            ? supervisorConfig.clone()
            : LlmSupervisorConfig.disabled();
    }

    public void setLlmSupervisor(LlmSupervisor llmSupervisor) {
        this.llmSupervisor = llmSupervisor != null ? llmSupervisor : new NoOpLlmSupervisor();
    }

    public String getLastSupervisorDecisionSummary() {
        return lastSupervisorDecisionSummary != null ? lastSupervisorDecisionSummary : "";
    }

    public String getPendingSupervisorDecisionSummary() {
        return pendingSupervisorDecisionSummary != null ? pendingSupervisorDecisionSummary : "";
    }

    public boolean applyPendingSupervisorDecision() {
        SupervisorDecision decision = pendingSupervisorDecision;
        if (!isManuallyApplicable(decision)) {
            return false;
        }
        applySupervisorDecision(decision);
        lastSupervisorDecisionSummary = "Supervisor pending decision manually applied: " + decision.getAction();
        clearPendingSupervisorDecision();
        return true;
    }

    /**
     * Fully resets derived runtime state (stats, best scores, grids, memory)
     * but KEEPS the model weights and core configuration.
     */
    public void resetRuntimeState() {
        this.bestScore = -1;
        this.bestTotalReward = -Double.MAX_VALUE;
        this.bestGridModel.clear();
        this.bestRewardGridModel.clear();
        this.currentGridModel.clear();
        this.lastTrainLoss = 0;
        this.recentRewards.clear();
        this.recentEvalScores.clear();
        this.avgReward = 0;
        this.avgEvalScore = 0;
        this.lastEpisodeInvalidActions = 0;
        this.lastEpisodeTotalActions = 0;
        this.bestBaseBlocks = 0;
        this.bestBaseHasTC = false;
        this.bestBaseDoors = 0;
        this.bestBaseStepReward = 0;
        this.bestBaseFinalReward = 0;
        this.bestBaseTotalReward = 0;
        this.totalTrainingTimeMs = 0;
        this.lastSupervisorDecisionSummary = "";
        clearPendingSupervisorDecision();

        if (this.multiDiscreteMemory != null) {
            this.multiDiscreteMemory.clear();
        }

        // Re-create memory buffers to clear them
        this.multiDiscreteMemory = new MultiDiscreteExperienceReplay(MEMORY_CAPACITY);
    }

    // --- Getters for Summary Stats ---

    public int getEpisodesTrained() { return episodesTrained; }
    public void setEpisodesTrained(int ep) { this.episodesTrained = ep; }

    public double getBestScore() { return bestScore; }
    public void setBestScore(double score) { this.bestScore = score; }

    public double getEpsilon() { return epsilon; }
    public void setEpsilon(double epsilon) { this.epsilon = epsilon; }

    public double getAvgReward() { return avgReward; }
    public double getAvgEvalScore() { return avgEvalScore; }
    public double getLastTrainLoss() { return lastTrainLoss; }

    public double getInvalidActionRate() {
        if (lastEpisodeTotalActions == 0) return 0;
        return (double) lastEpisodeInvalidActions / lastEpisodeTotalActions;
    }

    public int getMemorySize() {
        return multiDiscreteMemory != null ? multiDiscreteMemory.size() : 0;
    }

    public int getBestBaseBlocks() { return bestBaseBlocks; }
    public boolean isBestBaseHasTC() { return bestBaseHasTC; }
    public int getBestBaseDoors() { return bestBaseDoors; }
    private double getBestTotalRewardForDisplay() {
        return bestTotalReward == -Double.MAX_VALUE ? 0.0 : bestTotalReward;
    }

    public String getFormattedTrainingTime() {
        long total = totalTrainingTimeMs;
        if (trainingRunning) {
            total += (System.currentTimeMillis() - trainingStartTime);
        }
        long sec = (total / 1000) % 60;
        long min = (total / (1000 * 60)) % 60;
        long hr = (total / (1000 * 60 * 60));
        return String.format("%02d:%02d:%02d", hr, min, sec);
    }

    public String getRamUsage() {
        Runtime rt = Runtime.getRuntime();
        long used = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024);
        return used + " MB";
    }


    public GridModel getBestGridModel() { return bestGridModel; }
    public void setBestGridModel(GridModel model) {
        synchronized (bestGridLock) {
            this.bestGridModel = model;
        }
    }
    public GridModel getBestGridModelSnapshot() {
        synchronized (bestGridLock) {
            GridModel snapshot = new GridModel();
            for (BuildingBlock b : bestGridModel.getAllBlocks()) {
                BuildingBlock clone = cloneBlock(b);
                if (clone != null) {
                    snapshot.addBlockSilent(clone);
                }
            }
            return snapshot;
        }
    }
    public double getBestTotalReward() { return bestTotalReward; }
    public void setBestRewardGridModel(GridModel model) {
        synchronized (bestGridLock) {
            this.bestRewardGridModel = model;
        }
    }
    public GridModel getBestRewardGridModelSnapshot() {
        synchronized (bestGridLock) {
            GridModel snapshot = new GridModel();
            for (BuildingBlock b : bestRewardGridModel.getAllBlocks()) {
                BuildingBlock clone = cloneBlock(b);
                if (clone != null) {
                    snapshot.addBlockSilent(clone);
                }
            }
            return snapshot;
        }
    }
    public GridModel getCurrentEpisodeGrid() { return currentGridModel; }
    public MultiDiscreteDQNAgent getMultiDiscreteAgent() { return multiDiscreteAgent; }

    public TrainingMetrics getMetrics() {
        double invalidRate = lastEpisodeTotalActions > 0 ? (double) lastEpisodeInvalidActions / lastEpisodeTotalActions : 0.0;
        int mSize = multiDiscreteMemory != null ? multiDiscreteMemory.size() : 0;
        return new TrainingMetrics(0, 0, 0, 0, episodesTrained, bestScore, epsilon, lastTrainLoss, avgReward, invalidRate, lastEpisodeInvalidActions, lastEpisodeTotalActions, bestBaseBlocks, bestBaseHasTC, bestBaseDoors, avgEvalScore, 0.0, 0.0, 0.0, "", "", getBestTotalRewardForDisplay(), bestBaseStepReward, bestBaseFinalReward, bestBaseTotalReward, mSize);
    }

    public void requestStop() {
        this.stopRequested = true;
    }

    public boolean isStopRequested() {
        return stopRequested;
    }

    public boolean isTrainingRunning() {
        return trainingRunning;
    }

    public TrainingLoadProfile getTrainingLoadProfile() {
        return trainingLoadProfile;
    }

    public void setTrainingLoadProfile(TrainingLoadProfile trainingLoadProfile) {
        this.trainingLoadProfile = trainingLoadProfile != null
            ? trainingLoadProfile
            : TrainingLoadProfile.MAXIMUM;
    }

    public com.rustbuilder.ai.rl.env.spec.EncodingRuntimeConfig getRuntimeConfig() {
        return this.activeConfig;
    }
    
    public boolean isUse2dCnn() {
        return use2dCnn;
    }
    
    public void setUse2dCnn(boolean use2dCnn) {
        if (this.use2dCnn != use2dCnn) {
            this.use2dCnn = use2dCnn;
            reinitializeForEncoderSwitch();
        }
    }

    public boolean isUseAimSectorLearning() {
        return useAimSectorLearning;
    }

    public void setUseAimSectorLearning(boolean useAimSectorLearning) {
        this.useAimSectorLearning = useAimSectorLearning;
        if (this.multiDiscreteNeuralProvider != null) {
            this.multiDiscreteNeuralProvider.setUseAimSectorLearning(useAimSectorLearning);
        }
    }

    public MultiDiscretePhasePolicy getMultiDiscretePolicy() { return multiDiscretePolicy; }
    public void setMultiDiscretePolicy(MultiDiscretePhasePolicy policy) { this.multiDiscretePolicy = policy; }

    /**
     * Chooses one greedy action for the current construction grid without advancing training.
     */
    public MultiDiscreteAction chooseSingleStepAction(GridModel grid, MultiDiscreteStateObserver observer) {
        if (grid == null || multiDiscretePolicy == null) {
            return null;
        }

        if (multiDiscreteNeuralProvider != null) {
            multiDiscreteNeuralProvider.setEpsilon(0.0);
        }

        boolean hasTC = false;
        boolean hasLootRoom = false;
        for (BuildingBlock block : grid.getAllBlocks()) {
            if (block.getType() == BuildingType.TC) {
                hasTC = true;
            } else if (block.getType() == BuildingType.LOOT_ROOM) {
                hasLootRoom = true;
            }
        }

        int currentStep = grid.getAllBlocks().isEmpty() ? 0 : Math.max(1, grid.getAllBlocks().size());
        MultiDiscretePhaseContext context = new MultiDiscretePhaseContext(
                grid,
                hasTC,
                hasLootRoom,
                currentStep,
                currentStep + 1
        );

        return multiDiscretePolicy.chooseAction(context, observer);
    }

    /**
     * Explicitly switches the active multi-discrete policy to use the provided decision provider.
     */
    public void setMultiDiscreteDecisionProvider(MultiDiscretePhaseDecisionProvider provider) {
        this.multiDiscretePolicy = new ProvidedPhaseMultiDiscretePolicy(provider);
    }

    private boolean isFoundation(BuildingBlock b) {
        if (b == null) return false;
        return b.getType() == BuildingType.FOUNDATION || b.getType() == BuildingType.TRIANGLE_FOUNDATION;
    }
}
