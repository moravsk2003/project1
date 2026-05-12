package com.rustbuilder.ai.rl;

import com.rustbuilder.ai.core.TrainingMetrics;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Map;
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
import com.rustbuilder.ai.rl.supervisor.RLBranchComparator;
import com.rustbuilder.ai.rl.supervisor.RLDualTrainingCoordinator;
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
    private String activeTrainingModelName = "";
    private int activeMaxStepsPerEpisode = 0;
    private int supervisorChecks = 0;
    private int lastSupervisorEpisode = 0;
    private int lastSupervisorBestImprovementEpisode = 0;
    private double lastSupervisorBestScore = Double.NaN;
    private double lastSupervisorAvgEvalScore = Double.NaN;
    private double lastSupervisorAvgReward = Double.NaN;
    private RLTrainingConfig activeTrainingConfig;
    private final LinkedList<Map<String, Object>> supervisorHistory = new LinkedList<>();
    private final Object branchExperimentLock = new Object();
    private volatile boolean branchExperimentRunning = false;
    private volatile String branchExperimentStatus = "No branch experiment yet.";
    private volatile Map<String, Object> latestBranchComparison = Map.of();
    private volatile RLRewardConfig latestBranchCandidateRewardConfig;
    private volatile Double latestBranchCandidateEpsilon;
    private volatile String latestBranchCandidateModelName = "";
    private volatile String pendingBranchSwitchModelName = "";
    private volatile boolean pendingBranchSwitchRequested = false;
    private final LinkedList<Map<String, Object>> branchHistory = new LinkedList<>();
    private final java.util.Set<String> knownBranchModelNames =
        java.util.Collections.synchronizedSet(new java.util.LinkedHashSet<>());

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
        this.activeTrainingModelName = modelName != null ? modelName : "";
        this.activeMaxStepsPerEpisode = maxStepsPerEpisode;
        this.activeTrainingConfig = trainingConfig;
        resetSupervisorTrendState();
        
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
            applyPendingBranchSwitchIfReady();
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

        Map<String, Object> trendMetrics = buildSupervisorTrendMetrics(metrics);
        SupervisorObservation observation = trainingAnalyzer.summarize(
            config.getBranchId(),
            metrics,
            result,
            getRewardConfig(),
            buildSupervisorTrainingContext(metrics, config),
            trendMetrics);
        rememberSupervisorTrend(metrics);
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
            applied = applySupervisorDecision(decision);
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

    private Map<String, Object> buildSupervisorTrainingContext(TrainingMetrics metrics, LlmSupervisorConfig config) {
        Map<String, Object> context = new LinkedHashMap<>();
        context.put("trainingRunning", trainingRunning);
        context.put("modelName", activeTrainingModelName);
        context.put("runId", currentRunId);
        context.put("currentEpoch", metrics.currentEpoch);
        context.put("totalEpochs", metrics.totalEpochs);
        context.put("currentEpisodeInEpoch", metrics.currentEpisodeInEpoch);
        context.put("totalEpisodesPerEpoch", metrics.totalEpisodesPerEpoch);
        context.put("totalEpisodesTarget", metrics.totalEpochs * metrics.totalEpisodesPerEpoch);
        context.put("maxStepsPerEpisode", activeMaxStepsPerEpisode);
        context.put("avgReward", metrics.avgReward);
        context.put("minEpsilon", minEpsilon);
        context.put("epsilonDecay", epsilonDecay);
        context.put("supervisorCallIntervalEpisodes", config != null ? config.getCallIntervalEpisodes() : 0);
        context.put("encoderMode", encoderMode.name());
        context.put("use2dCnn", use2dCnn);
        context.put("useAimSectorLearning", useAimSectorLearning);
        context.put("trainingObjective", buildTrainingObjective());
        return context;
    }

    private Map<String, Object> buildIdleSupervisorTrainingContext(TrainingMetrics metrics, LlmSupervisorConfig config) {
        Map<String, Object> context = buildSupervisorTrainingContext(metrics, config);
        context.put("trainingRunning", false);
        context.put("currentEpoch", 0);
        context.put("totalEpochs", 0);
        context.put("currentEpisodeInEpoch", 0);
        context.put("totalEpisodesPerEpoch", 0);
        context.put("totalEpisodesTarget", 0);
        return context;
    }

    public SupervisorObservation createIdleSupervisorObservation(String branchId) {
        LlmSupervisorConfig config = getSupervisorConfig();
        TrainingMetrics metrics = getMetrics();
        return trainingAnalyzer.summarize(
            branchId,
            metrics,
            null,
            getRewardConfig(),
            buildIdleSupervisorTrainingContext(metrics, config),
            Map.of(
                "supervisorChecks", supervisorChecks,
                "episodesSinceLastSupervisor", lastSupervisorEpisode > 0
                    ? Math.max(0, metrics.totalEpisodesTrained - lastSupervisorEpisode)
                    : 0));
    }

    private Map<String, Object> buildSupervisorTrendMetrics(TrainingMetrics metrics) {
        Map<String, Object> trends = new LinkedHashMap<>();
        trends.put("supervisorChecks", supervisorChecks);
        trends.put("lastSupervisorEpisode", lastSupervisorEpisode);
        trends.put("episodesSinceLastSupervisor", lastSupervisorEpisode > 0
            ? Math.max(0, metrics.totalEpisodesTrained - lastSupervisorEpisode)
            : 0);
        trends.put("bestScoreDeltaSinceLastSupervisor", finiteDelta(metrics.bestScore, lastSupervisorBestScore));
        trends.put("avgEvalScoreDeltaSinceLastSupervisor", finiteDelta(metrics.avgEvalScore, lastSupervisorAvgEvalScore));
        trends.put("avgRewardDeltaSinceLastSupervisor", finiteDelta(metrics.avgReward, lastSupervisorAvgReward));
        trends.put("episodesSinceBestScoreImproved", lastSupervisorBestImprovementEpisode > 0
            ? Math.max(0, metrics.totalEpisodesTrained - lastSupervisorBestImprovementEpisode)
            : 0);
        trends.put("stalledBySupervisorBestScore", supervisorChecks > 0
            && finiteDelta(metrics.bestScore, lastSupervisorBestScore) <= 1e-9);
        trends.put("recentSupervisorHistory", java.util.List.copyOf(supervisorHistory));
        trends.put("branchExperimentRunning", branchExperimentRunning);
        trends.put("branchExperimentStatus", branchExperimentStatus);
        if (!latestBranchComparison.isEmpty()) {
            trends.put("latestBranchComparison", latestBranchComparison);
        }
        trends.put("branchHistory", branchHistorySnapshot());
        trends.put("knownBranchModelNames", availableBranchModelNamesSnapshot());
        if (pendingBranchSwitchRequested) {
            trends.put("pendingBranchSwitchModelName", pendingBranchSwitchModelName);
        }
        return trends;
    }

    private void rememberSupervisorTrend(TrainingMetrics metrics) {
        if (metrics == null) {
            return;
        }
        if (Double.isNaN(lastSupervisorBestScore) || metrics.bestScore > lastSupervisorBestScore + 1e-9) {
            lastSupervisorBestImprovementEpisode = metrics.totalEpisodesTrained;
        }
        lastSupervisorBestScore = metrics.bestScore;
        lastSupervisorAvgEvalScore = metrics.avgEvalScore;
        lastSupervisorAvgReward = metrics.avgReward;
        lastSupervisorEpisode = metrics.totalEpisodesTrained;
        supervisorChecks++;
        addSupervisorHistorySnapshot(metrics);
    }

    private void resetSupervisorTrendState() {
        supervisorChecks = 0;
        lastSupervisorEpisode = 0;
        lastSupervisorBestImprovementEpisode = episodesTrained;
        lastSupervisorBestScore = Double.NaN;
        lastSupervisorAvgEvalScore = Double.NaN;
        lastSupervisorAvgReward = Double.NaN;
        supervisorHistory.clear();
    }

    private Map<String, Object> buildTrainingObjective() {
        Map<String, Object> objective = new LinkedHashMap<>();
        objective.put("primaryGoal", "Train a Rust base builder that creates connected, TC-protected, raid-resistant, usable bases with low invalid action rate.");
        objective.put("priorityOrder", java.util.List.of(
            "valid connected growth",
            "tool cupboard present and enclosed/protected",
            "raid resistance",
            "logistics and working area usability",
            "resource cost efficiency",
            "stable training with low invalid action rate"));
        objective.put("successCriteria", java.util.List.of(
            "candidate bestScore and avgEvalScore improve versus baseline",
            "candidate invalidActionRate does not regress by more than 5 percentage points",
            "candidate best base has at least as many useful blocks as baseline",
            "candidate does not lose TC presence when baseline has TC"));
        objective.put("rewardTuningPolicy", "Do not mutate the live training branch for reward changes. Propose bounded changes; they are tested as candidate branches against a baseline before promotion.");
        return objective;
    }

    private void addSupervisorHistorySnapshot(TrainingMetrics metrics) {
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("episode", metrics.totalEpisodesTrained);
        snapshot.put("bestScore", metrics.bestScore);
        snapshot.put("avgEvalScore", metrics.avgEvalScore);
        snapshot.put("avgReward", metrics.avgReward);
        snapshot.put("invalidActionRate", metrics.invalidActionRate);
        snapshot.put("epsilon", metrics.epsilon);
        snapshot.put("bestBaseBlocks", metrics.bestBaseBlocks);
        snapshot.put("bestBaseHasTC", metrics.bestBaseHasTC);
        snapshot.put("bestTotalReward", metrics.bestTotalReward);
        supervisorHistory.addLast(snapshot);
        while (supervisorHistory.size() > 10) {
            supervisorHistory.removeFirst();
        }
    }

    private double finiteDelta(double current, double previous) {
        if (Double.isNaN(previous) || Double.isInfinite(previous)) {
            return 0.0;
        }
        double delta = current - previous;
        return Double.isNaN(delta) || Double.isInfinite(delta) ? 0.0 : delta;
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
        if (!isTrainingHookApplicable(decision)) {
            return false;
        }
        return config.getApplyMode() == LlmSupervisorApplyMode.AUTO_APPLY;
    }

    private boolean isTrainingHookApplicable(SupervisorDecision decision) {
        if (decision == null || decision.getAction() == null) {
            return false;
        }
        return decision.getAction() == com.rustbuilder.ai.rl.supervisor.SupervisorAction.SET_EPSILON
            || decision.getAction() == com.rustbuilder.ai.rl.supervisor.SupervisorAction.REPLACE_REWARD_CONFIG
            || decision.getAction() == com.rustbuilder.ai.rl.supervisor.SupervisorAction.STOP_TRAINING
            || decision.getAction() == com.rustbuilder.ai.rl.supervisor.SupervisorAction.PROMOTE_BRANCH
            || decision.getAction() == com.rustbuilder.ai.rl.supervisor.SupervisorAction.JUMP_TO_BRANCH;
    }

    private boolean isManuallyApplicable(SupervisorDecision decision) {
        if (decision == null || decision.getAction() == null) {
            return false;
        }
        return isTrainingHookApplicable(decision);
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

    private boolean applySupervisorDecision(SupervisorDecision decision) {
        if (decision == null) {
            return false;
        }

        switch (decision.getAction()) {
            case SET_EPSILON:
                if (decision.getProposedEpsilon() != null) {
                    return startBranchExperiment(decision);
                }
                break;
            case REPLACE_REWARD_CONFIG:
                if (decision.getProposedRewardConfig() != null) {
                    return startBranchExperiment(decision);
                }
                break;
            case STOP_TRAINING:
                requestStop();
                return true;
            case PROMOTE_BRANCH:
                return promoteLatestBranch();
            case JUMP_TO_BRANCH:
                return requestBranchJump(decision.getProposedModelName(), decision.getReason());
            case START_NEW_RUN:
            case RESTART_TRAINING:
            case REQUEST_PROMOTION_CHECK:
            case KEEP_GOING:
            default:
                break;
        }
        return false;
    }

    private boolean startBranchExperiment(SupervisorDecision decision) {
        RLRewardConfig candidateRewardConfig = decision.getProposedRewardConfig();
        if (candidateRewardConfig == null) {
            candidateRewardConfig = getRewardConfig();
        }
        Double candidateEpsilon = decision.getProposedEpsilon();
        if (candidateRewardConfig == null && candidateEpsilon == null) {
            return false;
        }
        if (candidateRewardConfig == null) {
            candidateRewardConfig = RLRewardConfig.createDefault();
        }
        synchronized (branchExperimentLock) {
            if (branchExperimentRunning) {
                branchExperimentStatus = "Branch experiment already running; new proposal ignored.";
                emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
                return false;
            }
            branchExperimentRunning = true;
        }

        RLRewardConfig baselineRewardConfig = getRewardConfig();
        RLTrainingConfig sourceConfig = activeTrainingConfig;
        String baseName = activeTrainingModelName == null || activeTrainingModelName.isBlank()
            ? "autopilot"
            : activeTrainingModelName;
        String stamp = String.valueOf(System.currentTimeMillis());
        String baselineName = safeBranchModelName(baseName + "_baseline_" + stamp);
        String candidateName = safeBranchModelName(baseName + "_candidate_" + stamp);
        latestBranchCandidateRewardConfig = candidateRewardConfig.clone();
        latestBranchCandidateEpsilon = candidateEpsilon;
        latestBranchCandidateModelName = candidateName;
        latestBranchComparison = Map.of();
        branchExperimentStatus = "Starting branch experiment: baseline=" + baselineName + ", candidate=" + candidateName;
        emitSupervisorLog("[BRANCH] " + branchExperimentStatus);

        final RLRewardConfig finalCandidateRewardConfig = candidateRewardConfig;
        final Double finalCandidateEpsilon = candidateEpsilon;
        final double finalBaselineEpsilon = getEpsilon();
        final String proposedAction = decision.getAction().name();
        final String proposalReason = decision.getReason();
        Thread thread = new Thread(() -> runBranchExperiment(
            sourceConfig,
            baselineName,
            candidateName,
            baselineRewardConfig,
            finalCandidateRewardConfig,
            finalBaselineEpsilon,
            finalCandidateEpsilon,
            proposedAction,
            proposalReason));
        thread.setDaemon(true);
        thread.start();
        return true;
    }

    private void runBranchExperiment(RLTrainingConfig sourceConfig,
                                     String baselineName,
                                     String candidateName,
                                     RLRewardConfig baselineRewardConfig,
                                     RLRewardConfig candidateRewardConfig,
                                     double baselineEpsilon,
                                     Double candidateEpsilon,
                                     String proposedAction,
                                     String reason) {
        RLTrainingService baselineService = null;
        RLTrainingService candidateService = null;
        try {
            RLTrainingConfig baselineConfig = branchConfig(sourceConfig, baselineName, "baseline", use2dCnn);
            RLTrainingConfig candidateConfig = branchConfig(sourceConfig, candidateName, candidateName, use2dCnn);
            baselineService = new RLTrainingService();
            candidateService = new RLTrainingService();
            baselineService.setEncoderMode(encoderMode);
            candidateService.setEncoderMode(encoderMode);
            baselineService.setUseAimSectorLearning(useAimSectorLearning);
            candidateService.setUseAimSectorLearning(useAimSectorLearning);
            baselineService.setRewardConfig(baselineRewardConfig);
            candidateService.setRewardConfig(candidateRewardConfig);
            baselineService.setEpsilon(baselineEpsilon);
            if (candidateEpsilon != null) {
                candidateService.setEpsilon(candidateEpsilon);
            }

            RLDualTrainingCoordinator.DualTrainingResult result =
                new RLDualTrainingCoordinator().trainInParallel(
                    baselineService,
                    candidateService,
                    baselineConfig,
                    candidateConfig,
                    progress -> {
                        if (progress.metrics != null) {
                            branchExperimentStatus = String.format(
                                "Branch %s ep=%d best=%.4f avg=%.4f invalid=%.3f",
                                progress.branchId,
                                progress.metrics.totalEpisodesTrained,
                                progress.metrics.bestScore,
                                progress.metrics.avgEvalScore,
                                progress.metrics.invalidActionRate);
                        }
                    });

            TrainingMetrics baselineMetrics = result.baselineService.getMetrics();
            TrainingMetrics candidateMetrics = result.candidateService.getMetrics();
            RLBranchComparator.BranchComparison comparison =
                new RLBranchComparator().compare(baselineMetrics, candidateMetrics);
            String savedBaselineModelName = saveBranchModel(result.baselineService, baselineName, baselineConfig);
            String savedCandidateModelName = saveBranchModel(result.candidateService, candidateName, candidateConfig);
            String savedWinnerModelName = comparison.promoteCandidate ? savedCandidateModelName : savedBaselineModelName;
            latestBranchComparison = branchComparisonMap(
                baselineName,
                candidateName,
                savedBaselineModelName,
                savedCandidateModelName,
                savedWinnerModelName,
                baselineMetrics,
                candidateMetrics,
                comparison,
                proposedAction,
                candidateEpsilon,
                reason);
            rememberBranchExperiment(latestBranchComparison);
            branchExperimentStatus = String.format(
                "Branch experiment finished: candidate=%s savedWinner=%s promoteRecommended=%s bestDelta=%.4f avgDelta=%.4f invalidDelta=%.4f",
                candidateName,
                savedWinnerModelName,
                comparison.promoteCandidate,
                comparison.bestScoreDelta,
                comparison.avgEvalDelta,
                comparison.invalidRateDelta);
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
            requestBranchJump(savedWinnerModelName, "Automatic jump to saved winner branch after experiment.");
        } catch (Exception e) {
            branchExperimentStatus = "Branch experiment failed: " + e.getMessage();
            latestBranchComparison = Map.of("status", "failed", "error", e.getMessage() != null ? e.getMessage() : "");
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
        } finally {
            if (baselineService != null) {
                baselineService.getLlmOrchestrator().stop();
            }
            if (candidateService != null) {
                candidateService.getLlmOrchestrator().stop();
            }
            branchExperimentRunning = false;
        }
    }

    private RLTrainingConfig branchConfig(RLTrainingConfig sourceConfig,
                                          String modelName,
                                          String branchId,
                                          boolean use2dCnnBranch) {
        RLTrainingConfig source = sourceConfig != null
            ? sourceConfig
            : new RLTrainingConfig(modelName, 100, Math.max(1, activeMaxStepsPerEpisode), 1.0, 0.8, 1.2, 1.0, 0.5, 3, LlmSupervisorConfig.disabled());
        LlmSupervisorConfig disabled = LlmSupervisorConfig.disabled();
        disabled.setBranchId(branchId);
        int episodes = Math.max(10, Math.min(100, source.getEpisodesPerEpoch()));
        int epochs = Math.max(1, Math.min(3, source.getEpochs()));
        return new RLTrainingConfig(
            modelName,
            episodes,
            source.getMaxStepsPerEpisode(),
            source.getLogisticsWeight(),
            source.getCostWeight(),
            source.getRaidWeight(),
            source.getWorkingAreaWeight(),
            source.getSafeZoneWeight(),
            epochs,
            disabled,
            0L,
            use2dCnnBranch);
    }

    private Map<String, Object> branchComparisonMap(String baselineName,
                                                    String candidateName,
                                                    String savedBaselineModelName,
                                                    String savedCandidateModelName,
                                                    String savedWinnerModelName,
                                                    TrainingMetrics baseline,
                                                    TrainingMetrics candidate,
                                                    RLBranchComparator.BranchComparison comparison,
                                                    String proposedAction,
                                                    Double candidateEpsilon,
                                                    String reason) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("status", "finished");
        map.put("baselineModelName", baselineName);
        map.put("candidateModelName", candidateName);
        map.put("savedBaselineModelName", savedBaselineModelName);
        map.put("savedCandidateModelName", savedCandidateModelName);
        map.put("savedWinnerModelName", savedWinnerModelName);
        map.put("savedWinnerBranch", comparison.promoteCandidate ? "candidate" : "baseline");
        map.put("proposedAction", proposedAction != null ? proposedAction : "");
        if (candidateEpsilon != null) {
            map.put("candidateEpsilon", candidateEpsilon);
        }
        map.put("proposalReason", reason != null ? reason : "");
        map.put("promoteRecommended", comparison.promoteCandidate);
        map.put("bestScoreDelta", comparison.bestScoreDelta);
        map.put("avgEvalDelta", comparison.avgEvalDelta);
        map.put("invalidRateDelta", comparison.invalidRateDelta);
        map.put("comparisonReason", comparison.reason);
        map.put("baseline", metricsMap(baseline));
        map.put("candidate", metricsMap(candidate));
        map.put("nextAllowedDecision", "Return PROMOTE_BRANCH to apply the candidate reward config, or KEEP_GOING/REPLACE_REWARD_CONFIG to keep exploring.");
        return map;
    }

    private String saveBranchModel(RLTrainingService branchService,
                                   String branchModelName,
                                   RLTrainingConfig branchConfig) throws java.io.IOException {
        RLTrainingConfig config = branchConfig != null
            ? branchConfig
            : branchConfig(activeTrainingConfig, branchModelName, "branch", use2dCnn);
        RLModelManager.RLModel snapshot = RLModelManager.createSnapshot(
            branchModelName,
            branchService,
            config.getLogisticsWeight(),
            config.getCostWeight(),
            config.getRaidWeight(),
            config.getWorkingAreaWeight(),
            config.getSafeZoneWeight());
        RLModelManager.saveModel(snapshot, branchService);
        knownBranchModelNames.add(branchModelName);
        return branchModelName;
    }

    private String saveLiveBranchModel(String branchModelName) throws java.io.IOException {
        RLTrainingConfig config = activeTrainingConfig != null
            ? activeTrainingConfig
            : branchConfig(activeTrainingConfig, branchModelName, "live", use2dCnn);
        RLModelManager.RLModel snapshot = RLModelManager.createSnapshot(
            branchModelName,
            this,
            config.getLogisticsWeight(),
            config.getCostWeight(),
            config.getRaidWeight(),
            config.getWorkingAreaWeight(),
            config.getSafeZoneWeight());
        RLModelManager.saveModel(snapshot, this);
        knownBranchModelNames.add(branchModelName);
        return branchModelName;
    }

    private void rememberBranchExperiment(Map<String, Object> comparison) {
        if (comparison == null || comparison.isEmpty()) {
            return;
        }
        synchronized (branchExperimentLock) {
            branchHistory.addLast(new LinkedHashMap<>(comparison));
            while (branchHistory.size() > 10) {
                branchHistory.removeFirst();
            }
        }
    }

    private java.util.List<Map<String, Object>> branchHistorySnapshot() {
        synchronized (branchExperimentLock) {
            return java.util.List.copyOf(branchHistory);
        }
    }

    private java.util.List<String> knownBranchModelNamesSnapshot() {
        synchronized (knownBranchModelNames) {
            return java.util.List.copyOf(knownBranchModelNames);
        }
    }

    private java.util.List<String> availableBranchModelNamesSnapshot() {
        java.util.LinkedHashSet<String> names = new java.util.LinkedHashSet<>(knownBranchModelNamesSnapshot());
        for (String modelName : RLModelManager.listModels()) {
            if (isBranchLikeModelName(modelName)) {
                names.add(modelName);
            }
        }
        return java.util.List.copyOf(names);
    }

    private boolean isKnownBranchModelName(String modelName) {
        if (knownBranchModelNames.contains(modelName)) {
            return true;
        }
        return isBranchLikeModelName(modelName) && RLModelManager.listModels().contains(modelName);
    }

    private boolean isBranchLikeModelName(String modelName) {
        return modelName != null
            && (modelName.contains("_baseline_")
                || modelName.contains("_candidate_")
                || modelName.contains("_before_jump_"));
    }

    private Map<String, Object> metricsMap(TrainingMetrics metrics) {
        Map<String, Object> map = new LinkedHashMap<>();
        if (metrics == null) {
            return map;
        }
        map.put("episodes", metrics.totalEpisodesTrained);
        map.put("bestScore", metrics.bestScore);
        map.put("avgEvalScore", metrics.avgEvalScore);
        map.put("invalidActionRate", metrics.invalidActionRate);
        map.put("bestBaseBlocks", metrics.bestBaseBlocks);
        map.put("bestBaseHasTC", metrics.bestBaseHasTC);
        map.put("bestTotalReward", metrics.bestTotalReward);
        return map;
    }

    public boolean promoteLatestBranch() {
        RLRewardConfig candidate = latestBranchCandidateRewardConfig;
        if (candidate == null) {
            branchExperimentStatus = "No candidate branch reward config available to promote.";
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
            return false;
        }
        setRewardConfig(candidate);
        if (latestBranchCandidateEpsilon != null) {
            setEpsilon(latestBranchCandidateEpsilon);
        }
        branchExperimentStatus = "Promoted candidate branch reward config: " + latestBranchCandidateModelName;
        emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
        return true;
    }

    public boolean requestBranchJump(String modelName, String reason) {
        String safeModelName = modelName != null ? modelName.trim() : "";
        if (safeModelName.isBlank() || !isKnownBranchModelName(safeModelName)) {
            branchExperimentStatus = "Refused branch jump to unknown model: " + safeModelName;
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
            return false;
        }
        pendingBranchSwitchModelName = safeModelName;
        pendingBranchSwitchRequested = true;
        branchExperimentStatus = "Queued branch jump to " + safeModelName
            + (reason != null && !reason.isBlank() ? " (" + reason + ")" : "");
        emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
        if (trainingRunning) {
            requestStop();
            return true;
        }
        return applyPendingBranchSwitchIfReady();
    }

    private boolean applyPendingBranchSwitchIfReady() {
        if (!pendingBranchSwitchRequested || trainingRunning) {
            return false;
        }
        String modelName = pendingBranchSwitchModelName;
        pendingBranchSwitchRequested = false;
        pendingBranchSwitchModelName = "";
        if (modelName == null || modelName.isBlank()) {
            return false;
        }
        try {
            String previousModel = activeTrainingModelName == null || activeTrainingModelName.isBlank()
                ? "live"
                : activeTrainingModelName;
            LlmSupervisorConfig previousSupervisorConfig = getSupervisorConfig();
            String savedPrevious = saveLiveBranchModel(safeBranchModelName(previousModel + "_before_jump_" + System.currentTimeMillis()));
            RLModelManager.RLModel model = RLModelManager.loadMetadata(modelName);
            RLModelManager.restoreFromModel(this, model);
            RLModelManager.loadNetworkWeights(modelName, this);
            setSupervisorConfig(previousSupervisorConfig);
            activeTrainingModelName = modelName;
            branchExperimentStatus = "Jumped to branch " + modelName + "; previous live branch saved as " + savedPrevious;
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
            return true;
        } catch (Exception e) {
            branchExperimentStatus = "Branch jump failed for " + modelName + ": " + e.getMessage();
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
            return false;
        }
    }

    public String getSupervisorBranchStatus() {
        return branchExperimentStatus != null ? branchExperimentStatus : "";
    }

    private String safeBranchModelName(String value) {
        String safe = value == null ? "branch" : value.replaceAll("[^A-Za-z0-9._-]+", "_");
        if (safe.length() > 90) {
            safe = safe.substring(0, 90);
        }
        return safe.isBlank() ? "branch_" + System.currentTimeMillis() : safe;
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
        if (!applySupervisorDecision(decision)) {
            return false;
        }
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
        resetSupervisorTrendState();

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
        return new TrainingMetrics(0, 0, 0, 0,
            episodesTrained, bestScore, epsilon, lastTrainLoss,
            avgReward, invalidRate, lastEpisodeInvalidActions, lastEpisodeTotalActions,
            bestBaseBlocks, bestBaseHasTC, bestBaseDoors,
            avgEvalScore, 0.0, 0.0, 0.0,
            "", "", getBestTotalRewardForDisplay(), bestBaseStepReward, bestBaseFinalReward, bestBaseTotalReward,
            mSize,
            trainingStartTime,
            System.currentTimeMillis(),
            trainingDeadlineMs,
            getCurrentTrainingElapsedMs(),
            getCurrentTrainingRemainingMs(),
            currentTrainingDurationMs > 0,
            isTrainingTimeLimitReached());
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
