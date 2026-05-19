package com.rustbuilder.ai.rl.application;


import com.rustbuilder.ai.rl.domain.EpisodeEvaluator;
import com.rustbuilder.ai.rl.domain.EpisodeResult;
import com.rustbuilder.ai.rl.domain.log.StopReason;
import com.rustbuilder.ai.rl.domain.RLRewardConfig;
import com.rustbuilder.ai.rl.environment.spec.StateEncodingSpec;
import com.rustbuilder.ai.rl.infrastructure.DefaultStateEncoderFactory;
import com.rustbuilder.ai.rl.infrastructure.DefaultTrainingAgentFactory;
import com.rustbuilder.ai.rl.infrastructure.RLModelCatalogService;
import com.rustbuilder.ai.rl.infrastructure.RLModelManager;
import com.rustbuilder.ai.rl.policy.multidiscrete.HeuristicMaskingUtils;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscreteAction;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscreteDQNAgent;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscreteExperienceReplay;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscretePhaseContext;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscretePhaseDecisionProvider;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscretePhasePolicy;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscreteStateObserver;
import com.rustbuilder.ai.rl.policy.multidiscrete.NeuralMultiDiscreteDecisionProvider;
import com.rustbuilder.ai.rl.ports.StateEncoderBundle;
import com.rustbuilder.ai.rl.ports.StateEncoderFactory;
import com.rustbuilder.ai.rl.ports.TrainingAgentFactory;
import com.rustbuilder.ai.rl.supervisor.application.LlmOrchestrator;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction;
import com.rustbuilder.ai.rl.supervisor.observability.LlmSupervisorRateLimiter;
import com.rustbuilder.ai.rl.supervisor.observability.SupervisorDecisionLogWriter;
import com.rustbuilder.ai.rl.supervisor.validation.SupervisorDecisionValidator;
import com.rustbuilder.ai.core.TrainingMetrics;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.time.Instant;
import com.rustbuilder.core.action.BuildAction;
import com.rustbuilder.core.placement.PlacementError;
import com.rustbuilder.ai.rl.policy.multidiscrete.*;
import com.rustbuilder.ai.rl.environment.state.StateRepresentationEncoder;
import com.rustbuilder.ai.rl.environment.spec.EncodingRuntimeConfig;
import com.rustbuilder.ai.rl.supervisor.ports.LlmSupervisor;
import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorApplyMode;
import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.supervisor.ports.LlmSupervisorDiagnostics;
import com.rustbuilder.ai.rl.supervisor.provider.NoOpLlmSupervisor;
import com.rustbuilder.ai.rl.supervisor.application.RLBranchComparator;
import com.rustbuilder.ai.rl.supervisor.application.RLDualTrainingCoordinator;
import com.rustbuilder.ai.rl.supervisor.application.RLTrainingAnalyzer;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.serialization.SimpleJson;
import com.rustbuilder.ai.rl.supervisor.serialization.SupervisorJson;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorObservation;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.structure.Door;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.GridModelFactory;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.service.evaluator.HouseEvaluationService;
import com.rustbuilder.service.evaluator.HouseEvaluatorFactory;
import com.rustbuilder.service.physics.PlacementService;
import java.io.IOException;

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

    private final HouseEvaluationService evaluator;
    private final Random random;
    private final GridModelFactory gridModelFactory;
    private final StateEncoderFactory stateEncoderFactory;
    private final TrainingAgentFactory trainingAgentFactory;
    private final EpisodeRunnerFactory episodeRunnerFactory;
    private final RLTrainingServiceFactory branchTrainingServiceFactory;
    private final Supplier<RLDualTrainingCoordinator> dualTrainingCoordinatorFactory;
    private final Supplier<RLBranchComparator> branchComparatorFactory;
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

    private static final class CurriculumEpisodeSample {
        final int episode;
        final double score;
        final double invalidActionRate;
        final double validPlacementRate;
        final int blocksPlaced;
        final boolean hasTC;
        final boolean tcEnclosed;
        final boolean hasLootRoom;
        final int raidSulfurToTC;
        final double connectedMainComponentRate;
        final boolean goodBase;

        CurriculumEpisodeSample(int episode,
                                double score,
                                double invalidActionRate,
                                double validPlacementRate,
                                int blocksPlaced,
                                boolean hasTC,
                                boolean tcEnclosed,
                                boolean hasLootRoom,
                                int raidSulfurToTC,
                                double connectedMainComponentRate,
                                boolean goodBase) {
            this.episode = episode;
            this.score = score;
            this.invalidActionRate = invalidActionRate;
            this.validPlacementRate = validPlacementRate;
            this.blocksPlaced = blocksPlaced;
            this.hasTC = hasTC;
            this.tcEnclosed = tcEnclosed;
            this.hasLootRoom = hasLootRoom;
            this.raidSulfurToTC = raidSulfurToTC;
            this.connectedMainComponentRate = connectedMainComponentRate;
            this.goodBase = goodBase;
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
    private static final int AUTOSAVE_INTERVAL_EPISODES = 100;

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
    private static final int CURRICULUM_STABILITY_WINDOW_LIMIT = 1000;
    private static final int CURRICULUM_MIN_CHANGE_INTERVAL_EPISODES = 100;
    private static final int CURRICULUM_HISTORY_LIMIT = 20;
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
    private int lastAutosaveEpisode = 0;
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
    private final LinkedList<Map<String, Object>> supervisorDecisionMemory = new LinkedList<>();
    private volatile Map<String, Object> currentCurriculumObjective = Map.of();
    private final LinkedList<Map<String, Object>> curriculumObjectiveHistory = new LinkedList<>();
    private final LinkedList<CurriculumEpisodeSample> curriculumSamples = new LinkedList<>();
    private int lastCurriculumObjectiveChangeEpisode = 0;
    private int curriculumObjectiveRevision = 0;
    private volatile boolean pendingAutoResume = false;
    private volatile String lastCurriculumObjectiveStatus = "";
    private final Object branchExperimentLock = new Object();
    private volatile boolean branchExperimentRunning = false;
    private volatile boolean branchReviewInProgress = false;
    private volatile String branchExperimentStatus = "No branch experiment yet.";
    private volatile Map<String, Object> latestBranchComparison = Map.of();
    private volatile RLRewardConfig latestBranchCandidateRewardConfig;
    private volatile Double latestBranchCandidateEpsilon;
    private volatile String latestBranchCandidateModelName = "";
    private volatile String lastPromotedBranchModelName = "";
    private volatile long lastPromotedBranchAtMs = 0L;
    private volatile String pendingBranchSwitchModelName = "";
    private volatile String pendingBranchSwitchOwnerModelName = "";
    private volatile String pendingBranchSwitchReason = "";
    private volatile boolean pendingBranchSwitchPromotion = false;
    private volatile boolean pendingBranchSwitchRequested = false;
    private final LinkedList<Map<String, Object>> branchHistory = new LinkedList<>();
    private final LinkedList<Map<String, Object>> appliedChangeJournal = new LinkedList<>();
    private final java.util.Set<String> promotedBranchModelNames =
        java.util.Collections.synchronizedSet(new java.util.LinkedHashSet<>());
    private final java.util.Set<String> knownBranchModelNames =
        java.util.Collections.synchronizedSet(new java.util.LinkedHashSet<>());
    private volatile int supervisorProviderFailureStreak = 0;
    private volatile long lastSupervisorProviderFailureMs = 0L;
    private volatile String lastSupervisorProviderFailureReason = "";

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
    private final com.rustbuilder.ai.rl.supervisor.application.LlmOrchestrator llmOrchestrator;

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
        this(HouseEvaluatorFactory.createDefault(),
                GridModelFactory.defaultFactory(),
                new Random(),
                new DefaultStateEncoderFactory(),
                new DefaultTrainingAgentFactory());
    }

    public RLTrainingService(HouseEvaluationService evaluator,
                             GridModelFactory gridModelFactory,
                             Random random,
                             StateEncoderFactory stateEncoderFactory,
                             TrainingAgentFactory trainingAgentFactory) {
        this(evaluator,
                gridModelFactory,
                random,
                stateEncoderFactory,
                trainingAgentFactory,
                new EpisodeRunnerFactory(gridModelFactory),
                RLTrainingService::new,
                RLDualTrainingCoordinator::new,
                RLBranchComparator::new);
    }

    public RLTrainingService(HouseEvaluationService evaluator,
                             GridModelFactory gridModelFactory,
                             Random random,
                             StateEncoderFactory stateEncoderFactory,
                             TrainingAgentFactory trainingAgentFactory,
                             EpisodeRunnerFactory episodeRunnerFactory,
                             RLTrainingServiceFactory branchTrainingServiceFactory,
                             Supplier<RLDualTrainingCoordinator> dualTrainingCoordinatorFactory,
                             Supplier<RLBranchComparator> branchComparatorFactory) {
        this.evaluator = Objects.requireNonNull(evaluator, "evaluator");
        this.gridModelFactory = Objects.requireNonNull(gridModelFactory, "gridModelFactory");
        this.random = Objects.requireNonNull(random, "random");
        this.stateEncoderFactory = Objects.requireNonNull(stateEncoderFactory, "stateEncoderFactory");
        this.trainingAgentFactory = Objects.requireNonNull(trainingAgentFactory, "trainingAgentFactory");
        this.episodeRunnerFactory = Objects.requireNonNull(episodeRunnerFactory, "episodeRunnerFactory");
        this.branchTrainingServiceFactory = Objects.requireNonNull(branchTrainingServiceFactory, "branchTrainingServiceFactory");
        this.dualTrainingCoordinatorFactory = Objects.requireNonNull(dualTrainingCoordinatorFactory, "dualTrainingCoordinatorFactory");
        this.branchComparatorFactory = Objects.requireNonNull(branchComparatorFactory, "branchComparatorFactory");
        this.bestGridModel = this.gridModelFactory.create();
        this.bestRewardGridModel = this.gridModelFactory.create();
        this.currentGridModel = this.gridModelFactory.create();
        this.currentRunId = "run_" + System.currentTimeMillis();
        resetCurriculumStateForNewRun("service_start");

        applyEncoderMode(EncoderMode.V3);

        // Neural Multi-Discrete Flow configuration
        this.rewardConfig = RLRewardConfig.createDefault();
        recreateLearningComponents();
        
        this.llmOrchestrator = new com.rustbuilder.ai.rl.supervisor.application.LlmOrchestrator(this);
        this.llmOrchestrator.start();
    }

    public com.rustbuilder.ai.rl.supervisor.application.LlmOrchestrator getLlmOrchestrator() {
        return llmOrchestrator;
    }

    public LlmSupervisorConfig getSupervisorConfig() {
        return supervisorConfig != null ? supervisorConfig.clone() : LlmSupervisorConfig.disabled();
    }

    public void setSupervisorConfig(LlmSupervisorConfig supervisorConfig) {
        this.supervisorConfig = supervisorConfig != null ? supervisorConfig.clone() : LlmSupervisorConfig.disabled();
    }

    public RLRewardConfig getRewardConfig() {
        return rewardConfig != null ? rewardConfig.clone() : RLRewardConfig.createDefault();
    }

    public void setRewardConfig(RLRewardConfig rewardConfig) {
        this.rewardConfig = rewardConfig != null ? rewardConfig.clone() : RLRewardConfig.createDefault();
        if (this.multiDiscreteAgent != null) {
            this.multiDiscreteAgent.setRewardConfig(this.rewardConfig);
        }
    }

    public void setLlmSupervisor(LlmSupervisor llmSupervisor) {
        this.llmSupervisor = llmSupervisor != null ? llmSupervisor : new NoOpLlmSupervisor();
    }

    public String getActiveTrainingModelName() {
        return activeTrainingModelName != null ? activeTrainingModelName : "";
    }

    public RLModelManager.RLModel loadExistingModelForTraining(String modelName)
            throws IOException, ClassNotFoundException {
        if (trainingRunning) {
            throw new IllegalStateException("Cannot load a model while training is running.");
        }
        if (modelName == null || modelName.isBlank()) {
            throw new IllegalArgumentException("modelName must not be blank");
        }
        String safeName = modelName.trim();
        RLModelManager.RLModel model = RLModelManager.loadMetadata(safeName);
        setEncoderMode(encoderModeFromVersion(model.stateEncoderVersion));
        RLModelManager.restoreFromModel(this, model);
        RLModelManager.loadNetworkWeights(safeName, this);
        activeTrainingModelName = safeName;
        return model;
    }

    public void setEncoderMode(EncoderMode mode) {
        if (this.encoderMode == mode) return;
        this.encoderMode = mode;
        reinitializeForEncoderSwitch();
    }

    public EncoderMode getEncoderMode() {
        return encoderMode;
    }

    public static EncoderMode encoderModeFromVersion(String encoderVersion) {
        if ("v3".equalsIgnoreCase(encoderVersion)) {
            return EncoderMode.V3;
        }
        if ("v2".equalsIgnoreCase(encoderVersion)) {
            return EncoderMode.V2;
        }
        return EncoderMode.V1;
    }

    private void applyEncoderMode(EncoderMode mode) {
        StateEncoderBundle bundle = stateEncoderFactory.create(mode);
        this.activeConfig = bundle.getConfig();
        this.stateEncoder = bundle.getEncoder();
    }

    private void recreateLearningComponents() {
        this.multiDiscreteMemory = trainingAgentFactory.createReplay(MEMORY_CAPACITY);
        this.multiDiscreteAgent = trainingAgentFactory.createAgent(this.activeConfig, this.use2dCnn);
        if (this.multiDiscreteAgent != null) {
            this.multiDiscreteAgent.setRewardConfig(this.rewardConfig);
        }
        this.multiDiscreteNeuralProvider = trainingAgentFactory.createNeuralProvider(this.multiDiscreteAgent, this.stateEncoder);
        this.multiDiscreteNeuralProvider.setUseAimSectorLearning(this.useAimSectorLearning);
        this.multiDiscretePolicy = trainingAgentFactory.createPolicy(this.multiDiscreteNeuralProvider);
    }

    private void reinitializeForEncoderSwitch() {
        applyEncoderMode(encoderMode);

        if (this.multiDiscreteMemory != null) {
            this.multiDiscreteMemory.clear();
        }

        recreateLearningComponents();

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
        resetCurriculumStateForNewRun("encoder_switch");
        this.logger.init(); // Re-initialize log files for new encoder mode
    }

    /**
     * Starts an independent training lineage with clean runtime reward state while
     * preserving user-facing supervisor settings and encoder mode.
     */
    public void resetForNewTrainingRun(boolean use2dCnn) {
        if (trainingRunning) {
            throw new IllegalStateException("Cannot reset while training is running.");
        }

        this.use2dCnn = use2dCnn;
        this.rewardConfig = RLRewardConfig.createDefault();
        if (this.supervisorConfig != null) {
            this.supervisorConfig.setCallFrequency(LlmSupervisorConfig.CallFrequency.MEDIUM);
        }
        this.multiDiscreteObserver = null;
        recreateLearningComponents();

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
        this.totalTrainingTimeMs = 0;
        this.currentTrainingDurationMs = 0L;
        this.trainingStartTime = 0L;
        this.trainingDeadlineMs = 0L;
        this.trainingTimeLimitAnnounced = false;
        this.lastAutosaveEpisode = 0;
        this.currentRunId = "run_" + System.currentTimeMillis();
        this.activeTrainingModelName = "";
        this.activeMaxStepsPerEpisode = 0;
        this.activeTrainingConfig = null;
        this.latestBranchComparison = Map.of();
        this.latestBranchCandidateRewardConfig = null;
        this.latestBranchCandidateEpsilon = null;
        this.latestBranchCandidateModelName = "";
        this.lastPromotedBranchModelName = "";
        this.lastPromotedBranchAtMs = 0L;
        synchronized (promotedBranchModelNames) {
            this.promotedBranchModelNames.clear();
        }
        synchronized (branchExperimentLock) {
            this.branchHistory.clear();
        }
        this.pendingBranchSwitchModelName = "";
        this.pendingBranchSwitchOwnerModelName = "";
        this.pendingBranchSwitchReason = "";
        this.pendingBranchSwitchPromotion = false;
        this.pendingBranchSwitchRequested = false;
        this.branchExperimentRunning = false;
        this.branchReviewInProgress = false;
        this.branchExperimentStatus = "No branch experiment yet.";
        this.stopRequested = false;
        this.lastSupervisorDecisionSummary = "";
        synchronized (appliedChangeJournal) {
            this.appliedChangeJournal.clear();
        }
        resetCurriculumStateForNewRun("new_training_run");
        clearPendingSupervisorDecision();
        resetSupervisorTrendState();
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
        this.lastAutosaveEpisode = episodesTrained;

        java.nio.file.Path trainingOutputDir = RLModelManager.normalizeModelOutputDirectory(
            modelName,
            trainingConfig.getOutputDirectory());
        logger.setLogFile(modelName, true, trainingOutputDir);
        logger.setRunContext(currentRunId,
            activeConfig.stateEncodingSpec.encoderVersion,
            activeConfig.stateEncodingSpec.voxelChannels,
            activeConfig.stateEncodingSpec.hasGlobalVector,
            activeConfig.stateEncodingSpec.globalFeatureCount);

        String modelRunDir = trainingOutputDir.toString();
        logger.writeRunMetadata(modelName, activeConfig,
            "default_config",
            "default_training",
            modelRunDir, modelRunDir,
            trainingOutputDir,
            trainingConfig);

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

            EpisodeEvaluator episodeEvaluator = episodeRunnerFactory.createEvaluator(evaluator, rewardConfig);
            EpisodeRunner runner = episodeRunnerFactory.createRunner(
                multiDiscreteAgent,
                multiDiscreteMemory,
                multiDiscretePolicy,
                multiDiscreteObserver,
                random,
                rewardConfig,
                logger,
                this,
                stateEncoder,
                evaluator);

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
                recordCurriculumEpisodeSample(result, episodeEvalScore, epTotal);

                TrainingMetrics metrics = createTrainingMetrics(
                    epoch + 1, epochs, ep + 1, episodes, result, episodeEvalScore);
                maybeInvokeSupervisor(modelName, metrics, result);
                maybeAutosaveModel(modelName, trainingConfig, "interval", false);

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
            maybeAutosaveModel(modelName, trainingConfig, "epoch " + (epoch + 1), true);
        }
        } finally {
            maybeAutosaveModel(modelName, trainingConfig, "final", true);
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

    private void maybeAutosaveModel(String modelName, RLTrainingConfig trainingConfig, String reason, boolean force) {
        if (modelName == null || modelName.isBlank() || trainingConfig == null || multiDiscreteAgent == null) {
            return;
        }
        if (episodesTrained <= 0) {
            return;
        }
        if (!force && episodesTrained - lastAutosaveEpisode < AUTOSAVE_INTERVAL_EPISODES) {
            return;
        }
        if (force && episodesTrained == lastAutosaveEpisode) {
            return;
        }
        try {
            RLModelManager.RLModel snapshot = RLModelManager.createSnapshot(
                modelName,
                this,
                trainingConfig.getLogisticsWeight(),
                trainingConfig.getCostWeight(),
                trainingConfig.getRaidWeight(),
                trainingConfig.getWorkingAreaWeight(),
                trainingConfig.getSafeZoneWeight());
            RLModelManager.saveModel(snapshot, this, trainingConfig.getOutputDirectory());
            lastAutosaveEpisode = episodesTrained;
            lastSupervisorDecisionSummary = String.format(
                "Autosaved model '%s' at episode %d (%s).",
                modelName,
                episodesTrained,
                reason != null ? reason : "training");
            emitSupervisorLog(lastSupervisorDecisionSummary);
        } catch (Exception e) {
            lastSupervisorDecisionSummary = "Autosave failed: " + e.getMessage();
            emitSupervisorLog(lastSupervisorDecisionSummary);
        }
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
        if (branchExperimentRunning || branchReviewInProgress) {
            return;
        }

        int interval = config.getEffectiveCallIntervalEpisodes();
        if (metrics.totalEpisodesTrained <= 0 || interval <= 0 || metrics.totalEpisodesTrained % interval != 0) {
            return;
        }
        if (shouldStopForTrainingTimeLimit()) {
            return;
        }

        Map<String, Object> trendMetrics = buildSupervisorTrendMetrics(metrics);
        trendMetrics.put("currentEpisodeEvaluationDiagnostics", evaluationDiagnostics(result));
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
        if (shouldStopForTrainingTimeLimit()) {
            return;
        }

        SupervisorDecision rawDecision;
        try {
            rawDecision = llmSupervisor.review(observation);
        } catch (Exception e) {
            emitSupervisorLog("Supervisor call failed: " + e.getMessage());
            rawDecision = SupervisorDecision.keepGoing("Supervisor call failed: " + e.getMessage());
        }
        if (shouldStopForTrainingTimeLimit()) {
            return;
        }

        Map<String, Object> diagnostics = supervisorDiagnostics();
        SupervisorDecision decision = supervisorDecisionValidator.validate(rawDecision, config, getRewardConfig(), observation);
        recordSupervisorDecisionOutcome(decision, "training");
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
        supervisorDecisionLogWriter.write(modelName, observation, decision, applied, diagnostics);
        rememberSupervisorDecision(observation, decision, applied, diagnostics);
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
        context.put("supervisorBaseCallIntervalEpisodes", config != null ? config.getCallIntervalEpisodes() : 0);
        context.put("supervisorCallFrequency", config != null ? config.getCallFrequency().name() : "");
        context.put("supervisorCallFrequencyMultiplier", config != null ? config.getCallFrequency().getMultiplier() : 1.0);
        context.put("supervisorCallIntervalEpisodes", config != null ? config.getEffectiveCallIntervalEpisodes() : 0);
        context.put("supervisorCautionLevel", config != null ? config.getCautionLevel().name() : "");
        context.put("supervisorDecisionMode", config != null ? config.getDecisionMode().name() : "");
        context.put("supervisorPrimaryModel", config != null ? config.getPrimaryModel() : "");
        context.put("supervisorFallbackModel", config != null ? config.getFallbackModel() : "");
        context.put("autopilotTrainingDurationMs", config != null ? config.getAutopilotTrainingDurationMs() : 0L);
        context.put("autopilotTrainingDurationSeconds", config != null ? config.getAutopilotTrainingDurationMs() / 1000.0 : 0.0);
        context.put("autopilotTrainingTimeLimitEnabled", config != null && config.getAutopilotTrainingDurationMs() > 0);
        context.put("encoderMode", encoderMode.name());
        context.put("use2dCnn", use2dCnn);
        context.put("useAimSectorLearning", useAimSectorLearning);
        context.put("trainingObjective", buildTrainingObjective());
        context.put("curriculumObjective", currentCurriculumObjectiveSnapshot());
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
            idleSupervisorTrendMetrics(metrics));
    }

    private Map<String, Object> idleSupervisorTrendMetrics(TrainingMetrics metrics) {
        Map<String, Object> trends = new LinkedHashMap<>();
        trends.put("supervisorChecks", supervisorChecks);
        trends.put("episodesSinceLastSupervisor", lastSupervisorEpisode > 0
            ? Math.max(0, metrics.totalEpisodesTrained - lastSupervisorEpisode)
            : 0);
        trends.put("recentSupervisorDecisions", supervisorDecisionMemorySnapshot());
        trends.put("branchHistory", branchHistorySnapshot());
        trends.put("knownBranchModelNames", availableBranchModelNamesSnapshot());
        trends.put("promotedBranchModelNames", promotedBranchModelNamesSnapshot());
        trends.put("lastPromotedBranchModelName", lastPromotedBranchModelName);
        trends.put("lastPromotedBranchAt", lastPromotedBranchAtMs > 0 ? Instant.ofEpochMilli(lastPromotedBranchAtMs).toString() : "");
        trends.put("appliedChangeJournal", appliedChangeJournalSnapshot());
        trends.put("supervisorHealth", supervisorHealthSnapshot());
        trends.put("curriculumStability", curriculumStabilitySummary());
        trends.put("curriculumObjectiveProgress", curriculumObjectiveProgressSnapshot());
        trends.put("curriculumObjectiveHistory", curriculumObjectiveHistorySnapshot());
        trends.put("availableModels", RLModelCatalogService.availableModelSummaries(activeConfig));
        Map<String, Object> comparison = latestBranchComparisonSnapshot();
        if (!comparison.isEmpty()) {
            trends.put("latestBranchComparison", comparison);
            trends.put("latestBranchPromotionOpen", isBranchPromotionOpen(comparison));
        }
        return trends;
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
        trends.put("recentSupervisorDecisions", supervisorDecisionMemorySnapshot());
        trends.put("branchExperimentRunning", branchExperimentRunning);
        trends.put("branchReviewInProgress", branchReviewInProgress);
        trends.put("branchExperimentStatus", branchExperimentStatus);
        if (!latestBranchComparison.isEmpty()) {
            Map<String, Object> comparison = latestBranchComparisonSnapshot();
            trends.put("latestBranchComparison", comparison);
            trends.put("latestBranchPromotionOpen", isBranchPromotionOpen(comparison));
        }
        trends.put("branchHistory", branchHistorySnapshot());
        trends.put("knownBranchModelNames", availableBranchModelNamesSnapshot());
        trends.put("promotedBranchModelNames", promotedBranchModelNamesSnapshot());
        trends.put("lastPromotedBranchModelName", lastPromotedBranchModelName);
        trends.put("lastPromotedBranchAt", lastPromotedBranchAtMs > 0 ? Instant.ofEpochMilli(lastPromotedBranchAtMs).toString() : "");
        trends.put("appliedChangeJournal", appliedChangeJournalSnapshot());
        trends.put("supervisorHealth", supervisorHealthSnapshot());
        trends.put("curriculumStability", curriculumStabilitySummary());
        trends.put("curriculumObjectiveProgress", curriculumObjectiveProgressSnapshot());
        trends.put("curriculumObjectiveHistory", curriculumObjectiveHistorySnapshot());
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
        supervisorDecisionMemory.clear();
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
        objective.put("curriculumPolicy", "Use SET_CURRICULUM_OBJECTIVE to maintain one measurable intermediate objective. Change it only when stability windows show the current skill is reproducible or the objective is clearly wrong.");
        objective.put("evalDiagnosticsPolicy", "Zero eval components are not automatically a pipeline failure. Check trendMetrics.currentEpisodeEvaluationDiagnostics: zero scores are expected for episodes without TC, without protected access, or with open TC exposure.");
        objective.put("stateTruthPolicy", "Use trendMetrics.appliedChangeJournal for changes that actually reached the live runtime. recentSupervisorDecisions can include proposals and branch tests that were not promoted.");
        return objective;
    }

    private void resetCurriculumStateForNewRun(String reason) {
        synchronized (curriculumSamples) {
            curriculumSamples.clear();
        }
        synchronized (curriculumObjectiveHistory) {
            curriculumObjectiveHistory.clear();
        }
        curriculumObjectiveRevision = 0;
        lastCurriculumObjectiveChangeEpisode = episodesTrained;
        lastCurriculumObjectiveStatus = "";
        Map<String, Object> initial = curriculumObjectiveMap(
            "valid_connected_growth",
            "Stabilize valid connected block placement before optimizing TC, loot rooms, or raid resistance.",
            Map.of(
                "invalidActionRateMax", 0.30,
                "medianBlocksMin", 14.0,
                "connectedMainComponentRateMin", 0.80),
            Map.of("invalidActionRateMax", 0.45),
            reason != null ? reason : "new run",
            "system");
        currentCurriculumObjective = initial;
        appendCurriculumHistory(initial);
    }

    private Map<String, Object> curriculumObjectiveMap(String id,
                                                       String description,
                                                       Map<String, Double> successCriteria,
                                                       Map<String, Double> failureSignals,
                                                       String reason,
                                                       String source) {
        Map<String, Object> objective = new LinkedHashMap<>();
        objective.put("objectiveId", id != null ? id : "");
        objective.put("objectiveDescription", description != null ? description : "");
        objective.put("successCriteria", successCriteria != null ? new LinkedHashMap<>(successCriteria) : Map.of());
        objective.put("failureSignals", failureSignals != null ? new LinkedHashMap<>(failureSignals) : Map.of());
        objective.put("startedAtEpisode", episodesTrained);
        objective.put("lastChangedAtEpisode", episodesTrained);
        objective.put("changeReason", reason != null ? reason : "");
        objective.put("source", source != null ? source : "");
        objective.put("revision", ++curriculumObjectiveRevision);
        return objective;
    }

    public boolean applyCurriculumObjective(SupervisorDecision decision) {
        if (decision == null
                || decision.getAction() != com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction.SET_CURRICULUM_OBJECTIVE) {
            return false;
        }
        String objectiveId = decision.getObjectiveId();
        Map<String, Double> successCriteria = decision.getObjectiveSuccessCriteria();
        if (objectiveId == null || objectiveId.isBlank() || successCriteria.isEmpty()) {
            lastCurriculumObjectiveStatus = "Curriculum objective rejected: missing objectiveId or successCriteria.";
            emitSupervisorLog("[CURRICULUM] " + lastCurriculumObjectiveStatus);
            return false;
        }
        Map<String, Object> current = currentCurriculumObjectiveSnapshot();
        if (objectiveId.equals(String.valueOf(current.getOrDefault("objectiveId", "")))
                && successCriteria.equals(criteriaFromObjective(current, "successCriteria"))) {
            lastCurriculumObjectiveStatus = "Curriculum objective already current: " + objectiveId;
            emitSupervisorLog("[CURRICULUM] " + lastCurriculumObjectiveStatus);
            return true;
        }
        int episodesSinceChange = Math.max(0, episodesTrained - lastCurriculumObjectiveChangeEpisode);
        if (curriculumObjectiveRevision > 1 && episodesSinceChange < CURRICULUM_MIN_CHANGE_INTERVAL_EPISODES) {
            lastCurriculumObjectiveStatus = "Curriculum objective change skipped; only " + episodesSinceChange
                + " episodes since last change.";
            emitSupervisorLog("[CURRICULUM] " + lastCurriculumObjectiveStatus);
            return false;
        }

        Map<String, Object> next = curriculumObjectiveMap(
            objectiveId,
            decision.getObjectiveDescription(),
            successCriteria,
            decision.getObjectiveFailureSignals(),
            decision.getReason(),
            "llm");
        currentCurriculumObjective = next;
        lastCurriculumObjectiveChangeEpisode = episodesTrained;
        lastCurriculumObjectiveStatus = "Curriculum objective set to " + objectiveId;
        appendCurriculumHistory(next);
        recordAppliedChange("SET_CURRICULUM_OBJECTIVE", objectiveId, decision.getReason());
        emitSupervisorLog("[CURRICULUM] " + lastCurriculumObjectiveStatus);
        return true;
    }

    private void appendCurriculumHistory(Map<String, Object> objective) {
        synchronized (curriculumObjectiveHistory) {
            curriculumObjectiveHistory.addLast(new LinkedHashMap<>(objective));
            while (curriculumObjectiveHistory.size() > CURRICULUM_HISTORY_LIMIT) {
                curriculumObjectiveHistory.removeFirst();
            }
        }
    }

    public Map<String, Object> getCurriculumObjectiveSnapshot() {
        return currentCurriculumObjectiveSnapshot();
    }

    private Map<String, Object> currentCurriculumObjectiveSnapshot() {
        Map<String, Object> objective = currentCurriculumObjective;
        return objective == null || objective.isEmpty()
            ? Map.of()
            : deepCopyObjectMap(objective);
    }

    private java.util.List<Map<String, Object>> curriculumObjectiveHistorySnapshot() {
        synchronized (curriculumObjectiveHistory) {
            java.util.List<Map<String, Object>> copy = new java.util.ArrayList<>(curriculumObjectiveHistory.size());
            for (Map<String, Object> entry : curriculumObjectiveHistory) {
                copy.add(deepCopyObjectMap(entry));
            }
            return copy;
        }
    }

    private void recordCurriculumEpisodeSample(EpisodeResult result, double episodeEvalScore, double totalReward) {
        if (result == null) {
            return;
        }
        double invalidRate = result.totalActions > 0
            ? clamp01((double) result.invalidActions / result.totalActions)
            : 0.0;
        double validPlacementRate = result.totalActions > 0
            ? clamp01(1.0 - invalidRate)
            : (result.blocksPlaced > 0 ? 1.0 : 0.0);
        int blocks = result.grid != null ? result.grid.getAllBlocks().size() : result.blocksPlaced;
        boolean hasTC = result.hasTC || result.finalRewardHasTC;
        boolean tcEnclosed = result.finalRewardTcEnclosed;
        double connectedRate = blocks > 0
            ? clamp01((double) Math.max(0, result.mainComponentBlocks) / blocks)
            : 0.0;
        double score = episodeEvalScore != 0.0 ? episodeEvalScore : totalReward;
        boolean goodBase = hasTC && blocks >= 12 && (tcEnclosed || result.raidSulfurToTC > 0 || episodeEvalScore > 0.0);
        CurriculumEpisodeSample sample = new CurriculumEpisodeSample(
            episodesTrained,
            score,
            invalidRate,
            validPlacementRate,
            blocks,
            hasTC,
            tcEnclosed,
            result.hasLootRoom,
            result.raidSulfurToTC,
            connectedRate,
            goodBase);
        synchronized (curriculumSamples) {
            curriculumSamples.addLast(sample);
            while (curriculumSamples.size() > CURRICULUM_STABILITY_WINDOW_LIMIT) {
                curriculumSamples.removeFirst();
            }
        }
    }

    public Map<String, Object> getCurriculumStabilitySummary() {
        return curriculumStabilitySummary();
    }

    private Map<String, Object> curriculumStabilitySummary() {
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("last100", curriculumWindowSummary(100));
        summary.put("last300", curriculumWindowSummary(300));
        summary.put("last1000", curriculumWindowSummary(1000));
        return summary;
    }

    private Map<String, Object> curriculumWindowSummary(int window) {
        java.util.List<CurriculumEpisodeSample> samples = curriculumSampleWindow(window);
        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("window", window);
        summary.put("sampleCount", samples.size());
        if (samples.isEmpty()) {
            return summary;
        }
        java.util.List<Double> scores = new java.util.ArrayList<>();
        java.util.List<Double> blocks = new java.util.ArrayList<>();
        java.util.List<Double> raids = new java.util.ArrayList<>();
        double invalidSum = 0.0;
        double validSum = 0.0;
        double connectedSum = 0.0;
        int tcCount = 0;
        int enclosedCount = 0;
        int lootCount = 0;
        int goodCount = 0;
        for (CurriculumEpisodeSample sample : samples) {
            scores.add(sample.score);
            blocks.add((double) sample.blocksPlaced);
            if (sample.raidSulfurToTC >= 0) {
                raids.add((double) sample.raidSulfurToTC);
            }
            invalidSum += sample.invalidActionRate;
            validSum += sample.validPlacementRate;
            connectedSum += sample.connectedMainComponentRate;
            if (sample.hasTC) tcCount++;
            if (sample.tcEnclosed) enclosedCount++;
            if (sample.hasLootRoom) lootCount++;
            if (sample.goodBase) goodCount++;
        }
        int count = samples.size();
        summary.put("medianScore", percentile(scores, 0.50));
        summary.put("p25Score", percentile(scores, 0.25));
        summary.put("medianBlocks", percentile(blocks, 0.50));
        summary.put("p25Blocks", percentile(blocks, 0.25));
        summary.put("invalidActionRateAvg", invalidSum / count);
        summary.put("validPlacementRateAvg", validSum / count);
        summary.put("connectedMainComponentRateAvg", connectedSum / count);
        summary.put("tcPresentRate", (double) tcCount / count);
        summary.put("tcEnclosedRate", (double) enclosedCount / count);
        summary.put("lootRoomPresentRate", (double) lootCount / count);
        summary.put("goodBaseRate", (double) goodCount / count);
        summary.put("repeatedGoodBaseCount", goodCount);
        summary.put("medianRaidSulfurToTc", raids.isEmpty() ? 0.0 : percentile(raids, 0.50));
        summary.put("p25RaidSulfurToTc", raids.isEmpty() ? 0.0 : percentile(raids, 0.25));
        summary.put("firstEpisode", samples.get(0).episode);
        summary.put("lastEpisode", samples.get(samples.size() - 1).episode);
        return summary;
    }

    private Map<String, Object> curriculumObjectiveProgressSnapshot() {
        Map<String, Object> objective = currentCurriculumObjectiveSnapshot();
        Map<String, Double> criteria = criteriaFromObjective(objective, "successCriteria");
        Map<String, Object> progress = new LinkedHashMap<>();
        progress.put("objectiveId", objective.getOrDefault("objectiveId", ""));
        progress.put("episodesSinceObjectiveChange", Math.max(0, episodesTrained - lastCurriculumObjectiveChangeEpisode));
        progress.put("lastStatus", lastCurriculumObjectiveStatus);
        progress.put("successCriteria", criteria);
        Map<String, Object> last300 = curriculumWindowSummary(300);
        Map<String, Object> criteriaStatus = new LinkedHashMap<>();
        for (Map.Entry<String, Double> criterion : criteria.entrySet()) {
            Double currentValue = criterionValueFromSummary(criterion.getKey(), last300);
            Map<String, Object> status = new LinkedHashMap<>();
            status.put("target", criterion.getValue());
            status.put("current", currentValue != null ? currentValue : 0.0);
            status.put("passed", currentValue != null && criterionPassed(criterion.getKey(), currentValue, criterion.getValue()));
            criteriaStatus.put(criterion.getKey(), status);
        }
        progress.put("criteriaStatusLast300", criteriaStatus);
        progress.put("objectivePassRateLast100", objectivePassRate(100, criteria));
        progress.put("objectivePassRateLast300", objectivePassRate(300, criteria));
        progress.put("objectivePassRateLast1000", objectivePassRate(1000, criteria));
        return progress;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Double> criteriaFromObjective(Map<String, Object> objective, String key) {
        if (objective == null || !(objective.get(key) instanceof Map<?, ?> raw)) {
            return Map.of();
        }
        Map<String, Double> criteria = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : raw.entrySet()) {
            Object value = entry.getValue();
            if (entry.getKey() instanceof String name && value instanceof Number number) {
                criteria.put(name, number.doubleValue());
            }
        }
        return criteria;
    }

    private double objectivePassRate(int window, Map<String, Double> criteria) {
        java.util.List<CurriculumEpisodeSample> samples = curriculumSampleWindow(window);
        if (samples.isEmpty() || criteria == null || criteria.isEmpty()) {
            return 0.0;
        }
        int passed = 0;
        for (CurriculumEpisodeSample sample : samples) {
            if (objectiveSamplePasses(sample, criteria)) {
                passed++;
            }
        }
        return (double) passed / samples.size();
    }

    private boolean objectiveSamplePasses(CurriculumEpisodeSample sample, Map<String, Double> criteria) {
        for (Map.Entry<String, Double> criterion : criteria.entrySet()) {
            Double value = criterionValueFromSample(criterion.getKey(), sample);
            if (value == null || !criterionPassed(criterion.getKey(), value, criterion.getValue())) {
                return false;
            }
        }
        return true;
    }

    private boolean criterionPassed(String key, double value, double target) {
        return key != null && key.endsWith("Max")
            ? value <= target
            : value >= target;
    }

    private Double criterionValueFromSummary(String key, Map<String, Object> summary) {
        if (summary == null || key == null) {
            return null;
        }
        Object value = switch (key) {
            case "medianScoreMin" -> summary.get("medianScore");
            case "p25ScoreMin" -> summary.get("p25Score");
            case "goodBaseRateMin" -> summary.get("goodBaseRate");
            case "validPlacementRateMin" -> summary.get("validPlacementRateAvg");
            case "invalidActionRateMax" -> summary.get("invalidActionRateAvg");
            case "medianBlocksMin" -> summary.get("medianBlocks");
            case "p25BlocksMin" -> summary.get("p25Blocks");
            case "tcPresentRateMin" -> summary.get("tcPresentRate");
            case "tcEnclosedRateMin" -> summary.get("tcEnclosedRate");
            case "lootRoomPresentRateMin" -> summary.get("lootRoomPresentRate");
            case "connectedMainComponentRateMin" -> summary.get("connectedMainComponentRateAvg");
            case "medianRaidSulfurToTcMin" -> summary.get("medianRaidSulfurToTc");
            case "p25RaidSulfurToTcMin" -> summary.get("p25RaidSulfurToTc");
            default -> null;
        };
        return value instanceof Number number ? number.doubleValue() : null;
    }

    private Double criterionValueFromSample(String key, CurriculumEpisodeSample sample) {
        if (sample == null || key == null) {
            return null;
        }
        return switch (key) {
            case "medianScoreMin", "p25ScoreMin" -> sample.score;
            case "goodBaseRateMin" -> sample.goodBase ? 1.0 : 0.0;
            case "validPlacementRateMin" -> sample.validPlacementRate;
            case "invalidActionRateMax" -> sample.invalidActionRate;
            case "medianBlocksMin", "p25BlocksMin" -> (double) sample.blocksPlaced;
            case "tcPresentRateMin" -> sample.hasTC ? 1.0 : 0.0;
            case "tcEnclosedRateMin" -> sample.tcEnclosed ? 1.0 : 0.0;
            case "lootRoomPresentRateMin" -> sample.hasLootRoom ? 1.0 : 0.0;
            case "connectedMainComponentRateMin" -> sample.connectedMainComponentRate;
            case "medianRaidSulfurToTcMin", "p25RaidSulfurToTcMin" -> (double) Math.max(0, sample.raidSulfurToTC);
            default -> null;
        };
    }

    private java.util.List<CurriculumEpisodeSample> curriculumSampleWindow(int window) {
        synchronized (curriculumSamples) {
            int size = curriculumSamples.size();
            int from = Math.max(0, size - Math.max(1, window));
            return new java.util.ArrayList<>(curriculumSamples.subList(from, size));
        }
    }

    private static double percentile(java.util.List<Double> values, double p) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        java.util.List<Double> sorted = new java.util.ArrayList<>(values);
        sorted.sort(Double::compareTo);
        int index = (int) Math.floor(clamp01(p) * (sorted.size() - 1));
        return sorted.get(Math.max(0, Math.min(sorted.size() - 1, index)));
    }

    private static double clamp01(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return 0.0;
        }
        return Math.max(0.0, Math.min(1.0, value));
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> deepCopyObjectMap(Map<String, Object> source) {
        Map<String, Object> copy = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : source.entrySet()) {
            Object value = entry.getValue();
            if (value instanceof Map<?, ?> map) {
                Map<String, Object> nested = new LinkedHashMap<>();
                for (Map.Entry<?, ?> nestedEntry : map.entrySet()) {
                    if (nestedEntry.getKey() != null) {
                        nested.put(String.valueOf(nestedEntry.getKey()), nestedEntry.getValue());
                    }
                }
                copy.put(entry.getKey(), nested);
            } else {
                copy.put(entry.getKey(), value);
            }
        }
        return copy;
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

    private Map<String, Object> supervisorDiagnostics() {
        LlmSupervisor supervisor = this.llmSupervisor;
        if (supervisor instanceof LlmSupervisorDiagnostics diagnostics) {
            Map<String, Object> map = diagnostics.getLastDiagnostics();
            return map != null ? map : Map.of();
        }
        return Map.of();
    }

    private void rememberSupervisorDecision(SupervisorObservation observation,
                                            SupervisorDecision decision,
                                            boolean applied,
                                            Map<String, Object> diagnostics) {
        if (observation == null || decision == null) {
            return;
        }
        synchronized (supervisorDecisionMemory) {
            if (!supervisorDecisionMemory.isEmpty()) {
                Map<String, Object> previous = supervisorDecisionMemory.getLast();
                previous.put("effectAfterDecision", effectSince(previous, observation));
            }

            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("episode", observation.totalEpisodesTrained);
            entry.put("action", decision.getAction().name());
            entry.put("applied", applied);
            entry.put("reason", decision.getReason());
            entry.put("callFrequency", decision.getProposedCallFrequency() != null
                ? decision.getProposedCallFrequency().name()
                : "");
            if (decision.getProposedEpsilon() != null) {
                entry.put("proposedEpsilon", decision.getProposedEpsilon());
            }
            RLRewardConfig proposedRewardConfig = decision.getProposedRewardConfig();
            entry.put("hasRewardConfig", proposedRewardConfig != null);
            if (proposedRewardConfig != null) {
                entry.put("proposedRewardConfig", proposedRewardConfig.toMap());
            }
            if (decision.getAction() == com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction.SET_CURRICULUM_OBJECTIVE) {
                entry.put("objectiveId", decision.getObjectiveId());
                entry.put("objectiveDescription", decision.getObjectiveDescription());
                entry.put("successCriteria", decision.getObjectiveSuccessCriteria());
                entry.put("failureSignals", decision.getObjectiveFailureSignals());
            }
            entry.put("executionStatus", supervisorDecisionExecutionStatus(decision, applied));
            if (decision.getConfidence() != null) {
                entry.put("confidence", decision.getConfidence());
            }
            if (!decision.getRiskLevel().isBlank()) {
                entry.put("riskLevel", decision.getRiskLevel());
            }
            if (!decision.getChangeMagnitude().isBlank()) {
                entry.put("changeMagnitude", decision.getChangeMagnitude());
            }
            if (!decision.getExpectedEffect().isBlank()) {
                entry.put("expectedEffect", decision.getExpectedEffect());
            }
            if (decision.getRequiresBranchTest() != null) {
                entry.put("requiresBranchTest", decision.getRequiresBranchTest());
            }
            Object selectedModel = selectedModelFromDiagnostics(diagnostics);
            if (selectedModel != null) {
                entry.put("selectedModel", selectedModel);
            }
            entry.put("observationEpisode", observation.totalEpisodesTrained);
            entry.put("observationBestScore", observation.bestScore);
            entry.put("observationAvgEvalScore", observation.avgEvalScore);
            entry.put("observationInvalidActionRate", observation.invalidActionRate);
            entry.put("observationEpsilon", observation.epsilon);
            entry.put("observationBestBaseBlocks", observation.bestBaseBlocks);
            entry.put("observationBestBaseHasTC", observation.bestBaseHasTC);

            supervisorDecisionMemory.addLast(entry);
            while (supervisorDecisionMemory.size() > 10) {
                supervisorDecisionMemory.removeFirst();
            }
        }
    }

    private Object selectedModelFromDiagnostics(Map<String, Object> diagnostics) {
        if (diagnostics == null || diagnostics.isEmpty()) {
            return null;
        }
        Object stagesObject = diagnostics.get("stages");
        if (!(stagesObject instanceof java.util.List<?> stages) || stages.isEmpty()) {
            return null;
        }
        Object last = stages.get(stages.size() - 1);
        if (last instanceof Map<?, ?> stage) {
            Object selected = stage.get("selectedModel");
            return selected != null ? selected : null;
        }
        return null;
    }

    private String supervisorDecisionExecutionStatus(SupervisorDecision decision, boolean applied) {
        if (decision == null) {
            return "none";
        }
        return switch (decision.getAction()) {
            case SET_EPSILON, REPLACE_REWARD_CONFIG -> applied
                ? "branch_experiment_started"
                : "branch_experiment_not_started";
            case SET_CURRICULUM_OBJECTIVE -> applied ? "curriculum_objective_set" : "curriculum_objective_not_changed";
            case PROMOTE_BRANCH -> applied ? "promotion_applied_or_already_current" : "promotion_not_applied";
            case JUMP_TO_BRANCH -> applied ? "branch_jump_applied_or_queued" : "branch_jump_not_applied";
            case STOP_TRAINING -> applied ? "stop_requested" : "stop_not_requested";
            case START_NEW_RUN, RESTART_TRAINING -> applied ? "new_run_started" : "new_run_not_started";
            case LOAD_EXISTING_MODEL -> applied ? "existing_model_loaded" : "existing_model_not_loaded";
            case REQUEST_PROMOTION_CHECK -> applied ? "promotion_check_requested" : "promotion_check_not_applied";
            case REQUEST_HISTORICAL_REPORT -> "historical_report_requested";
            case KEEP_GOING -> "no_runtime_change";
        };
    }

    public void recordSupervisorDecisionOutcome(SupervisorDecision decision, String scope) {
        boolean failed = decision == null || isSupervisorProviderFailure(decision);
        if (failed) {
            supervisorProviderFailureStreak++;
            lastSupervisorProviderFailureMs = System.currentTimeMillis();
            lastSupervisorProviderFailureReason = decision != null && decision.getReason() != null
                ? decision.getReason()
                : "Supervisor returned no decision" + (scope != null && !scope.isBlank() ? " during " + scope : "");
            return;
        }
        supervisorProviderFailureStreak = 0;
        lastSupervisorProviderFailureReason = "";
    }

    private boolean isSupervisorProviderFailure(SupervisorDecision decision) {
        if (decision == null || decision.getAction() != com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction.KEEP_GOING) {
            return false;
        }
        String reason = decision.getReason() != null ? decision.getReason().toLowerCase(Locale.ROOT) : "";
        return reason.contains("gemini")
            || reason.contains("api_key")
            || reason.contains("http 429")
            || reason.contains("http 500")
            || reason.contains("request failed")
            || reason.contains("timed out")
            || reason.contains("timeout");
    }

    public void recordSupervisorAppliedChange(String action, String subject, String reason) {
        recordAppliedChange(action, subject, reason);
    }

    private void recordAppliedChange(String action, String subject, String reason) {
        Map<String, Object> entry = new LinkedHashMap<>();
        long now = System.currentTimeMillis();
        entry.put("timestamp", Instant.ofEpochMilli(now).toString());
        entry.put("episode", episodesTrained);
        entry.put("action", action != null ? action : "");
        entry.put("subject", subject != null ? subject : "");
        entry.put("reason", reason != null ? reason : "");
        entry.put("activeModelName", activeTrainingModelName);
        entry.put("epsilon", epsilon);
        RLRewardConfig current = getRewardConfig();
        if (current != null) {
            entry.put("rewardConfig", current.toMap());
        }
        synchronized (appliedChangeJournal) {
            appliedChangeJournal.addLast(entry);
            while (appliedChangeJournal.size() > 12) {
                appliedChangeJournal.removeFirst();
            }
        }
    }

    private Map<String, Object> effectSince(Map<String, Object> previous, SupervisorObservation current) {
        Map<String, Object> effect = new LinkedHashMap<>();
        effect.put("episodesElapsed", current.totalEpisodesTrained - intMapValue(previous, "observationEpisode"));
        effect.put("bestScoreDelta", current.bestScore - doubleMapValue(previous, "observationBestScore"));
        effect.put("avgEvalScoreDelta", current.avgEvalScore - doubleMapValue(previous, "observationAvgEvalScore"));
        effect.put("invalidActionRateDelta", current.invalidActionRate - doubleMapValue(previous, "observationInvalidActionRate"));
        effect.put("epsilonDelta", current.epsilon - doubleMapValue(previous, "observationEpsilon"));
        effect.put("bestBaseBlocksDelta", current.bestBaseBlocks - intMapValue(previous, "observationBestBaseBlocks"));
        effect.put("bestBaseStillHasTC", current.bestBaseHasTC);
        return effect;
    }

    private java.util.List<Map<String, Object>> supervisorDecisionMemorySnapshot() {
        synchronized (supervisorDecisionMemory) {
            java.util.List<Map<String, Object>> copy = new java.util.ArrayList<>(supervisorDecisionMemory.size());
            for (Map<String, Object> entry : supervisorDecisionMemory) {
                copy.add(new LinkedHashMap<>(entry));
            }
            return copy;
        }
    }

    private static double doubleMapValue(Map<String, Object> map, String key) {
        Object value = map != null ? map.get(key) : null;
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        return 0.0;
    }

    private static int intMapValue(Map<String, Object> map, String key) {
        Object value = map != null ? map.get(key) : null;
        if (value instanceof Number number) {
            return number.intValue();
        }
        return 0;
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
        if (delay.isZero() || delay.isNegative()) {
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
        while (remainingMs > 0 && !stopRequested && !isTrainingTimeLimitReached()) {
            long sleepMs = Math.min(remainingMs, 1_000L);
            try {
                Thread.sleep(sleepMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
            remainingMs -= sleepMs;
        }
        if (stopRequested || shouldStopForTrainingTimeLimit()) {
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
        return decision.getProposedCallFrequency() != null
            || decision.getAction() == com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction.SET_EPSILON
            || decision.getAction() == com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction.REPLACE_REWARD_CONFIG
            || decision.getAction() == com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction.SET_CURRICULUM_OBJECTIVE
            || decision.getAction() == com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction.STOP_TRAINING
            || decision.getAction() == com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction.PROMOTE_BRANCH
            || decision.getAction() == com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction.JUMP_TO_BRANCH;
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
        String frequency = "";
        if (decision.getProposedCallFrequency() != null) {
            LlmSupervisorConfig.CallFrequency proposed = decision.getProposedCallFrequency();
            frequency = String.format(Locale.US, ", callFrequency=%s x%.2f effective=%d",
                proposed.name(),
                proposed.getMultiplier(),
                Math.max(1, (int) Math.round(config.getCallIntervalEpisodes() * proposed.getMultiplier())));
        }
        return String.format("Supervisor ep %d [%s]: %s%s, applied=%s, execution=%s, reason=%s",
            observation.totalEpisodesTrained,
            config.getApplyMode(),
            decision.getAction(),
            frequency,
            applied ? "yes" : "no",
            supervisorDecisionExecutionStatus(decision, applied),
            reason);
    }

    private boolean applySupervisorDecision(SupervisorDecision decision) {
        return applySupervisorDecision(decision, null);
    }

    private boolean applySupervisorDecision(SupervisorDecision decision, Map<String, Object> branchComparisonContext) {
        if (decision == null) {
            return false;
        }
        boolean applied = applySupervisorCallFrequency(decision.getProposedCallFrequency());

        switch (decision.getAction()) {
            case SET_EPSILON:
                if (decision.getProposedEpsilon() != null) {
                    return startBranchExperiment(decision) || applied;
                }
                break;
            case REPLACE_REWARD_CONFIG:
                if (decision.getProposedRewardConfig() != null) {
                    return startBranchExperiment(decision) || applied;
                }
                break;
            case SET_CURRICULUM_OBJECTIVE:
                return applyCurriculumObjective(decision) || applied;
            case STOP_TRAINING:
                requestStop();
                recordAppliedChange("STOP_TRAINING", activeTrainingModelName, decision.getReason());
                return true;
            case PROMOTE_BRANCH:
                return promoteBranchFromComparison(branchComparisonContext, decision.getReason()) || applied;
            case JUMP_TO_BRANCH:
                return requestBranchJump(decision.getProposedModelName(), decision.getReason()) || applied;
            case START_NEW_RUN:
            case RESTART_TRAINING:
            case LOAD_EXISTING_MODEL:
            case REQUEST_PROMOTION_CHECK:
            case KEEP_GOING:
            default:
                break;
        }
        return applied;
    }

    public boolean applySupervisorCallFrequency(LlmSupervisorConfig.CallFrequency frequency) {
        if (frequency == null || supervisorConfig == null || !supervisorConfig.isEnabled()) {
            return false;
        }
        if (supervisorConfig.getCallFrequency() == frequency) {
            return false;
        }
        supervisorConfig.setCallFrequency(frequency);
        emitSupervisorLog(String.format(Locale.US,
            "Supervisor call frequency set by LLM: %s x%.2f, effective interval %d episodes.",
            frequency.name(),
            frequency.getMultiplier(),
            supervisorConfig.getEffectiveCallIntervalEpisodes()));
        recordAppliedChange("SET_CALL_FREQUENCY", frequency.name(), "LLM selected next supervisor cadence.");
        return true;
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
            if (branchExperimentRunning || branchReviewInProgress) {
                branchExperimentStatus = "Branch experiment/review already running; new proposal ignored.";
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
            copyMainLogsToBranch(baselineConfig.getOutputDirectory());
            copyMainLogsToBranch(candidateConfig.getOutputDirectory());
            baselineService = branchTrainingServiceFactory.create();
            candidateService = branchTrainingServiceFactory.create();
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
                dualTrainingCoordinatorFactory.get().trainInParallel(
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
                branchComparatorFactory.get().compare(baselineMetrics, candidateMetrics);
            String savedBaselineModelName = saveBranchModel(result.baselineService, baselineName, baselineConfig);
            String savedCandidateModelName = saveBranchModel(result.candidateService, candidateName, candidateConfig);
            String savedWinnerModelName = comparison.promoteCandidate ? savedCandidateModelName : savedBaselineModelName;
            Map<String, Object> completedComparison = branchComparisonMap(
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
                reason,
                result.baselineService.getCurriculumStabilitySummary(),
                result.candidateService.getCurriculumStabilitySummary());
            latestBranchComparison = completedComparison;
            rememberBranchExperiment(latestBranchComparison);
            writeBranchExperimentLog(activeTrainingModelName, latestBranchComparison);
            RLModelManager.pruneGeneratedBranchDirectories(branchOwnerModelName(sourceConfig));
            branchExperimentStatus = String.format(
                "Branch experiment finished: candidate=%s savedWinner=%s promoteRecommended=%s bestDelta=%.4f avgDelta=%.4f invalidDelta=%.4f",
                candidateName,
                savedWinnerModelName,
                comparison.promoteCandidate,
                comparison.bestScoreDelta,
                comparison.avgEvalDelta,
                comparison.invalidRateDelta);
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
            branchReviewInProgress = true;
            branchExperimentRunning = false;
            maybeInvokeSupervisorAfterBranchExperiment(activeTrainingModelName, baselineMetrics, candidateMetrics, completedComparison);
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
            branchReviewInProgress = false;
        }
    }

    private void maybeInvokeSupervisorAfterBranchExperiment(String modelName,
                                                            TrainingMetrics baselineMetrics,
                                                            TrainingMetrics candidateMetrics,
                                                            Map<String, Object> branchComparison) {
        LlmSupervisorConfig config = supervisorConfig;
        if (config == null || !config.isEnabled()) {
            return;
        }
        TrainingMetrics liveMetrics = getMetrics();
        Map<String, Object> trendMetrics = buildSupervisorTrendMetrics(liveMetrics);
        Map<String, Object> comparisonForReview = branchComparison != null
            ? new LinkedHashMap<>(branchComparison)
            : latestBranchComparisonSnapshot();
        trendMetrics.put("latestBranchComparison", comparisonForReview);
        trendMetrics.put("latestBranchPromotionOpen", isBranchPromotionOpen(comparisonForReview));
        trendMetrics.put("supervisorTrigger", "BRANCH_EXPERIMENT_FINISHED");
        trendMetrics.put("branchReviewInstruction",
            "Review baseline and candidate branch results now. The timed supervisor interval was paused while both branches trained.");
        trendMetrics.put("baselineBranchFinal", metricsMap(baselineMetrics));
        trendMetrics.put("candidateBranchFinal", metricsMap(candidateMetrics));

        SupervisorObservation observation = trainingAnalyzer.summarize(
            config.getBranchId(),
            liveMetrics,
            null,
            getRewardConfig(),
            buildSupervisorTrainingContext(liveMetrics, config),
            trendMetrics);

        rememberSupervisorTrend(liveMetrics);
        emitSupervisorLog(String.format(
            "Supervisor branch review: baseline=%s candidate=%s winner=%s",
            latestBranchComparison.getOrDefault("savedBaselineModelName", ""),
            latestBranchComparison.getOrDefault("savedCandidateModelName", ""),
            latestBranchComparison.getOrDefault("savedWinnerModelName", "")));
        if (!waitForSupervisorRateLimit(observation)) {
            return;
        }

        SupervisorDecision rawDecision;
        try {
            rawDecision = llmSupervisor.review(observation);
        } catch (Exception e) {
            emitSupervisorLog("Supervisor branch review failed: " + e.getMessage());
            rawDecision = SupervisorDecision.keepGoing("Supervisor branch review failed: " + e.getMessage());
        }

        Map<String, Object> diagnostics = supervisorDiagnostics();
        SupervisorDecision decision = supervisorDecisionValidator.validate(rawDecision, config, getRewardConfig(), observation);
        recordSupervisorDecisionOutcome(decision, "branch_review");
        boolean applied = shouldApplySupervisorDecision(config, decision);
        if (applied) {
            applied = applySupervisorDecision(decision, comparisonForReview);
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
        supervisorDecisionLogWriter.write(modelName, observation, decision, applied, diagnostics);
        rememberSupervisorDecision(observation, decision, applied, diagnostics);
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
        LlmSupervisorConfig sourceSupervisor = supervisorConfig != null && supervisorConfig.isEnabled()
            ? supervisorConfig
            : source.getSupervisorConfig();
        int episodes = sourceSupervisor != null && sourceSupervisor.getEffectiveCallIntervalEpisodes() > 0
            ? sourceSupervisor.getEffectiveCallIntervalEpisodes()
            : source.getEpisodesPerEpoch();
        episodes = Math.max(1, episodes);
        int epochs = 1;
        java.nio.file.Path outputDirectory = RLModelManager.getBranchDirectory(branchOwnerModelName(source), modelName);
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
            use2dCnnBranch,
            outputDirectory);
    }

    private String branchOwnerModelName(RLTrainingConfig source) {
        if (activeTrainingModelName != null && !activeTrainingModelName.isBlank()) {
            return activeTrainingModelName;
        }
        if (source != null && source.getModelName() != null && !source.getModelName().isBlank()) {
            return source.getModelName();
        }
        return "autopilot";
    }

    private void copyMainLogsToBranch(java.nio.file.Path branchOutputDirectory) {
        if (branchOutputDirectory == null) {
            return;
        }
        String owner = branchOwnerModelName(activeTrainingConfig);
        java.nio.file.Path mainDir = RLModelManager.getModelMainDirectory(owner);
        java.nio.file.Path inheritedDir = branchOutputDirectory.resolve("inherited_main");
        try {
            java.nio.file.Files.createDirectories(inheritedDir);
            try (java.util.stream.Stream<java.nio.file.Path> files = java.nio.file.Files.list(mainDir)) {
                files
                    .filter(java.nio.file.Files::isRegularFile)
                    .filter(path -> {
                        String name = path.getFileName().toString();
                        return name.endsWith(".csv") || name.endsWith(".json");
                    })
                    .forEach(path -> {
                        try {
                            java.nio.file.Files.copy(
                                path,
                                inheritedDir.resolve(path.getFileName()),
                                java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                        } catch (Exception ignored) {
                        }
                    });
            }
        } catch (Exception e) {
            emitSupervisorLog("[BRANCH] Could not copy inherited main logs: " + e.getMessage());
        }
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
                                                    String reason,
                                                    Map<String, Object> baselineStability,
                                                    Map<String, Object> candidateStability) {
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
        map.put("curriculumObjective", currentCurriculumObjectiveSnapshot());
        map.put("baseline", metricsMap(baseline));
        map.put("candidate", metricsMap(candidate));
        map.put("baselineCurriculumStability", baselineStability != null ? baselineStability : Map.of());
        map.put("candidateCurriculumStability", candidateStability != null ? candidateStability : Map.of());
        map.put("baselineLogEvidence", branchLogEvidence(baselineName));
        map.put("candidateLogEvidence", branchLogEvidence(candidateName));
        map.put("promotionApplied", false);
        map.put("promotionStatus", comparison.promoteCandidate ? "pending_review" : "not_recommended");
        map.put("nextAllowedDecision", comparison.promoteCandidate
            ? "Return PROMOTE_BRANCH only once to apply the saved candidate branch, or KEEP_GOING/REPLACE_REWARD_CONFIG to keep exploring."
            : "Candidate did not win; do not return PROMOTE_BRANCH for this comparison.");
        return map;
    }

    private Map<String, Object> branchLogEvidence(String branchModelName) {
        Map<String, Object> evidence = new LinkedHashMap<>();
        String owner = branchOwnerModelName(activeTrainingConfig);
        java.nio.file.Path dir = RLModelManager.getBranchDirectoryPath(owner, branchModelName);
        evidence.put("directory", dir.toString());
        evidence.put("hasEpisodesCsv", java.nio.file.Files.exists(dir.resolve(branchModelName + "_episodes.csv")));
        evidence.put("hasTrainingCsv", java.nio.file.Files.exists(dir.resolve(branchModelName + "_multi_discrete_training.csv")));
        evidence.put("hasPerformanceCsv", java.nio.file.Files.exists(dir.resolve(branchModelName + "_performance_tmp.csv")));
        evidence.put("hasRunMetadata", java.nio.file.Files.exists(dir.resolve(branchModelName + "_run_metadata.json")));
        evidence.put("hasSnapshot", java.nio.file.Files.exists(dir.resolve(branchModelName + ".rmeta"))
            && java.nio.file.Files.exists(dir.resolve(branchModelName + ".rnet")));
        return evidence;
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
        RLModelManager.saveModel(snapshot, branchService, config.getOutputDirectory());
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
        RLModelManager.saveModel(snapshot, this, RLModelManager.getBranchDirectory(branchOwnerModelName(config), branchModelName));
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

    private void writeBranchExperimentLog(String modelName, Map<String, Object> comparison) {
        if (comparison == null || comparison.isEmpty()) {
            return;
        }
        String safeModelName = modelName == null || modelName.isBlank() ? "autopilot" : modelName.trim();
        java.nio.file.Path dir = RLModelManager.getModelLlmDirectory(safeModelName);
        java.nio.file.Path jsonlPath = dir.resolve(safeModelName + "_branch_experiments.jsonl");
        java.nio.file.Path csvPath = dir.resolve(safeModelName + "_branch_experiments.csv");
        String timestamp = java.time.LocalDateTime.now().toString();
        try {
            Map<String, Object> jsonLine = new LinkedHashMap<>();
            jsonLine.put("timestamp", timestamp);
            jsonLine.put("comparison", comparison);
            java.nio.file.Files.writeString(jsonlPath,
                SimpleJson.stringify(jsonLine) + System.lineSeparator(),
                java.nio.charset.StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.CREATE,
                java.nio.file.StandardOpenOption.APPEND);

            boolean needsHeader = !java.nio.file.Files.exists(csvPath) || java.nio.file.Files.size(csvPath) == 0;
            StringBuilder csv = new StringBuilder();
            if (needsHeader) {
                csv.append(String.join(",",
                    "timestamp",
                    "baseline_model",
                    "candidate_model",
                    "saved_baseline_model",
                    "saved_candidate_model",
                    "saved_winner_model",
                    "saved_winner_branch",
                    "promote_recommended",
                    "best_score_delta",
                    "avg_eval_delta",
                    "invalid_rate_delta",
                    "comparison_reason")).append(System.lineSeparator());
            }
            csv.append(String.join(",",
                csv(timestamp),
                csv(stringMapValue(comparison, "baselineModelName")),
                csv(stringMapValue(comparison, "candidateModelName")),
                csv(stringMapValue(comparison, "savedBaselineModelName")),
                csv(stringMapValue(comparison, "savedCandidateModelName")),
                csv(stringMapValue(comparison, "savedWinnerModelName")),
                csv(stringMapValue(comparison, "savedWinnerBranch")),
                csv(String.valueOf(comparison.getOrDefault("promoteRecommended", false))),
                csv(String.format(Locale.US, "%.6f", numberMapValue(comparison, "bestScoreDelta"))),
                csv(String.format(Locale.US, "%.6f", numberMapValue(comparison, "avgEvalDelta"))),
                csv(String.format(Locale.US, "%.6f", numberMapValue(comparison, "invalidRateDelta"))),
                csv(stringMapValue(comparison, "comparisonReason")))).append(System.lineSeparator());
            java.nio.file.Files.writeString(csvPath,
                csv.toString(),
                java.nio.charset.StandardCharsets.UTF_8,
                java.nio.file.StandardOpenOption.CREATE,
                java.nio.file.StandardOpenOption.APPEND);
            emitSupervisorLog("[BRANCH] Branch comparison logged to " + jsonlPath);
        } catch (Exception e) {
            emitSupervisorLog("[BRANCH] Branch comparison log failed: " + e.getMessage());
        }
    }

    private static String stringMapValue(Map<String, Object> map, String key) {
        Object value = map != null ? map.get(key) : null;
        return value != null ? String.valueOf(value) : "";
    }

    private static double numberMapValue(Map<String, Object> map, String key) {
        Object value = map != null ? map.get(key) : null;
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        return 0.0;
    }

    private static String csv(String value) {
        if (value == null) {
            return "";
        }
        return "\"" + value
            .replace("\"", "\"\"")
            .replace("\r", " ")
            .replace("\n", " ") + "\"";
    }

    private java.util.List<Map<String, Object>> branchHistorySnapshot() {
        synchronized (branchExperimentLock) {
            java.util.List<Map<String, Object>> copy = new java.util.ArrayList<>(branchHistory.size());
            for (Map<String, Object> entry : branchHistory) {
                copy.add(new LinkedHashMap<>(entry));
            }
            return copy;
        }
    }

    private Map<String, Object> latestBranchComparisonSnapshot() {
        Map<String, Object> comparison = latestBranchComparison;
        return comparison == null || comparison.isEmpty()
            ? Map.of()
            : new LinkedHashMap<>(comparison);
    }

    private boolean isBranchPromotionOpen(Map<String, Object> comparison) {
        if (comparison == null || comparison.isEmpty()) {
            return false;
        }
        if (Boolean.TRUE.equals(comparison.get("promotionApplied"))) {
            return false;
        }
        return Boolean.TRUE.equals(comparison.get("promoteRecommended"))
            && "candidate".equals(String.valueOf(comparison.getOrDefault("savedWinnerBranch", "")));
    }

    private java.util.List<String> promotedBranchModelNamesSnapshot() {
        synchronized (promotedBranchModelNames) {
            return java.util.List.copyOf(promotedBranchModelNames);
        }
    }

    private java.util.List<Map<String, Object>> appliedChangeJournalSnapshot() {
        synchronized (appliedChangeJournal) {
            java.util.List<Map<String, Object>> copy = new java.util.ArrayList<>(appliedChangeJournal.size());
            for (Map<String, Object> entry : appliedChangeJournal) {
                copy.add(new LinkedHashMap<>(entry));
            }
            return copy;
        }
    }

    private Map<String, Object> supervisorHealthSnapshot() {
        Map<String, Object> health = new LinkedHashMap<>();
        health.put("providerFailureStreak", supervisorProviderFailureStreak);
        health.put("degraded", supervisorProviderFailureStreak >= 3);
        health.put("lastFailureReason", lastSupervisorProviderFailureReason);
        health.put("lastFailureAt", lastSupervisorProviderFailureMs > 0
            ? Instant.ofEpochMilli(lastSupervisorProviderFailureMs).toString()
            : "");
        health.put("guidance", "If degraded is true, avoid high-impact stale actions unless current evidence is fresh and explicit.");
        return health;
    }

    private java.util.List<String> knownBranchModelNamesSnapshot() {
        synchronized (knownBranchModelNames) {
            return java.util.List.copyOf(knownBranchModelNames);
        }
    }

    private java.util.List<String> availableBranchModelNamesSnapshot() {
        java.util.LinkedHashSet<String> names = new java.util.LinkedHashSet<>(knownBranchModelNamesSnapshot());
        for (String modelName : RLModelManager.listBranchModels()) {
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
        return isBranchLikeModelName(modelName) && RLModelManager.listBranchModels().contains(modelName);
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

    private Map<String, Object> evaluationDiagnostics(EpisodeResult result) {
        Map<String, Object> diagnostics = new LinkedHashMap<>();
        if (result == null) {
            diagnostics.put("evaluationSampleAvailable", false);
            return diagnostics;
        }
        int finalBlockCount = result.grid != null ? result.grid.getAllBlocks().size() : result.blocksPlaced;
        boolean evaluationRan = result.evaluationResult != null;
        boolean allEvalScoresZero = evaluationRan
            && result.evalLogisticsScore == 0.0
            && result.evalCostScore == 0.0
            && result.evalRaidScore == 0.0
            && result.evalWorkingAreaScore == 0.0
            && result.evalSafeZoneScore == 0.0;
        diagnostics.put("evaluationSampleAvailable", true);
        diagnostics.put("evaluationRan", evaluationRan);
        diagnostics.put("allEvalScoresZero", allEvalScoresZero);
        diagnostics.put("finalBlockCount", finalBlockCount);
        diagnostics.put("blocksPlaced", result.blocksPlaced);
        diagnostics.put("hasTC", result.hasTC || result.finalRewardHasTC);
        diagnostics.put("raidSulfurToTC", result.raidSulfurToTC);
        diagnostics.put("componentCount", result.componentCount);
        diagnostics.put("mainComponentBlocks", result.mainComponentBlocks);
        diagnostics.put("stopReason", result.stopReason != null ? result.stopReason.name() : "");
        diagnostics.put("zeroEvalInterpretation", zeroEvalInterpretation(result, finalBlockCount, evaluationRan, allEvalScoresZero));
        return diagnostics;
    }

    private String zeroEvalInterpretation(EpisodeResult result,
                                          int finalBlockCount,
                                          boolean evaluationRan,
                                          boolean allEvalScoresZero) {
        if (!evaluationRan) {
            return finalBlockCount <= 5
                ? "final evaluator skipped because the episode ended with five or fewer blocks"
                : "final evaluator did not produce a result despite enough blocks";
        }
        if (!allEvalScoresZero) {
            return "evaluation produced at least one non-zero component";
        }
        if (!result.hasTC && !result.finalRewardHasTC) {
            return "zero eval components are expected when the episode has no TC/protected objective target";
        }
        if (result.raidSulfurToTC == 0) {
            return "TC appears reachable from outside, so logistics/raid quality can legitimately score zero";
        }
        return "all evaluator components are zero even though evaluation ran; inspect graph connectivity and protected-area conditions";
    }

    public boolean promoteLatestBranch() {
        return promoteBranchFromComparison(latestBranchComparisonSnapshot(), "Promote candidate branch to main");
    }

    private boolean promoteBranchFromComparison(Map<String, Object> comparison, String reason) {
        Map<String, Object> safeComparison = comparison != null ? comparison : Map.of();
        String candidateModelName = stringMapValue(safeComparison, "savedCandidateModelName");
        if (candidateModelName.isBlank()) {
            candidateModelName = latestBranchCandidateModelName;
        }
        if (candidateModelName == null || candidateModelName.isBlank()) {
            branchExperimentStatus = "No saved candidate branch model available to promote.";
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
            return false;
        }
        if (isAlreadyPromotedBranch(candidateModelName)) {
            markBranchPromotionApplied(candidateModelName, "already_applied");
            branchExperimentStatus = "Candidate branch already promoted; duplicate PROMOTE_BRANCH ignored: " + candidateModelName;
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
            return true;
        }
        if (!isKnownBranchModelName(candidateModelName)) {
            branchExperimentStatus = "No saved candidate branch model available to promote.";
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
            return false;
        }
        if (!safeComparison.isEmpty() && !isBranchPromotionOpen(safeComparison)) {
            branchExperimentStatus = "Promotion skipped because the latest branch comparison is not open for promotion: " + candidateModelName;
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
            return false;
        }
        return requestBranchJump(candidateModelName, reason != null && !reason.isBlank()
            ? reason
            : "Promote candidate branch to main", true);
    }

    private boolean isAlreadyPromotedBranch(String modelName) {
        return modelName != null
            && !modelName.isBlank()
            && (modelName.equals(lastPromotedBranchModelName) || promotedBranchModelNames.contains(modelName));
    }

    private void markBranchPromotionApplied(String modelName, String status) {
        if (modelName == null || modelName.isBlank()) {
            return;
        }
        Map<String, Object> current = latestBranchComparisonSnapshot();
        if (!current.isEmpty()
                && (modelName.equals(stringMapValue(current, "savedCandidateModelName"))
                    || modelName.equals(stringMapValue(current, "candidateModelName"))
                    || modelName.equals(stringMapValue(current, "savedWinnerModelName")))) {
            Map<String, Object> updated = new LinkedHashMap<>(current);
            updated.put("promotionApplied", true);
            updated.put("promotionStatus", status != null ? status : "applied");
            updated.put("promotedModelName", modelName);
            updated.put("promotedAt", Instant.ofEpochMilli(System.currentTimeMillis()).toString());
            updated.put("nextAllowedDecision", "This candidate has already been promoted. Do not return PROMOTE_BRANCH for it again.");
            latestBranchComparison = updated;
            updateBranchHistoryEntry(updated);
        }
        latestBranchCandidateModelName = "";
        latestBranchCandidateRewardConfig = null;
        latestBranchCandidateEpsilon = null;
    }

    private void updateBranchHistoryEntry(Map<String, Object> updatedComparison) {
        String candidate = stringMapValue(updatedComparison, "savedCandidateModelName");
        if (candidate.isBlank()) {
            return;
        }
        synchronized (branchExperimentLock) {
            for (Map<String, Object> entry : branchHistory) {
                if (candidate.equals(stringMapValue(entry, "savedCandidateModelName"))) {
                    entry.clear();
                    entry.putAll(updatedComparison);
                    return;
                }
            }
        }
    }

    public boolean requestBranchJump(String modelName, String reason) {
        return requestBranchJump(modelName, reason, false);
    }

    private boolean requestBranchJump(String modelName, String reason, boolean promotion) {
        String safeModelName = modelName != null ? modelName.trim() : "";
        if (safeModelName.isBlank() || !isKnownBranchModelName(safeModelName)) {
            branchExperimentStatus = "Refused branch jump to unknown model: " + safeModelName;
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
            return false;
        }
        String ownerModelName = activeMainOwnerForBranch(safeModelName);
        pendingBranchSwitchModelName = safeModelName;
        pendingBranchSwitchOwnerModelName = ownerModelName;
        pendingBranchSwitchReason = reason != null ? reason : "";
        pendingBranchSwitchPromotion = promotion;
        pendingBranchSwitchRequested = true;
        branchExperimentStatus = "Queued branch jump to " + safeModelName + " as main " + ownerModelName
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
        pendingAutoResume = true;
        String modelName = pendingBranchSwitchModelName;
        String ownerModelName = pendingBranchSwitchOwnerModelName;
        String reason = pendingBranchSwitchReason;
        boolean promotion = pendingBranchSwitchPromotion;
        pendingBranchSwitchRequested = false;
        pendingBranchSwitchModelName = "";
        pendingBranchSwitchOwnerModelName = "";
        pendingBranchSwitchReason = "";
        pendingBranchSwitchPromotion = false;
        if (modelName == null || modelName.isBlank()) {
            return false;
        }
        if (ownerModelName == null || ownerModelName.isBlank()) {
            ownerModelName = activeMainOwnerForBranch(modelName);
        }
        try {
            String previousModel = ownerModelName == null || ownerModelName.isBlank() ? "live" : ownerModelName;
            LlmSupervisorConfig previousSupervisorConfig = getSupervisorConfig();
            String savedPrevious = saveLiveBranchModel(safeBranchModelName(previousModel + "_before_jump_" + System.currentTimeMillis()));
            copyMainLogsToBranch(RLModelManager.getBranchDirectory(ownerModelName, savedPrevious));
            RLModelManager.RLModel model = RLModelManager.loadMetadata(modelName);
            RLModelManager.restoreFromModel(this, model);
            RLModelManager.loadNetworkWeights(modelName, this);
            setSupervisorConfig(previousSupervisorConfig);
            int lineageEpisodes = RLModelManager.promoteBranchLineageToMain(ownerModelName, modelName);
            if (lineageEpisodes > getEpisodesTrained()) {
                setEpisodesTrained(lineageEpisodes);
            }
            RLModelManager.RLModel mainSnapshot = RLModelManager.createSnapshot(
                ownerModelName,
                this,
                activeTrainingConfig != null ? activeTrainingConfig.getLogisticsWeight() : model.logisticsWeight,
                activeTrainingConfig != null ? activeTrainingConfig.getCostWeight() : model.costWeight,
                activeTrainingConfig != null ? activeTrainingConfig.getRaidWeight() : model.raidWeight,
                activeTrainingConfig != null ? activeTrainingConfig.getWorkingAreaWeight() : model.workingAreaWeight,
                activeTrainingConfig != null ? activeTrainingConfig.getSafeZoneWeight() : model.safeZoneWeight);
            RLModelManager.saveModel(mainSnapshot, this, RLModelManager.getModelMainDirectory(ownerModelName));
            activeTrainingModelName = ownerModelName;
            if (promotion) {
                promotedBranchModelNames.add(modelName);
                lastPromotedBranchModelName = modelName;
                lastPromotedBranchAtMs = System.currentTimeMillis();
                markBranchPromotionApplied(modelName, "applied");
                recordAppliedChange("PROMOTE_BRANCH", modelName, reason);
            } else {
                recordAppliedChange("JUMP_TO_BRANCH", modelName, reason);
            }
            branchExperimentStatus = "Jumped to branch " + modelName
                + "; main " + ownerModelName + " now follows that lineage"
                + "; previous live branch saved as " + savedPrevious;
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
            return true;
        } catch (Exception e) {
            branchExperimentStatus = "Branch jump failed for " + modelName + ": " + e.getMessage();
            emitSupervisorLog("[BRANCH] " + branchExperimentStatus);
            return false;
        }
    }

    private String activeMainOwnerForBranch(String branchModelName) {
        String active = activeTrainingModelName != null ? activeTrainingModelName.trim() : "";
        if (!active.isBlank() && !isBranchLikeModelName(active)) {
            return active;
        }
        if (activeTrainingConfig != null
                && activeTrainingConfig.getModelName() != null
                && !activeTrainingConfig.getModelName().isBlank()
                && !isBranchLikeModelName(activeTrainingConfig.getModelName())) {
            return activeTrainingConfig.getModelName().trim();
        }
        String derived = deriveOwnerFromBranchModelName(branchModelName);
        return !derived.isBlank() ? derived : "autopilot";
    }

    private String deriveOwnerFromBranchModelName(String branchModelName) {
        String name = branchModelName != null ? branchModelName.trim() : "";
        return name
            .replaceFirst("_(baseline|candidate|before_jump)_\\d+$", "")
            .trim();
    }

    public String getSupervisorBranchStatus() {
        return branchExperimentStatus != null ? branchExperimentStatus : "";
    }

    public boolean isSupervisorBranchWorkInProgress() {
        return branchExperimentRunning || branchReviewInProgress;
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
        com.rustbuilder.service.physics.PlacementResult placement = PlacementService.calculatePlacement(gridModel, action);

        res.minDist = placement.nearestDistance();
        res.socketDist = placement.socketDistance();

        if (!(placement instanceof com.rustbuilder.service.physics.PlacementResult.Valid validPlacement)) {
            res.error = (placement.error() != PlacementError.NONE) ? placement.error() : PlacementError.UNKNOWN;
            res.failReason = "invalid_placement(" + res.error + "): " + action.actionType;
            return res;
        }

        double finalRotation = validPlacement.rotation();
        Orientation finalOrientation = validPlacement.orientation();
        int z = (action.actionType == BuildAction.ActionType.FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) ? 0 : action.floor;
        if (z < 0) z = 0;

        BuildingTier tier = com.rustbuilder.util.BlockFactory.tierFromInt(action.tier);
        DoorType doorType = com.rustbuilder.util.BlockFactory.doorTypeFromInt(action.doorType);
        BuildingBlock block = PlacementService.createRealBlock(action, validPlacement, tier, doorType);

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
        resetCurriculumStateForNewRun("runtime_reset");
        clearPendingSupervisorDecision();
        resetSupervisorTrendState();

        if (this.multiDiscreteMemory != null) {
            this.multiDiscreteMemory.clear();
        }

        // Re-create memory buffers to clear them
        this.multiDiscreteMemory = trainingAgentFactory.createReplay(MEMORY_CAPACITY);
    }

    public boolean hasPendingAutoResume() {
        return pendingAutoResume;
    }

    public void clearPendingAutoResume() {
        this.pendingAutoResume = false;
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
            GridModel snapshot = gridModelFactory.create();
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
            GridModel snapshot = gridModelFactory.create();
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

    public com.rustbuilder.ai.rl.environment.spec.EncodingRuntimeConfig getRuntimeConfig() {
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
        this.multiDiscretePolicy = trainingAgentFactory.createPolicy(provider);
    }

    private boolean isFoundation(BuildingBlock b) {
        if (b == null) return false;
        return b.getType() == BuildingType.FOUNDATION || b.getType() == BuildingType.TRIANGLE_FOUNDATION;
    }
}
