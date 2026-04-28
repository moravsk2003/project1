package com.rustbuilder.ai.rl;

import com.rustbuilder.ai.core.TrainingMetrics;
import java.util.LinkedList;
import java.util.Random;
import java.util.function.Consumer;
import org.nd4j.linalg.api.ndarray.INDArray;
import com.rustbuilder.ai.ea.BaseGenome.BuildAction;
import com.rustbuilder.ai.rl.legacy.*;
import com.rustbuilder.ai.rl.multidiscrete.*;
import com.rustbuilder.ai.rl.nn.DQNAgent;
import com.rustbuilder.ai.rl.nn.ExperienceReplay;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.structure.Door;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.service.evaluator.HouseEvaluator;

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

    private DQNAgent agent;
    private ExperienceReplay memory;
    private final HouseEvaluator evaluator;
    private final Random random;

    /**
     * Experimental 5-phase multi-discrete action space integration.
     */
    private MultiDiscreteAction currentMultiAction;
    
    // Hyperparameters
    private double epsilon = 1.0;
    private double epsilonDecay;
    private final double minEpsilon = 0.05;
    private final int batchSize = 32;
    private final int targetUpdateFreq = 10;
    private static final int MEMORY_CAPACITY = 10000;
    
    // Runtime State
    private int episodesTrained = 0;
    private double bestScore = -1;
    private GridModel bestGridModel;
    private GridModel currentGridModel;
    private double lastTrainLoss = 0;
    
    // Experimental 5-phase flow settings
    private boolean useMultiDiscreteFlow = true;
    private boolean useMultiDiscreteLearning = true;
    private MultiDiscretePhasePolicy multiDiscretePolicy;
    private NeuralMultiDiscreteDecisionProvider multiDiscreteNeuralProvider;
    private MultiDiscreteDQNAgent multiDiscreteAgent;
    private MultiDiscreteExperienceReplay multiDiscreteMemory;
    private MultiDiscreteStateObserver multiDiscreteObserver;
    // Legacy QTable fallback maintained for internal debug baseline
    private QTableMultiDiscreteDecisionProvider multiDiscreteLearningProvider;
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
    
    private RLRewardConfig rewardConfig;
    
    private volatile boolean stopRequested = false;
    private volatile boolean trainingRunning = false;
    private long trainingStartTime = 0;
    private long totalTrainingTimeMs = 0;
    
    // Training log (file I/O delegated to RLTrainingLogger)
    private final RLTrainingLogger logger = new RLTrainingLogger();

    public RLTrainingService() {
        this.memory = new ExperienceReplay(MEMORY_CAPACITY);
        this.evaluator = new HouseEvaluator();
        this.bestGridModel = new GridModel();
        this.currentGridModel = new GridModel();
        this.random = new Random();
        
        // Neural Multi-Discrete Flow configuration
        this.rewardConfig = RLRewardConfig.createDefault();
        this.multiDiscreteMemory = new MultiDiscreteExperienceReplay(MEMORY_CAPACITY);
        this.multiDiscreteAgent = new MultiDiscreteDQNAgent(StateEncoder.CHANNELS, StateEncoder.MAX_FLOORS, StateEncoder.GRID_SIZE, StateEncoder.GRID_SIZE);
        this.multiDiscreteAgent.setRewardConfig(this.rewardConfig);
        this.multiDiscreteNeuralProvider = new NeuralMultiDiscreteDecisionProvider(this.multiDiscreteAgent);
        
        // Legacy QTable baseline
        this.multiDiscreteLearningProvider = new QTableMultiDiscreteDecisionProvider();
        
        // Default policy is Neural
        this.multiDiscretePolicy = new ProvidedPhaseMultiDiscretePolicy(this.multiDiscreteNeuralProvider);
        
        // Two-phase action space: 11 types + 2048 positions = 2059 total outputs
        this.agent = new DQNAgent(StateEncoder.CHANNELS, StateEncoder.MAX_FLOORS, StateEncoder.GRID_SIZE, StateEncoder.GRID_SIZE, ActionSpace.TOTAL_ACTIONS);
    }
    
    /**
     * Train for a number of epochs, each epoch running 'episodes' training episodes.
     * Progress callback receives a TrainingMetrics object containing current status.
     */
    public void train(int episodes, int maxStepsPerEpisode, double logW, double costW, double raidW, double workingAreaW,
                      int epochs, Consumer<TrainingMetrics> progressCallback, Runnable epochCompleteCallback) {
        evaluator.setWeights(logW, costW, raidW, workingAreaW);
        
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
        
        logger.init();
        
        try {
            for (int epoch = 0; epoch < epochs; epoch++) {
            if (stopRequested) break;
            double epochTotalReward = 0;
            int epochTotalInvalid = 0;
            int epochTotalActions = 0;
            int epochTotalBlocks = 0;
            double epochBestEpScore = -1;
            long epochStartMs = System.currentTimeMillis();
            
            EpisodeEvaluator episodeEvaluator = new EpisodeEvaluator(evaluator, rewardConfig, useMultiDiscreteFlow);
            EpisodeRunner runner = new EpisodeRunner(agent, memory, multiDiscreteAgent, multiDiscreteMemory, multiDiscretePolicy, multiDiscreteObserver, random, rewardConfig, useMultiDiscreteLearning, logger, this);
            
            for (int ep = 0; ep < episodes; ep++) {
                if (stopRequested) break;
                EpisodeResult result;
                if (useMultiDiscreteFlow) {
                    // Set epsilon BEFORE the episode to ensure correct exploration rate
                    if (useMultiDiscreteLearning && multiDiscreteNeuralProvider != null) {
                        multiDiscreteNeuralProvider.setEpsilon(epsilon);
                    }
                    result = runner.runExperimentalMultiDiscreteEpisode(maxStepsPerEpisode, episodesTrained);
                } else {
                    result = runner.runLegacyEpisode(maxStepsPerEpisode, epsilon, batchSize);
                }
                
                this.currentMultiAction = runner.getCurrentMultiAction();
                if (runner.getLastTrainLoss() > 0) {
                    this.lastTrainLoss = runner.getLastTrainLoss();
                }
                
                episodeEvaluator.evaluate(result, maxStepsPerEpisode, epoch, ep, episodes, useMultiDiscreteLearning, multiDiscreteMemory);
                
                writeEpisodeLog(epoch + 1, ep + 1, episodesTrained + 1, result, epsilon);
                
                if (!useMultiDiscreteFlow) {
                    finalizeLegacyTraining(result);
                }
                updateBestBase(result);
                
                double episodeEvalScore = 0.0;
                if (result.evaluationResult != null) {
                    episodeEvalScore = result.evaluationResult.finalScore;
                }
                recentEvalScores.addLast(episodeEvalScore);
                if (recentEvalScores.size() > AVG_WINDOW) recentEvalScores.removeFirst();
                avgEvalScore = recentEvalScores.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                
                if (ep > 0 && ep % targetUpdateFreq == 0) {
                    agent.updateTargetNetwork();
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
                
                if (progressCallback != null && (ep % 5 == 0 || ep == episodes - 1)) {
                    double invalidRate = result.totalActions > 0 ? (double) result.invalidActions / result.totalActions : 0.0;
                    double loss = lastTrainLoss;
                    
                    TrainingMetrics metrics = new TrainingMetrics(
                        epoch + 1, epochs, ep + 1, episodes,
                        episodesTrained, bestScore, epsilon, loss,
                        avgReward, invalidRate, result.invalidActions, result.totalActions,
                        bestBaseBlocks, bestBaseHasTC, bestBaseDoors,
                        avgEvalScore, episodeEvalScore, result.accStepReward, result.finalEvalReward,
                        useMultiDiscreteFlow ? multiDiscreteMemory.size() : memory.size()
                    );
                    progressCallback.accept(metrics);
                }
                
                epochTotalReward += result.accStepReward;
                epochTotalActions += result.totalActions;
                epochTotalInvalid += result.invalidActions;
                epochTotalBlocks += result.blocksPlaced;
                if (result.finalEvalReward > epochBestEpScore) epochBestEpScore = result.finalEvalReward;
            }
            
            writeEpochLog(epoch + 1, episodes, epochTotalReward, epochTotalInvalid, epochTotalActions,
                          epochTotalBlocks, epochBestEpScore, System.currentTimeMillis() - epochStartMs);
            
            if (epochCompleteCallback != null) {
                epochCompleteCallback.run();
            }
        }
        } finally {
            logger.close();
            totalTrainingTimeMs += System.currentTimeMillis() - trainingStartTime;
            trainingRunning = false;
        }
    }



    /**
     * Track best base based on evaluation score.
     */
    private void updateBestBase(EpisodeResult result) {
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
        com.rustbuilder.util.GridPlacementUtils.Placement placement = com.rustbuilder.util.GridPlacementUtils.calculatePlacement(gridModel, action);
        
        res.minDist = placement.minDist;
        res.socketDist = placement.socketDist;

        if (!placement.valid) {
            res.error = (placement.error != PlacementError.NONE) ? placement.error : PlacementError.UNKNOWN;
            res.failReason = "invalid_placement(" + res.error + "): " + action.actionType;
            return res;
        }

        double finalX = placement.x;
        double finalY = placement.y;
        double finalRotation = placement.rotation;
        Orientation finalOrientation = placement.orientation;
        int z = (action.actionType == BuildAction.ActionType.FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) ? 0 : action.floor;
        if (z < 0) z = 0;

        BuildingTier tier = com.rustbuilder.util.BlockFactory.tierFromInt(action.tier);
        DoorType doorType = com.rustbuilder.util.BlockFactory.doorTypeFromInt(action.doorType);
        BuildingBlock block = com.rustbuilder.util.BlockFactory.create(
                action.actionType, finalX, finalY, z, finalRotation, finalOrientation, doorType);

        if (block != null) {
            block.setRotation(finalRotation);
            block.setTier(tier);
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
                        Door d = new Door(finalX, finalY, z, finalOrientation, doorType);
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
        this.rewardConfig = (rewardConfig != null) ? rewardConfig.clone() : null;
        if (this.multiDiscreteAgent != null) {
            this.multiDiscreteAgent.setRewardConfig(this.rewardConfig);
        }
    }

    /**
     * Fully resets derived runtime state (stats, best scores, grids, memory) 
     * but KEEPS the model weights and core configuration.
     */
    public void resetRuntimeState() {
        this.bestScore = -1;
        this.bestGridModel.clear();
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
        this.totalTrainingTimeMs = 0;
        
        // Re-create memory buffers to clear them (ExperienceReplay may not have clear())
        this.memory = new ExperienceReplay(MEMORY_CAPACITY);
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
        return useMultiDiscreteFlow ? (multiDiscreteMemory != null ? multiDiscreteMemory.size() : 0) 
                                   : (memory != null ? memory.size() : 0);
    }

    public int getBestBaseBlocks() { return bestBaseBlocks; }
    public boolean isBestBaseHasTC() { return bestBaseHasTC; }
    public int getBestBaseDoors() { return bestBaseDoors; }

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


    // ---- Training log ----
    
    public void setLogFile(String modelName) {
        logger.setLogFile(modelName, this.useMultiDiscreteFlow);
    }

    private void writeEpochLog(int epoch, int episodesInEpoch, double epochTotalReward,
                                int epochInvalid, int epochActions, int epochBlocks,
                                double epochBestEpScore, long epochMs) {
        logger.writeEpochLog(epoch, episodesInEpoch, epochTotalReward, epochInvalid, epochActions,
                epochBlocks, epochBestEpScore, epochMs,
                bestScore, epsilon, lastTrainLoss, bestBaseBlocks, bestBaseHasTC, bestBaseDoors);
    }

    private void writeInvalidActionLog(int step, int episode, String failReason,
                                        String actionType, int floor, int tileX, int tileY,
                                        int rotation, int aimSector,
                                        double minDist, double socketDist) {
        logger.writeInvalidActionLog(step, episode, failReason, actionType, floor,
                tileX, tileY, rotation, aimSector, minDist, socketDist);
    }

    private void writeEpisodeLog(int epoch, int epochEpisode, int totalEpisode, EpisodeResult result, double eps) {
        logger.writeEpisodeLog(epoch, epochEpisode, totalEpisode,
                result.totalActions, result.invalidActions, result.blocksPlaced,
                result.accStepReward, result.finalEvalReward, eps,
                result.grid.getAllBlocks());
    }
    
    public GridModel getBestGridModel() { return bestGridModel; }
    public GridModel getCurrentEpisodeGrid() { return currentGridModel; }
    public DQNAgent getAgent() { return agent; }
    public ExperienceReplay getMemory() { return memory; }
    
    public TrainingMetrics getMetrics() {
        double invalidRate = lastEpisodeTotalActions > 0 ? (double) lastEpisodeInvalidActions / lastEpisodeTotalActions : 0.0;
        int mSize = useMultiDiscreteFlow ? multiDiscreteMemory.size() : memory.size();
        return new TrainingMetrics(0, 0, 0, 0, episodesTrained, bestScore, epsilon, lastTrainLoss, avgReward, invalidRate, lastEpisodeInvalidActions, lastEpisodeTotalActions, bestBaseBlocks, bestBaseHasTC, bestBaseDoors, avgEvalScore, 0.0, 0.0, 0.0, mSize);
    }

    public void requestStop() {
        this.stopRequested = true;
    }

    public boolean isStopRequested() {
        return stopRequested;
    }

    public boolean isUseMultiDiscreteFlow() { return useMultiDiscreteFlow; }
    public void setUseMultiDiscreteFlow(boolean use) { this.useMultiDiscreteFlow = use; }
    
    public boolean isUseMultiDiscreteLearning() {
        return useMultiDiscreteLearning;
    }

    public void setUseMultiDiscreteLearning(boolean useMultiDiscreteLearning) {
        this.useMultiDiscreteLearning = useMultiDiscreteLearning;
        if (useMultiDiscreteLearning) {
            this.multiDiscretePolicy = new ProvidedPhaseMultiDiscretePolicy(this.multiDiscreteNeuralProvider);
        } else {
            this.multiDiscretePolicy = new com.rustbuilder.ai.rl.multidiscrete.HeuristicMultiDiscretePhasePolicy(this.random);
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
    
    public QTableMultiDiscreteDecisionProvider getMultiDiscreteLearningProvider() { return multiDiscreteLearningProvider; }
    
    /**
     * Updates the Q-Table baseline provider (artifact) without switching the active policy.
     */
    public void updateMultiDiscreteQTableBaseline(QTableMultiDiscreteDecisionProvider provider) {
        this.multiDiscreteLearningProvider = provider;
    }

    /**
     * Explicitly switches the active multi-discrete policy to use the provided decision provider.
     */
    public void setMultiDiscreteDecisionProvider(MultiDiscretePhaseDecisionProvider provider) {
        this.multiDiscretePolicy = new ProvidedPhaseMultiDiscretePolicy(provider);
    }

    /**
     * Check if two blocks are structurally connected (socket proximity or shared foundation for furniture).
     */
    private boolean areBlocksConnected(BuildingBlock a, BuildingBlock b) {
        // Quick distance reject
        if (Math.abs(a.getX() - b.getX()) > 200 || Math.abs(a.getY() - b.getY()) > 200) return false;
        
        // Furniture connects to the foundation it sits on (same x,y position)
        boolean aFurn = com.rustbuilder.util.BuildingTypeUtils.isFurniture(a.getType());
        boolean bFurn = com.rustbuilder.util.BuildingTypeUtils.isFurniture(b.getType());
        boolean aBase = com.rustbuilder.util.BuildingTypeUtils.isFoundation(a.getType()) || com.rustbuilder.util.BuildingTypeUtils.isFloor(a.getType());
        boolean bBase = com.rustbuilder.util.BuildingTypeUtils.isFoundation(b.getType()) || com.rustbuilder.util.BuildingTypeUtils.isFloor(b.getType());
        
        if ((aFurn && bBase) || (bFurn && aBase)) {
            if (a.getZ() == b.getZ() && Math.abs(a.getX() - b.getX()) < 1.0 && Math.abs(a.getY() - b.getY()) < 1.0) {
                return true;
            }
        }
        
        // Socket-based connection
        for (com.rustbuilder.model.core.Socket s1 : a.getSockets()) {
            for (com.rustbuilder.model.core.Socket s2 : b.getSockets()) {
                double dx = s1.getX() - s2.getX();
                double dy = s1.getY() - s2.getY();
                if (dx * dx + dy * dy < 1.3) return true;
            }
        }
        return false;
    }

    private boolean isFoundation(BuildingBlock b) {
        if (b == null) return false;
        return b.getType() == BuildingType.FOUNDATION || b.getType() == BuildingType.TRIANGLE_FOUNDATION;
    }

    private void finalizeLegacyTraining(EpisodeResult result) {
        if (result == null || result.grid == null || memory == null || agent == null) {
            return;
        }

        INDArray terminalState = StateEncoder.encodeWithPhase(result.grid, -1);

        memory.add(new ExperienceReplay.Transition(
            terminalState,
            ActionSpace.STOP_TYPE_INDEX,
            result.finalEvalReward,
            terminalState,
            true,
            java.util.Collections.emptyList()
        ));

        if (memory.size() >= batchSize) {
            lastTrainLoss = agent.trainBatch(memory.sample(batchSize));
        }
    }
}
