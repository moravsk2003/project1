package com.rustbuilder.ai.rl;

import com.rustbuilder.ai.ea.BaseGenome.BuildAction;
import com.rustbuilder.ai.rl.legacy.ActionSpace;
import com.rustbuilder.ai.rl.legacy.StateEncoder;
import com.rustbuilder.ai.rl.multidiscrete.*;
import com.rustbuilder.ai.rl.nn.DQNAgent;
import com.rustbuilder.ai.rl.nn.ExperienceReplay;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.List;
import java.util.Random;

public class EpisodeRunner {

    private final DQNAgent agent;
    private final ExperienceReplay memory;
    private final MultiDiscreteDQNAgent multiDiscreteAgent;
    private final MultiDiscreteExperienceReplay multiDiscreteMemory;
    private final MultiDiscretePhasePolicy multiDiscretePolicy;
    private final MultiDiscreteStateObserver multiDiscreteObserver;
    private final Random random;
    private final RLRewardConfig rewardConfig;
    private final boolean useMultiDiscreteLearning;
    private final RLTrainingLogger logger;
    private final RLTrainingService rlService;
    
    // Output state
    private double lastTrainLoss = 0;
    private MultiDiscreteAction currentMultiAction;

    public EpisodeRunner(DQNAgent agent, ExperienceReplay memory, MultiDiscreteDQNAgent multiDiscreteAgent,
                         MultiDiscreteExperienceReplay multiDiscreteMemory, MultiDiscretePhasePolicy multiDiscretePolicy,
                         MultiDiscreteStateObserver multiDiscreteObserver, Random random, RLRewardConfig rewardConfig,
                         boolean useMultiDiscreteLearning, RLTrainingLogger logger, RLTrainingService rlService) {
        this.agent = agent;
        this.memory = memory;
        this.multiDiscreteAgent = multiDiscreteAgent;
        this.multiDiscreteMemory = multiDiscreteMemory;
        this.multiDiscretePolicy = multiDiscretePolicy;
        this.multiDiscreteObserver = multiDiscreteObserver;
        this.random = random;
        this.rewardConfig = rewardConfig;
        this.useMultiDiscreteLearning = useMultiDiscreteLearning;
        this.logger = logger;
        this.rlService = rlService;
    }

    public double getLastTrainLoss() {
        return lastTrainLoss;
    }

    public MultiDiscreteAction getCurrentMultiAction() {
        return currentMultiAction;
    }

    public EpisodeResult runLegacyEpisode(int maxStepsPerEpisode, double epsilon, int batchSize) {
        EpisodeResult result = new EpisodeResult();
        result.grid = new GridModel();
        
        // Reset masking counters for the new episode
        HeuristicMaskingUtils.prunedTypeCount = 0;
        HeuristicMaskingUtils.emptyTilesCount = 0;
        HeuristicMaskingUtils.emptyRotationsCount = 0;
        HeuristicMaskingUtils.tilesRejectedByNearStructureRule = 0;
        
        for (int step = 0; step < maxStepsPerEpisode; step++) {
            result.totalActions++;
            double reward;
            
            // PHASE 1: Select block type
            INDArray statePhase1 = StateEncoder.encodeWithPhase(result.grid, -1);
            List<Integer> validTypes = ActionSpace.getValidTypeActions(result.hasTC, result.hasLootRoom, step);
            
            int typeIdx;
            if (random.nextDouble() < epsilon) {
                typeIdx = validTypes.get(random.nextInt(validTypes.size()));
            } else {
                double[] qVals = agent.getQValues(statePhase1);
                double maxQ = -Double.MAX_VALUE;
                int bestType = validTypes.get(0);
                for (int t : validTypes) {
                    if (qVals[t] > maxQ) {
                        maxQ = qVals[t];
                        bestType = t;
                    }
                }
                typeIdx = bestType;
            }
            
            if (typeIdx == ActionSpace.STOP_TYPE_INDEX) break;
            BuildAction.ActionType selectedType = ActionSpace.decodeType(typeIdx);
            if (selectedType == null) break;
            
            // PHASE 2: Select position
            INDArray statePhase2 = StateEncoder.encodeWithPhase(result.grid, selectedType.ordinal());
            List<Integer> validPositions = ActionSpace.getValidPositionActions(selectedType, step, result.grid);
            
            int posIdx;
            if (random.nextDouble() < epsilon) {
                posIdx = validPositions.get(random.nextInt(validPositions.size()));
            } else {
                double[] qVals = agent.getQValues(statePhase2);
                double maxQ = -Double.MAX_VALUE;
                int bestPos = validPositions.get(0);
                for (int p : validPositions) {
                    if (qVals[p] > maxQ) {
                        maxQ = qVals[p];
                        bestPos = p;
                    }
                }
                posIdx = bestPos;
            }
            
            BuildAction action = ActionSpace.decodePosition(posIdx, selectedType);
            this.currentMultiAction = MultiDiscreteActionMapper.toMultiDiscrete(action);
            
            RLTrainingService.PlacementResult pResult = rlService.placeBlock(result.grid, action);
            if (pResult.inserted) {
                result.blocksPlaced++;
                if (action.actionType == BuildAction.ActionType.TC) result.hasTC = true;
                if (action.actionType == BuildAction.ActionType.LOOT_ROOM) result.hasLootRoom = true;
            } else {
                result.invalidActions++;
            }
            
            result.grid.finalizeLoad();
            pResult.survived = (pResult.inserted && result.grid.getAllBlocks().contains(pResult.placedBlock));
            reward = StepRewardFunction.calculate(result.grid, action, pResult.inserted, pResult.survived, pResult.placedBlock, pResult.error, rewardConfig);
            
            // Transitions
            INDArray nextPhase1 = StateEncoder.encodeWithPhase(result.grid, -1);
            List<Integer> nextValidTypes = ActionSpace.getValidTypeActions(result.hasTC, result.hasLootRoom, step + 1);
            memory.add(new ExperienceReplay.Transition(statePhase1, typeIdx, 0.0, statePhase2, false, validPositions));
            memory.add(new ExperienceReplay.Transition(statePhase2, posIdx, reward, nextPhase1, false, nextValidTypes));
            
            if (memory.size() >= batchSize) {
                lastTrainLoss = agent.trainBatch(memory.sample(batchSize));
            }
            
            result.accStepReward += reward;
        }
        return result;
    }

    public EpisodeResult runExperimentalMultiDiscreteEpisode(int maxStepsPerEpisode, int episodesTrained) {
        EpisodeResult result = new EpisodeResult();
        result.grid = new GridModel();
        
        // Reset masking counters for the new episode
        HeuristicMaskingUtils.prunedTypeCount = 0;
        HeuristicMaskingUtils.emptyTilesCount = 0;
        HeuristicMaskingUtils.emptyRotationsCount = 0;
        HeuristicMaskingUtils.rotationsRejectedByAimCount = 0;
        HeuristicMaskingUtils.tilesRejectedByNearStructureRule = 0;
        
        int consecutiveInvalidSteps = 0;
        int consecutiveNoGrowthSteps = 0;
        int lastBlockCount = 0;
        int consecutiveGrowthSteps = 0;
        
        for (int step = 0; step < maxStepsPerEpisode; step++) {
            result.totalActions++;
            
            MultiDiscretePhaseContext context = new MultiDiscretePhaseContext(
                result.grid, result.hasTC, result.hasLootRoom, step, maxStepsPerEpisode
            );
            
            // Encode state BEFORE grid is mutated
            INDArray stateTensor = null;
            GridModel gridBeforeAction = null;
            if (useMultiDiscreteLearning && multiDiscreteMemory != null) {
                stateTensor = StateEncoder.encodeWithPhase(result.grid, -1);
                gridBeforeAction = result.grid.clone();
            }
            
            MultiDiscreteAction multiAction = multiDiscretePolicy.chooseAction(context, this.multiDiscreteObserver);
            this.currentMultiAction = multiAction;
            
            if (multiAction == null || multiAction.getTypeIndex() == ActionSpace.STOP_TYPE_INDEX) {
                if (useMultiDiscreteLearning && multiDiscreteMemory != null && stateTensor != null) {
                    INDArray nextStateTensor = StateEncoder.encodeWithPhase(result.grid, -1);
                    
                    int missedSteps = maxStepsPerEpisode - step;
                    double stopReward = -Math.pow(Math.abs(missedSteps * rewardConfig.earlyStopPenaltyMult), 1.2);
                    
                    int blocksPlaced = result.grid.getAllBlocks().size();
                    int unbuiltBlocks = maxStepsPerEpisode - blocksPlaced;
                    stopReward += unbuiltBlocks * rewardConfig.stopUnbuiltBlockPenalty;
                    
                    if (blocksPlaced < 5) {
                        stopReward += rewardConfig.stopUnderbuildPenaltyHigh;
                    } else if (blocksPlaced < 8) {
                        stopReward += rewardConfig.stopUnderbuildPenaltyLow;
                    }
                    
                    stopReward = Math.max(stopReward, rewardConfig.stopRewardClampMin);
                    
                    com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay.Transition trans = new com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay.Transition(
                        stateTensor, multiAction, stopReward, nextStateTensor, true, step,
                        gridBeforeAction, result.grid.clone(), true
                    );
                    multiDiscreteMemory.add(trans);
                    result.episodeTransitions.add(trans);
                    
                    if (multiDiscreteMemory.size() >= 48 && step % 2 == 0) {
                        List<com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay.Transition> batch = multiDiscreteMemory.sample(64);
                        lastTrainLoss = multiDiscreteAgent.trainBatch(batch);
                    }
                }
                break;
            }

            BuildAction legacyAction = MultiDiscreteActionMapper.toBuildAction(multiAction);
            int beforeBlockCount = result.grid.getAllBlocks().size();
            RLTrainingService.PlacementResult pResult = rlService.placeBlock(result.grid, legacyAction);

            if (pResult.inserted) {
                result.blocksPlaced++;
                
                if (legacyAction.actionType == BuildAction.ActionType.TC) {
                    result.hasTC = true;
                }
                if (legacyAction.actionType == BuildAction.ActionType.LOOT_ROOM) {
                    result.hasLootRoom = true;
                }
                
                consecutiveInvalidSteps = 0;
            } else {
                result.invalidActions++;
                consecutiveInvalidSteps++;
                
                // Collect analytics
                result.errorStats.merge(pResult.error, 1, Integer::sum);
                result.typeStats.merge(legacyAction.actionType, 1, Integer::sum);

                // Log invalid action reasons to CSV (unthrottled for debugging)
                if (logger != null) {
                    logger.writeInvalidActionLog(
                        step,
                        episodesTrained,
                        pResult.error.name(), 
                        legacyAction.actionType.name(), multiAction.getFloorIndex(),
                        multiAction.getTileX(), multiAction.getTileY(), multiAction.getRotationIndex(),
                        multiAction.getAimSector(), pResult.minDist, pResult.socketDist);
                }
            }
            
            result.grid.finalizeLoad();
            
            // Move reward calculation here so pResult.survived is correctly set by physics
            pResult.survived = (pResult.inserted && result.grid.getAllBlocks().contains(pResult.placedBlock));
            double stepReward = StepRewardFunction.calculate(result.grid, legacyAction, pResult.inserted, pResult.survived, pResult.placedBlock, pResult.error, rewardConfig);
            
            int afterBlockCount = result.grid.getAllBlocks().size();
            int growth = afterBlockCount - beforeBlockCount;
            if (growth > 0) {
                stepReward += growth * rewardConfig.blockGrowthReward;
                consecutiveGrowthSteps++;
                stepReward += consecutiveGrowthSteps * rewardConfig.growthStreakBonus;
                consecutiveNoGrowthSteps = 0;
            } else {
                consecutiveGrowthSteps = 0;
                consecutiveNoGrowthSteps++;
                stepReward += rewardConfig.noGrowthPenalty;
            }
            
            result.accStepReward += stepReward;
            
            // Determine if this is a terminal step (max steps or stagnation)
            boolean isTerminal = (step == maxStepsPerEpisode - 1)
                || (consecutiveInvalidSteps >= 15)
                || (consecutiveNoGrowthSteps >= 30 && afterBlockCount > 0);
            
            // Neural learning update for multi-discrete path
            if (useMultiDiscreteLearning && multiDiscreteMemory != null && stateTensor != null) {
                // Encode nextState AFTER grid mutation
                INDArray nextStateTensor = StateEncoder.encodeWithPhase(result.grid, -1);
                
                com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay.Transition trans = new com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay.Transition(
                    stateTensor, multiAction, stepReward, nextStateTensor, isTerminal, step,
                    gridBeforeAction, result.grid.clone(), pResult.survived
                );
                multiDiscreteMemory.add(trans);
                if (pResult.survived || pResult.inserted) {
                    result.episodeTransitions.add(trans);
                }
                
                // Start training once enough experience is collected
                if (multiDiscreteMemory.size() >= 48 && step % 2 == 0) {
                    List<com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay.Transition> batch = multiDiscreteMemory.sample(64);
                    lastTrainLoss = multiDiscreteAgent.trainBatch(batch);
                }
            }

            // Early stop on invalid streak or growth stall to save training time
            if (isTerminal && step < maxStepsPerEpisode - 1) {
                break;
            }
        }
        return result;
    }

    private boolean isFoundation(BuildingBlock block) {
        return block.getType() == BuildingType.FOUNDATION || block.getType() == BuildingType.TRIANGLE_FOUNDATION;
    }
}
