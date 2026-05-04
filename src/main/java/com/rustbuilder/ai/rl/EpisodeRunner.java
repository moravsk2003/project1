package com.rustbuilder.ai.rl;

import com.rustbuilder.ai.ea.BaseGenome.BuildAction;
import com.rustbuilder.ai.rl.env.state.EncodedState;
import com.rustbuilder.ai.rl.env.state.StateRepresentationEncoder;
import com.rustbuilder.ai.rl.log.StopReason;
import com.rustbuilder.ai.rl.multidiscrete.*;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import java.util.List;
import java.util.Random;

public class EpisodeRunner {

    private static final int TRAIN_START_MEMORY = 48;
    private static final int TRAIN_BATCH_SIZE = 32;
    private static final int TRAIN_EVERY_STEPS = 4;

    private final MultiDiscreteDQNAgent multiDiscreteAgent;
    private final MultiDiscreteExperienceReplay multiDiscreteMemory;
    private final MultiDiscretePhasePolicy multiDiscretePolicy;
    private final MultiDiscreteStateObserver multiDiscreteObserver;
    private final Random random;
    private final RLRewardConfig rewardConfig;
    private final RLTrainingLogger logger;
    private final RLTrainingService rlService;
    private final StateRepresentationEncoder stateEncoder;

    // Output state
    private double lastTrainLoss = 0;
    private MultiDiscreteAction currentMultiAction;

    public EpisodeRunner(MultiDiscreteDQNAgent multiDiscreteAgent,
                         MultiDiscreteExperienceReplay multiDiscreteMemory, MultiDiscretePhasePolicy multiDiscretePolicy,
                         MultiDiscreteStateObserver multiDiscreteObserver, Random random, RLRewardConfig rewardConfig,
                         RLTrainingLogger logger, RLTrainingService rlService,
                         StateRepresentationEncoder stateEncoder) {
        this.multiDiscreteAgent = multiDiscreteAgent;
        this.multiDiscreteMemory = multiDiscreteMemory;
        this.multiDiscretePolicy = multiDiscretePolicy;
        this.multiDiscreteObserver = multiDiscreteObserver;
        this.random = random;
        this.rewardConfig = rewardConfig;
        this.logger = logger;
        this.rlService = rlService;
        this.stateEncoder = stateEncoder;
    }

    public double getLastTrainLoss() {
        return lastTrainLoss;
    }

    public MultiDiscreteAction getCurrentMultiAction() {
        return currentMultiAction;
    }

    public EpisodeResult runExperimentalMultiDiscreteEpisode(int maxStepsPerEpisode, int episodesTrained) {
        long episodeStartNs = System.nanoTime();
        EpisodeResult result = new EpisodeResult();
        result.grid = new GridModel();

        // Reset masking counters for the new episode
        HeuristicMaskingUtils.prunedTypeCount = 0;
        HeuristicMaskingUtils.emptyTilesCount = 0;
        HeuristicMaskingUtils.emptyRotationsCount = 0;
        HeuristicMaskingUtils.rotationsRejectedByAimCount = 0;
        HeuristicMaskingUtils.tilesRejectedByNearStructureRule = 0;

        long episodeEncoderTime = 0;
        long episodeGlobalEncoderTime = 0;

        int consecutiveInvalidSteps = 0;
        int consecutiveNoGrowthSteps = 0;
        int lastBlockCount = 0;
        int consecutiveGrowthSteps = 0;

        try {
            for (int step = 0; step < maxStepsPerEpisode; step++) {
                result.totalActions++;

                long contextStartNs = System.nanoTime();
                MultiDiscretePhaseContext context = new MultiDiscretePhaseContext(
                    result.grid, result.hasTC, result.hasLootRoom, step, maxStepsPerEpisode
                );
                result.perfContextNs += System.nanoTime() - contextStartNs;

            // Encode state BEFORE grid is mutated
            long encStartNs = System.nanoTime();
            long encStart = System.currentTimeMillis();
            EncodedState stateEncoded = stateEncoder.encode(result.grid, -1);
            long encEnd = System.currentTimeMillis();
            result.perfStateEncodeNs += System.nanoTime() - encStartNs;
            episodeEncoderTime += (encEnd - encStart);
            if (stateEncoded != null && stateEncoded.getGlobalVector() != null) {
                // Global vector encoding time is part of total encoder time for now
            }

            GridModel gridBeforeAction = null;
            if (multiDiscreteMemory != null) {
                long cloneStartNs = System.nanoTime();
                gridBeforeAction = result.grid.clone();
                result.perfGridCloneNs += System.nanoTime() - cloneStartNs;
            }

            long actionStartNs = System.nanoTime();
            MultiDiscreteAction multiAction = multiDiscretePolicy.chooseAction(context, this.multiDiscreteObserver);
            result.perfActionSelectNs += System.nanoTime() - actionStartNs;
            this.currentMultiAction = multiAction;

            if (multiAction == null || multiAction.getTypeIndex() == MultiDiscreteActionSpace.STOP_TYPE_INDEX) {
                result.stopReason = (multiAction == null) ? StopReason.MASK_EMPTY : StopReason.AGENT_STOP;

                if (multiAction == null) {
                    if (stateEncoded != null && !stateEncoded.wasClosed()) {
                        stateEncoded.close();
                    }
                    break;
                }

                if (multiDiscreteMemory != null && stateEncoded != null) {
                    long stopEncodeStartNs = System.nanoTime();
                    long encStartStop = System.currentTimeMillis();
                    EncodedState nextStateEncoded = stateEncoder.encode(result.grid, -1);
                    long encEndStop = System.currentTimeMillis();
                    result.perfNextStateEncodeNs += System.nanoTime() - stopEncodeStartNs;
                    episodeEncoderTime += (encEndStop - encStartStop);

                    if (nextStateEncoded != null && nextStateEncoded.getDiagnostics() != null) {
                        result.globalDiag = nextStateEncoded.getDiagnostics().globalDiagnostics;
                    }

                    int missedSteps = Math.max(0, maxStepsPerEpisode - result.totalActions);
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
                        stateEncoded, multiAction, stopReward, nextStateEncoded, true, step,
                        gridBeforeAction, result.grid.clone(), true,
                        MultiDiscreteCreditAssignment.forPlacement(true, PlacementError.NONE)
                    );
                    long replayStartNs = System.nanoTime();
                    multiDiscreteMemory.add(trans);
                    result.perfReplayNs += System.nanoTime() - replayStartNs;

                    if (shouldTrainAtStep(step)) {
                        long trainStartNs = System.nanoTime();
                        List<com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay.Transition> batch = multiDiscreteMemory.sample(TRAIN_BATCH_SIZE);
                        lastTrainLoss = multiDiscreteAgent.trainBatch(batch);
                        result.perfTrainNs += System.nanoTime() - trainStartNs;
                    }
                }
                break;
            }

            long placementStartNs = System.nanoTime();
            BuildAction legacyAction = MultiDiscreteActionMapper.toBuildAction(multiAction);
            int beforeBlockCount = result.grid.getAllBlocks().size();
            RLTrainingService.PlacementResult pResult = rlService.placeBlock(result.grid, legacyAction);
            result.perfPlacementNs += System.nanoTime() - placementStartNs;

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

                // Sample invalid action details to keep long training runs lightweight.
                if (logger != null && logger.shouldWriteInvalidActionLog(episodesTrained, step)) {
                    long invalidLogStartNs = System.nanoTime();
                    boolean maskAllowed = HeuristicMaskingUtils
                        .getValidAimSectors(
                            result.grid,
                            multiAction.getTypeIndex(),
                            multiAction.getFloorIndex(),
                            multiAction.getTileIndex(),
                            multiAction.getRotationIndex()
                        )
                        .contains(multiAction.getAimSector());
                    boolean physicsAllowed = com.rustbuilder.util.GridPlacementUtils.isActionActuallyFeasible(result.grid, legacyAction);
                    logger.writeInvalidActionLog(
                        0,
                        episodesTrained,
                        step,
                        pResult.error.name(),
                        legacyAction.actionType.name(), multiAction.getFloorIndex(),
                        multiAction.getTileX(), multiAction.getTileY(), multiAction.getRotationIndex(),
                        multiAction.getAimSector(), pResult.minDist, pResult.socketDist,
                        maskAllowed, physicsAllowed);
                    result.perfInvalidLogNs += System.nanoTime() - invalidLogStartNs;
                }
            }

            long finalizeStartNs = System.nanoTime();
            result.grid.finalizeLoad();
            result.perfFinalizeNs += System.nanoTime() - finalizeStartNs;

            // Move reward calculation here so pResult.survived is correctly set by physics
            pResult.survived = (pResult.inserted && result.grid.getAllBlocks().contains(pResult.placedBlock));
            if (pResult.inserted && !pResult.survived) {
                pResult.error = PlacementError.NO_SUPPORT;
                pResult.failReason = "removed_by_stability";
                result.invalidActions++;
                consecutiveInvalidSteps++;
                result.blocksPlaced = Math.max(0, result.blocksPlaced - 1);
                result.errorStats.merge(pResult.error, 1, Integer::sum);
                result.typeStats.merge(legacyAction.actionType, 1, Integer::sum);
            }
            long rewardStartNs = System.nanoTime();
            double stepReward = StepRewardFunction.calculate(result.grid, legacyAction, pResult.inserted, pResult.survived, pResult.placedBlock, pResult.error, rewardConfig);
            double[] headRewardMultipliers = MultiDiscreteCreditAssignment.forPlacement(pResult.survived, pResult.error);

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
            result.perfRewardNs += System.nanoTime() - rewardStartNs;

            // Determine if this is a terminal step (max steps or stagnation)
            boolean isTerminal = (step == maxStepsPerEpisode - 1)
                || (consecutiveInvalidSteps >= 15)
                || (consecutiveNoGrowthSteps >= 30 && afterBlockCount > 0);

            // Neural learning update for multi-discrete path
            if (multiDiscreteMemory != null && stateEncoded != null) {
                // Encode nextState AFTER grid mutation
                long nextEncodeStartNs = System.nanoTime();
                long encStartNext = System.currentTimeMillis();
                EncodedState nextStateEncoded = stateEncoder.encode(result.grid, -1);
                long encEndNext = System.currentTimeMillis();
                result.perfNextStateEncodeNs += System.nanoTime() - nextEncodeStartNs;
                episodeEncoderTime += (encEndNext - encStartNext);

                com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay.Transition trans = new com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay.Transition(
                    stateEncoded, multiAction, stepReward, nextStateEncoded, isTerminal, step,
                    gridBeforeAction, result.grid.clone(), pResult.survived,
                    headRewardMultipliers
                );

                // Track global diagnostics from the last encoding
                result.globalDiag = (nextStateEncoded.getDiagnostics() != null) ? nextStateEncoded.getDiagnostics().globalDiagnostics : null;

                long replayStartNs = System.nanoTime();
                multiDiscreteMemory.add(trans);
                if (pResult.survived) {
                    result.episodeTransitions.add(trans);
                }
                result.perfReplayNs += System.nanoTime() - replayStartNs;

                // Start training once enough experience is collected
                if (shouldTrainAtStep(step)) {
                    long trainStartNs = System.nanoTime();
                    List<com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay.Transition> batch = multiDiscreteMemory.sample(TRAIN_BATCH_SIZE);
                    lastTrainLoss = multiDiscreteAgent.trainBatch(batch);
                    result.perfTrainNs += System.nanoTime() - trainStartNs;
                }
            }

            // Early stop on invalid streak or growth stall to save training time
            if (isTerminal) {
                if (result.stopReason == StopReason.UNKNOWN) {
                    if (step == maxStepsPerEpisode - 1) result.stopReason = StopReason.MAX_STEPS;
                    else if (consecutiveInvalidSteps >= 15) result.stopReason = StopReason.NO_VALID_ACTIONS;
                    else result.stopReason = StopReason.STAGNATION; // Fallback for no-growth stall
                }
                break;
            }
        }
        } finally {
            result.perfEpisodeNs = System.nanoTime() - episodeStartNs;
        }

        result.totalEncoderTimeMs = episodeEncoderTime;
        result.totalGlobalEncoderTimeMs = episodeGlobalEncoderTime;

        result.blocksPlaced = result.grid.getAllBlocks().size();

        return result;
    }

    private boolean isFoundation(BuildingBlock block) {
        return block.getType() == BuildingType.FOUNDATION || block.getType() == BuildingType.TRIANGLE_FOUNDATION;
    }

    private boolean shouldTrainAtStep(int step) {
        return multiDiscreteMemory != null
            && multiDiscreteMemory.size() >= TRAIN_START_MEMORY
            && step % TRAIN_EVERY_STEPS == 0;
    }
}
