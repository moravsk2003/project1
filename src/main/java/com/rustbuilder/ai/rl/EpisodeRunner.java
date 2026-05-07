package com.rustbuilder.ai.rl;

import com.rustbuilder.core.action.BuildAction;
import com.rustbuilder.core.placement.PlacementError;
import com.rustbuilder.ai.rl.env.state.EncodedState;
import com.rustbuilder.ai.rl.env.state.StateRepresentationEncoder;
import com.rustbuilder.ai.rl.log.StopReason;
import com.rustbuilder.ai.rl.multidiscrete.*;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.service.evaluator.HouseEvaluator;
import com.rustbuilder.service.physics.PlacementService;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

public class EpisodeRunner {

    private static final Logger LOGGER = Logger.getLogger(EpisodeRunner.class.getName());
    private static final int TRAIN_START_MEMORY = 48;
    private static final int TRAIN_BATCH_SIZE = 32;
    private static final int TRAIN_EVERY_STEPS = 4;
    private static final double GROWTH_STREAK_BONUS_MAX = 3.0;

    private final MultiDiscreteDQNAgent multiDiscreteAgent;
    private final MultiDiscreteExperienceReplay multiDiscreteMemory;
    private final MultiDiscretePhasePolicy multiDiscretePolicy;
    private final MultiDiscreteStateObserver multiDiscreteObserver;
    private final Random random;
    private final RLRewardConfig rewardConfig;
    private final RLTrainingLogger logger;
    private final RLTrainingService rlService;
    private final StateRepresentationEncoder stateEncoder;
    private final HouseEvaluator evaluator;

    // Output state
    private double lastTrainLoss = 0;
    private MultiDiscreteAction currentMultiAction;

    public EpisodeRunner(MultiDiscreteDQNAgent multiDiscreteAgent,
                         MultiDiscreteExperienceReplay multiDiscreteMemory, MultiDiscretePhasePolicy multiDiscretePolicy,
                         MultiDiscreteStateObserver multiDiscreteObserver, Random random, RLRewardConfig rewardConfig,
                         RLTrainingLogger logger, RLTrainingService rlService,
                         StateRepresentationEncoder stateEncoder, HouseEvaluator evaluator) {
        this.multiDiscreteAgent = multiDiscreteAgent;
        this.multiDiscreteMemory = multiDiscreteMemory;
        this.multiDiscretePolicy = multiDiscretePolicy;
        this.multiDiscreteObserver = multiDiscreteObserver;
        this.random = random;
        this.rewardConfig = rewardConfig;
        this.logger = logger;
        this.rlService = rlService;
        this.stateEncoder = stateEncoder;
        this.evaluator = evaluator;
    }

    public double getLastTrainLoss() {
        return lastTrainLoss;
    }

    public MultiDiscreteAction getCurrentMultiAction() {
        return currentMultiAction;
    }

    public EpisodeResult runExperimentalMultiDiscreteEpisode(int maxStepsPerEpisode, int episodesTrained, int currentEpoch, int currentEpochEpisode) {
        long episodeStartNs = System.nanoTime();
        EpisodeResult result = new EpisodeResult();
        result.grid = new GridModel();
        int totalEpisode = episodesTrained + 1;

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
        double previousStepEvalScore = 0.0;

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
                    double stopEarlyPenalty = -Math.pow(Math.abs(missedSteps * rewardConfig.earlyStopPenaltyMult), 1.2);
                    double stopReward = stopEarlyPenalty;

                    int blocksPlaced = result.grid.getAllBlocks().size();
                    int unbuiltBlocks = maxStepsPerEpisode - blocksPlaced;
                    double stopUnbuiltPenalty = unbuiltBlocks * rewardConfig.stopUnbuiltBlockPenalty;
                    stopReward += stopUnbuiltPenalty;

                    double stopUnderbuildPenalty = 0.0;
                    if (blocksPlaced < 5) {
                        stopUnderbuildPenalty = rewardConfig.stopUnderbuildPenaltyHigh;
                    } else if (blocksPlaced < 8) {
                        stopUnderbuildPenalty = rewardConfig.stopUnderbuildPenaltyLow;
                    }
                    stopReward += stopUnderbuildPenalty;

                    double stopBeforeClamp = stopReward;
                    stopReward = Math.max(stopReward, rewardConfig.stopRewardClampMin);
                    result.stopTransitionEarlyPenalty += stopEarlyPenalty;
                    result.stopTransitionUnbuiltPenalty += stopUnbuiltPenalty;
                    result.stopTransitionUnderbuildPenalty += stopUnderbuildPenalty;
                    result.stopTransitionClampAdjustment += stopReward - stopBeforeClamp;
                    result.stopTransitionReward += stopReward;

                    com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay.Transition trans = new com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay.Transition(
                        stateEncoded, multiAction, stopReward, nextStateEncoded, true, step,
                        null, cloneGridForReplay(result), true,
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
                if (logger != null && logger.shouldWriteInvalidActionLog(totalEpisode, step)) {
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
                    boolean physicsAllowed = PlacementService.isActionActuallyFeasible(result.grid, legacyAction);
                    logger.writeInvalidActionLog(
                        totalEpisode,
                        currentEpoch,
                        currentEpochEpisode,
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
            StepRewardFunction.Breakdown stepBreakdown = StepRewardFunction.calculateBreakdown(
                result.grid, legacyAction, pResult.inserted, pResult.survived, pResult.placedBlock, pResult.error, rewardConfig);
            double stepReward = stepBreakdown.total();
            result.addStepRewardBreakdown(stepBreakdown);
            double[] headRewardMultipliers = MultiDiscreteCreditAssignment.forPlacement(pResult.survived, pResult.error);

            int afterBlockCount = result.grid.getAllBlocks().size();
            int growth = afterBlockCount - beforeBlockCount;
            if (growth > 0) {
                double growthReward = growth * rewardConfig.blockGrowthReward;
                stepReward += growthReward;
                result.stepRewardGrowth += growthReward;
                consecutiveGrowthSteps++;
                double growthStreakReward = Math.min(
                    consecutiveGrowthSteps * rewardConfig.growthStreakBonus,
                    GROWTH_STREAK_BONUS_MAX
                );
                stepReward += growthStreakReward;
                result.stepRewardGrowthStreak += growthStreakReward;
                consecutiveNoGrowthSteps = 0;

                double currentStepEvalScore = calculateStepEvalScore(result.grid, previousStepEvalScore);
                double stepEvalDelta = currentStepEvalScore - previousStepEvalScore;
                double stepEvalDeltaReward = stepEvalDelta * rewardConfig.stepEvalDeltaMultiplier;
                stepReward += stepEvalDeltaReward;
                result.stepRewardEvalDelta += stepEvalDeltaReward;
                previousStepEvalScore = currentStepEvalScore;
            } else {
                consecutiveGrowthSteps = 0;
                consecutiveNoGrowthSteps++;
                stepReward += rewardConfig.noGrowthPenalty;
                result.stepRewardNoGrowthPenalty += rewardConfig.noGrowthPenalty;
            }

            result.accStepReward += stepReward;
            result.perfRewardNs += System.nanoTime() - rewardStartNs;

            // Determine if this is a terminal step (max steps or stagnation)
            boolean terminatedByInvalidStreak = consecutiveInvalidSteps >= 5;
            boolean isTerminal = (step == maxStepsPerEpisode - 1)
                || terminatedByInvalidStreak
                || (consecutiveNoGrowthSteps >= 30 && afterBlockCount > 0);

            // Apply harsh penalty for invalid-streak termination so agent prefers STOP over spamming
            if (terminatedByInvalidStreak) {
                double invalidStreakPenalty = -1.2;
                stepReward += invalidStreakPenalty;
                result.accStepReward += invalidStreakPenalty;
                result.stepRewardInvalidStreakPenalty += invalidStreakPenalty;
            }

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
                    null, cloneGridForReplay(result), pResult.survived,
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
                    else if (terminatedByInvalidStreak) result.stopReason = StopReason.NO_VALID_ACTIONS;
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

    private double calculateStepEvalScore(GridModel grid, double fallbackScore) {
        if (evaluator == null || rewardConfig.stepEvalDeltaMultiplier == 0.0) {
            return fallbackScore;
        }
        try {
            return evaluator.evaluate(grid).finalScore;
        } catch (Exception ex) {
            LOGGER.log(Level.FINE, "Step evaluation failed; keeping previous score.", ex);
            return fallbackScore;
        }
    }

    private boolean shouldTrainAtStep(int step) {
        return multiDiscreteMemory != null
            && multiDiscreteMemory.size() >= TRAIN_START_MEMORY
            && step % TRAIN_EVERY_STEPS == 0;
    }

    private GridModel cloneGridForReplay(EpisodeResult result) {
        long cloneStartNs = System.nanoTime();
        try {
            return result.grid.clone();
        } finally {
            result.perfGridCloneNs += System.nanoTime() - cloneStartNs;
        }
    }
}
