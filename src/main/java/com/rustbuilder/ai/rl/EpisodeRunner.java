package com.rustbuilder.ai.rl;

import com.rustbuilder.core.action.BuildAction;
import com.rustbuilder.core.placement.PlacementError;
import com.rustbuilder.ai.rl.env.state.EncodedState;
import com.rustbuilder.ai.rl.env.state.StateRepresentationEncoder;
import com.rustbuilder.config.GameConstants;
import com.rustbuilder.ai.rl.log.StopReason;
import com.rustbuilder.ai.rl.multidiscrete.*;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.core.Socket;
import com.rustbuilder.service.evaluator.HouseEvaluator;
import com.rustbuilder.service.physics.PlacementService;
import com.rustbuilder.util.BuildingTypeUtils;
import com.rustbuilder.util.SocketCompatibilityUtils;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

public class EpisodeRunner {

    private static final Logger LOGGER = Logger.getLogger(EpisodeRunner.class.getName());
    private static final int TRAIN_START_MEMORY = 48;
    private static final int TRAIN_BATCH_SIZE = 32;
    private static final int TRAIN_EVERY_STEPS = 4;
    private static final int GROWTH_STREAK_BONUS_STEP_CAP = 15;
    private static final int SPATIAL_COMPONENT_SEARCH_THRESHOLD = 128;

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
        int consecutiveGrowthSteps = 0;
        double previousStepEvalScore = 0.0;
        double remainingTcProtectionReward = Math.max(0.0, rewardConfig.tcProtectionEpisodeRewardCap);

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
            ComponentStats oldComponentStats = analyzeComponents(result.grid);
            double oldTcProtectionScore = calculateTcProtectionScore(result.grid);
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
                double growthStreakReward = calculateGrowthStreakReward(consecutiveGrowthSteps, rewardConfig);
                stepReward += growthStreakReward;
                result.stepRewardGrowthStreak += growthStreakReward;
                consecutiveNoGrowthSteps = 0;

                ComponentStats newComponentStats = analyzeComponents(result.grid);
                ComponentDeltaReward componentDeltaReward = calculateComponentDeltaReward(
                    oldComponentStats, newComponentStats, rewardConfig);
                stepReward += componentDeltaReward.total();
                result.stepRewardMainComponentDelta += componentDeltaReward.mainComponentReward;
                result.stepRewardFragmentationDelta -= componentDeltaReward.componentIncreasePenalty;

                double newTcProtectionScore = calculateTcProtectionScore(result.grid);
                double tcProtectionReward = calculateTcProtectionDeltaReward(
                    oldTcProtectionScore, newTcProtectionScore, rewardConfig, remainingTcProtectionReward);
                if (tcProtectionReward > 0.0) {
                    stepReward += tcProtectionReward;
                    result.stepRewardTcProtectionDelta += tcProtectionReward;
                    remainingTcProtectionReward -= tcProtectionReward;
                }

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

    static double calculateGrowthStreakReward(int consecutiveGrowthSteps, RLRewardConfig config) {
        if (consecutiveGrowthSteps <= 0 || config == null) {
            return 0.0;
        }
        int cappedStreak = Math.min(consecutiveGrowthSteps, GROWTH_STREAK_BONUS_STEP_CAP);
        return cappedStreak * Math.max(0.0, config.growthStreakBonus);
    }

    static ComponentDeltaReward calculateComponentDeltaReward(ComponentStats oldStats, ComponentStats newStats, RLRewardConfig config) {
        if (oldStats == null || newStats == null || config == null) {
            return ComponentDeltaReward.ZERO;
        }
        double mainComponentReward = Math.max(0, newStats.mainComponentBlocks - oldStats.mainComponentBlocks)
            * Math.max(0.0, config.mainComponentGrowthReward);
        double componentIncreasePenalty = Math.max(0, newStats.componentCount - oldStats.componentCount)
            * Math.max(0.0, config.componentCountIncreasePenalty);
        return new ComponentDeltaReward(mainComponentReward, componentIncreasePenalty);
    }

    static double calculateTcProtectionDeltaReward(double oldScore, double newScore, RLRewardConfig config, double remainingBudget) {
        if (config == null || remainingBudget <= 0.0) {
            return 0.0;
        }
        double scoreDelta = Math.max(0.0, newScore - oldScore);
        double rawReward = scoreDelta * Math.max(0.0, config.tcProtectionDeltaReward);
        return Math.min(rawReward, Math.max(0.0, remainingBudget));
    }

    static double calculateTcProtectionScore(GridModel grid) {
        if (grid == null) {
            return 0.0;
        }

        List<BuildingBlock> blocks = grid.getAllBlocks();
        BuildingBlock tc = null;
        for (BuildingBlock block : blocks) {
            if (block.getType() == BuildingType.TC) {
                tc = block;
                break;
            }
        }
        if (tc == null) {
            return 0.0;
        }

        int nearbyWallLike = 0;
        boolean hasRoof = false;
        boolean hasDoorLike = false;
        double nearRadiusSq = GameConstants.TILE_SIZE * GameConstants.TILE_SIZE * 2.25;

        for (BuildingBlock block : blocks) {
            if (block == tc) continue;

            double dx = block.getX() - tc.getX();
            double dy = block.getY() - tc.getY();
            double distSq = dx * dx + dy * dy;

            if (BuildingTypeUtils.isHorizontalSurface(block.getType())
                    && block.getZ() == tc.getZ() + 1
                    && Math.abs(dx) < 1.0
                    && Math.abs(dy) < 1.0) {
                hasRoof = true;
            }

            if (block.getZ() == tc.getZ() && distSq <= nearRadiusSq) {
                if (BuildingTypeUtils.isWall(block.getType())) {
                    nearbyWallLike++;
                    if (block.getType() == BuildingType.DOORWAY) {
                        hasDoorLike = true;
                    }
                } else if (block.getType() == BuildingType.DOOR) {
                    hasDoorLike = true;
                }
            }
        }

        double score = 0.0;
        score += Math.min(4, nearbyWallLike) * 0.5;
        if (hasRoof) {
            score += 1.0;
        }
        if (hasDoorLike) {
            score += 1.0;
        }
        return score;
    }

    static ComponentStats analyzeComponents(GridModel grid) {
        if (grid == null) {
            return ComponentStats.EMPTY;
        }
        List<BuildingBlock> allBlocks = grid.getAllBlocks();
        int blockCount = allBlocks.size();
        if (blockCount == 0) {
            return ComponentStats.EMPTY;
        }

        boolean useSpatialComponentSearch = blockCount > SPATIAL_COMPONENT_SEARCH_THRESHOLD;
        Map<BuildingBlock, Integer> blockIndex = null;
        if (useSpatialComponentSearch) {
            blockIndex = new IdentityHashMap<>(blockCount);
            for (int i = 0; i < blockCount; i++) {
                blockIndex.put(allBlocks.get(i), i);
            }
        }

        boolean[] visited = new boolean[blockCount];
        List<Integer> componentSizes = new ArrayList<>();
        for (int i = 0; i < blockCount; i++) {
            if (visited[i]) continue;
            int size = 0;
            java.util.ArrayDeque<Integer> queue = new java.util.ArrayDeque<>();
            queue.add(i);
            visited[i] = true;
            while (!queue.isEmpty()) {
                int cur = queue.poll();
                size++;
                BuildingBlock currentBlock = allBlocks.get(cur);
                if (useSpatialComponentSearch) {
                    List<BuildingBlock> neighbors = grid.getNearbyBlocks(
                        currentBlock.getX(), currentBlock.getY(), currentBlock.getZ(), GameConstants.TILE_SIZE * 1.5
                    );
                    for (BuildingBlock neighbor : neighbors) {
                        Integer neighborIndex = blockIndex.get(neighbor);
                        if (neighborIndex == null) continue;
                        int j = neighborIndex;
                        if (visited[j]) continue;
                        if (areBlocksConnected(currentBlock, neighbor)) {
                            visited[j] = true;
                            queue.add(j);
                        }
                    }
                } else {
                    for (int j = 0; j < blockCount; j++) {
                        if (visited[j]) continue;
                        if (!areBlocksConnected(currentBlock, allBlocks.get(j))) continue;
                        visited[j] = true;
                        queue.add(j);
                    }
                }
            }
            componentSizes.add(size);
        }

        int mainComponentBlocks = 0;
        for (int size : componentSizes) {
            if (size > mainComponentBlocks) {
                mainComponentBlocks = size;
            }
        }
        return new ComponentStats(componentSizes.size(), mainComponentBlocks);
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

    private static boolean areBlocksConnected(BuildingBlock b1, BuildingBlock b2) {
        if (b1 == null || b2 == null || b1 == b2) return false;
        if (Math.abs(b1.getX() - b2.getX()) > GameConstants.TILE_SIZE * 2.5 ||
            Math.abs(b1.getY() - b2.getY()) > GameConstants.TILE_SIZE * 2.5) {
            return false;
        }

        if (isFurnitureOnBase(b1, b2) || isFurnitureOnBase(b2, b1)) {
            return true;
        }
        if (isDoorInDoorway(b1, b2) || isDoorInDoorway(b2, b1)) {
            return true;
        }

        boolean allowCenterConnection = b1.getZ() == b2.getZ()
                && (BuildingTypeUtils.isFoundation(b1.getType()) || BuildingTypeUtils.isFoundation(b2.getType()));
        if (Math.abs(b1.getZ() - b2.getZ()) <= 1 && areSocketsConnected(b1, b2, allowCenterConnection)) {
            return true;
        }

        if (b1.getZ() != b2.getZ()) return false;
        double dx = b1.getX() - b2.getX();
        double dy = b1.getY() - b2.getY();
        double distSq = dx * dx + dy * dy;
        double threshold = GameConstants.TILE_SIZE * 1.5;
        return distSq <= threshold * threshold;
    }

    private static boolean isFurnitureOnBase(BuildingBlock furniture, BuildingBlock base) {
        return BuildingTypeUtils.isFurniture(furniture.getType())
                && BuildingTypeUtils.isHorizontalSurface(base.getType())
                && furniture.getZ() == base.getZ()
                && sameTilePosition(furniture, base);
    }

    private static boolean isDoorInDoorway(BuildingBlock door, BuildingBlock doorway) {
        return door.getType() == BuildingType.DOOR
                && doorway.getType() == BuildingType.DOORWAY
                && door.getZ() == doorway.getZ()
                && sameTilePosition(door, doorway);
    }

    private static boolean sameTilePosition(BuildingBlock a, BuildingBlock b) {
        return Math.abs(a.getX() - b.getX()) < 1.0
                && Math.abs(a.getY() - b.getY()) < 1.0;
    }

    private static boolean areSocketsConnected(BuildingBlock b1, BuildingBlock b2, boolean allowCenterConnection) {
        for (Socket s1 : b1.getSockets()) {
            for (Socket s2 : b2.getSockets()) {
                if (SocketCompatibilityUtils.areEdgeSocketsConnected(s1, s2, 1.3)) {
                    return true;
                }
                if (allowCenterConnection && s1.isCenter() && s2.isCenter()) {
                    double dx = s1.getX() - s2.getX();
                    double dy = s1.getY() - s2.getY();
                    if (dx * dx + dy * dy < 1.3) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    static final class ComponentStats {
        static final ComponentStats EMPTY = new ComponentStats(0, 0);

        final int componentCount;
        final int mainComponentBlocks;

        ComponentStats(int componentCount, int mainComponentBlocks) {
            this.componentCount = componentCount;
            this.mainComponentBlocks = mainComponentBlocks;
        }
    }

    static final class ComponentDeltaReward {
        static final ComponentDeltaReward ZERO = new ComponentDeltaReward(0.0, 0.0);

        final double mainComponentReward;
        final double componentIncreasePenalty;

        ComponentDeltaReward(double mainComponentReward, double componentIncreasePenalty) {
            this.mainComponentReward = mainComponentReward;
            this.componentIncreasePenalty = componentIncreasePenalty;
        }

        double total() {
            return mainComponentReward - componentIncreasePenalty;
        }
    }
}
