package com.rustbuilder.ai.rl;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.core.Socket;
import com.rustbuilder.util.BuildingTypeUtils;
import com.rustbuilder.util.SocketCompatibilityUtils;
import com.rustbuilder.ai.rl.log.StopReason;
import com.rustbuilder.service.evaluator.HouseEvaluator;
import com.rustbuilder.config.GameConstants;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

public class EpisodeEvaluator {
    private static final int SPATIAL_COMPONENT_SEARCH_THRESHOLD = 128;

    private final HouseEvaluator evaluator;
    private final RLRewardConfig rewardConfig;

    public EpisodeEvaluator(HouseEvaluator evaluator, RLRewardConfig rewardConfig) {
        this.evaluator = evaluator;
        this.rewardConfig = rewardConfig;
    }

    public void evaluate(EpisodeResult result, int maxStepsPerEpisode, int epoch, int ep, int totalEpisodes, com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay multiDiscreteMemory) {
        result.resetFinalRewardBreakdown();
        GridModel grid = result.grid;
        int missedSteps = Math.max(0, maxStepsPerEpisode - result.totalActions);
        double earlyStopPenalty = 0.0;
        if (result.stopReason == StopReason.AGENT_STOP && missedSteps > 0) {
            earlyStopPenalty = -Math.pow(Math.abs(missedSteps * rewardConfig.earlyStopPenaltyMult), 1.2);
        }
        result.earlyStopPenalty = earlyStopPenalty;

        if (grid.getAllBlocks().size() > 5) {
            result.evaluationResult = evaluator.evaluate(grid);
            result.evalLogisticsScore = result.evaluationResult.logistics.score;
            result.evalCostScore = result.evaluationResult.cost.score;
            result.evalRaidScore = result.evaluationResult.raid.score;
            result.evalWorkingAreaScore = result.evaluationResult.workingArea.score;
            result.evalSafeZoneScore = result.evaluationResult.safeZone.score;
            result.raidSulfurToTC = result.evaluationResult.raid.sulfurToTC;
            
            List<BuildingBlock> allBlocks = grid.getAllBlocks();
            int blockCount = allBlocks.size();
            boolean useSpatialComponentSearch = blockCount > SPATIAL_COMPONENT_SEARCH_THRESHOLD;
            Map<BuildingBlock, Integer> blockIndex = null;
            if (useSpatialComponentSearch) {
                blockIndex = new IdentityHashMap<>(blockCount);
                for (int i = 0; i < blockCount; i++) {
                    blockIndex.put(allBlocks.get(i), i);
                }
            }
            
            // Structural connectivity analysis (DFS-based components)
            boolean[] visited = new boolean[blockCount];
            int[] componentByBlock = new int[blockCount];
            List<List<Integer>> components = new ArrayList<>();
            
            for (int i = 0; i < blockCount; i++) {
                if (visited[i]) continue;
                List<Integer> comp = new ArrayList<>();
                java.util.ArrayDeque<Integer> queue = new java.util.ArrayDeque<>();
                queue.add(i);
                visited[i] = true;
                while (!queue.isEmpty()) {
                    int cur = queue.poll();
                    comp.add(cur);
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
                int componentIndex = components.size();
                for (int idx : comp) {
                    componentByBlock[idx] = componentIndex;
                }
                components.add(comp);
            }
            result.componentCount = components.size();
            
            double connectivityBonus = (components.size() == 1) ? rewardConfig.connectivityBonus : 0.0;
            double fragmentPenalty = 0.0;
            int mainIdx = 0;
            if (components.size() > 1) {
                for (int c = 1; c < components.size(); c++) {
                    if (components.get(c).size() > components.get(mainIdx).size()) mainIdx = c;
                }
                for (int c = 0; c < components.size(); c++) {
                    if (c == mainIdx) continue;
                    double minDistSq = Double.MAX_VALUE;
                    for (int fi : components.get(c)) {
                        for (int mi : components.get(mainIdx)) {
                            double dx = allBlocks.get(fi).getX() - allBlocks.get(mi).getX();
                            double dy = allBlocks.get(fi).getY() - allBlocks.get(mi).getY();
                            double dSq = dx * dx + dy * dy;
                            if (dSq < minDistSq) minDistSq = dSq;
                        }
                    }
                    double minDistTilesSq = minDistSq / (GameConstants.TILE_SIZE * GameConstants.TILE_SIZE);
                    fragmentPenalty += rewardConfig.fragmentBasePenalty + minDistTilesSq * rewardConfig.fragmentDistPenaltyMult;
                }
            }
            result.mainComponentBlocks = components.isEmpty() ? 0 : components.get(mainIdx).size();
            
            double tcPenalty = 0.0;
            BuildingBlock tcBlock = null;
            for (BuildingBlock b : allBlocks) {
                if (b.getType() == com.rustbuilder.model.core.BuildingType.TC) { tcBlock = b; break; }
            }
            if (tcBlock != null) {
                int tcIndex = indexOfIdentity(allBlocks, tcBlock);
                int tcCompIdx = tcIndex < 0 ? -1 : componentByBlock[tcIndex];
                for (int i = 0; i < blockCount; i++) {
                    BuildingBlock b = allBlocks.get(i);
                    if (b == tcBlock) continue;
                    int bComp = componentByBlock[i];
                    if (bComp != tcCompIdx) tcPenalty += rewardConfig.tcConnectivityPenalty;
                    double dx = b.getX() - tcBlock.getX();
                    double dy = b.getY() - tcBlock.getY();
                    double distToTC = Math.sqrt(dx * dx + dy * dy) / GameConstants.TILE_SIZE;
                    tcPenalty += distToTC * rewardConfig.tcDistancePenaltyMult;
                }
            }
            result.finalRewardHasTC = tcBlock != null;
            
            double rawScore = result.evaluationResult.finalScore * rewardConfig.finalScoreMultiplier;
            double logisticsBonus = rewardConfig.logisticsBonus * result.evaluationResult.logistics.score;
            double raidBonus = (result.evaluationResult.raid.sulfurToTC > 0) ? (result.evaluationResult.raid.score * rewardConfig.raidBonusMultiplier) : 0.0;
            double tcEnclosedBonus = (tcBlock != null && result.evaluationResult.raid.sulfurToTC > 0)
                    ? rewardConfig.tcEnclosedBonus
                    : 0.0;
            result.finalRewardRawScore = rawScore;
            result.finalRewardLogisticsBonus = logisticsBonus;
            result.finalRewardRaidBonus = raidBonus;
            result.finalRewardConnectivityBonus = connectivityBonus;
            result.finalRewardTcEnclosedBonus = tcEnclosedBonus;
            result.finalRewardFragmentPenalty = fragmentPenalty;
            result.finalRewardTcPenalty = tcPenalty;
            result.finalRewardTcEnclosed = tcEnclosedBonus > 0.0;
            
            result.finalEvalReward = rawScore + logisticsBonus + raidBonus + connectivityBonus
                        + tcEnclosedBonus + earlyStopPenalty + fragmentPenalty + tcPenalty;
            
            // Distribute final reward backwards in neural memory (Multi-discrete)
            if (multiDiscreteMemory != null && multiDiscreteMemory.size() > 0) {
                // terminal reward shaping: distribute a portion of final Eval back to useful steps
                double shapedTailReward = (result.finalEvalReward - earlyStopPenalty) * 0.25;
                if (result.episodeTransitions != null && !result.episodeTransitions.isEmpty()) {
                    // Класичний RL підхід: додаємо всю відкладену нагороду до останнього кроку (terminal state).
                    // Алгоритм DQN сам "протягне" її назад за допомогою параметра дисконтування (gamma).
                    result.episodeTransitions.get(result.episodeTransitions.size() - 1).reward += shapedTailReward;
                }
            }
        } else {
            result.finalEvalReward = rewardConfig.totalFailurePenalty + earlyStopPenalty;
            result.finalRewardFailurePenalty = rewardConfig.totalFailurePenalty;
        }
    }

    private boolean areBlocksConnected(BuildingBlock b1, BuildingBlock b2) {
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

    private boolean isFurnitureOnBase(BuildingBlock furniture, BuildingBlock base) {
        return BuildingTypeUtils.isFurniture(furniture.getType())
                && BuildingTypeUtils.isHorizontalSurface(base.getType())
                && furniture.getZ() == base.getZ()
                && sameTilePosition(furniture, base);
    }

    private boolean isDoorInDoorway(BuildingBlock door, BuildingBlock doorway) {
        return door.getType() == BuildingType.DOOR
                && doorway.getType() == BuildingType.DOORWAY
                && door.getZ() == doorway.getZ()
                && sameTilePosition(door, doorway);
    }

    private boolean sameTilePosition(BuildingBlock a, BuildingBlock b) {
        return Math.abs(a.getX() - b.getX()) < 1.0
                && Math.abs(a.getY() - b.getY()) < 1.0;
    }

    private boolean areSocketsConnected(BuildingBlock b1, BuildingBlock b2, boolean allowCenterConnection) {
        for (Socket s1 : b1.getSockets()) {
            for (Socket s2 : b2.getSockets()) {
                if (SocketCompatibilityUtils.areEdgeSocketsConnected(s1, s2, 1.3)) {
                    return true;
                }
                if (allowCenterConnection && s1.getSide() == 10 && s2.getSide() == 10) {
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

    private int indexOfIdentity(List<BuildingBlock> blocks, BuildingBlock target) {
        for (int i = 0; i < blocks.size(); i++) {
            if (blocks.get(i) == target) {
                return i;
            }
        }
        return -1;
    }
}
