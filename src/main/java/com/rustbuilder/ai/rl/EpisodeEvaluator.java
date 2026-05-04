package com.rustbuilder.ai.rl;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
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
        GridModel grid = result.grid;
        int missedSteps = Math.max(0, maxStepsPerEpisode - result.totalActions);
        double earlyStopPenalty = 0.0;
        if (result.stopReason == StopReason.AGENT_STOP && missedSteps > 0) {
            earlyStopPenalty = -Math.pow(Math.abs(missedSteps * rewardConfig.earlyStopPenaltyMult), 1.2);
        }
        result.earlyStopPenalty = earlyStopPenalty;

        if (grid.getAllBlocks().size() > 5) {
            result.evaluationResult = evaluator.evaluate(grid);
            
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
            
            double connectivityBonus = (components.size() == 1) ? rewardConfig.connectivityBonus : 0.0;
            double fragmentPenalty = 0.0;
            if (components.size() > 1) {
                int mainIdx = 0;
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
            
            double rawScore = result.evaluationResult.finalScore * rewardConfig.finalScoreMultiplier;
            double logisticsBonus = rewardConfig.logisticsBonus * result.evaluationResult.logistics.score;
            double raidBonus = (result.evaluationResult.raid.sulfurToTC > 0) ? (result.evaluationResult.raid.score * rewardConfig.raidBonusMultiplier) : 0.0;
            double tcEnclosedBonus = (tcBlock != null && result.evaluationResult.raid.sulfurToTC > 0)
                    ? rewardConfig.tcEnclosedBonus
                    : 0.0;
            
            result.finalEvalReward = rawScore + logisticsBonus + raidBonus + connectivityBonus
                        + tcEnclosedBonus + earlyStopPenalty + fragmentPenalty + tcPenalty;
            
            // Distribute final reward backwards in neural memory (Multi-discrete)
            if (multiDiscreteMemory != null && multiDiscreteMemory.size() > 0) {
                // terminal reward shaping: distribute a portion of final Eval back to useful steps
                double shapedTailReward = (result.finalEvalReward - earlyStopPenalty) * 0.25;
                if (result.episodeTransitions != null && !result.episodeTransitions.isEmpty()) {
                    int distributeCount = Math.min(result.episodeTransitions.size(), 8);
                    double rewardPerStep = shapedTailReward / distributeCount;
                    int startIdx = result.episodeTransitions.size() - distributeCount;
                    for (int i = startIdx; i < result.episodeTransitions.size(); i++) {
                        result.episodeTransitions.get(i).reward += rewardPerStep;
                    }
                }
            }
        } else {
            result.finalEvalReward = rewardConfig.totalFailurePenalty + earlyStopPenalty;
        }
    }

    private boolean areBlocksConnected(BuildingBlock b1, BuildingBlock b2) {
        if (b1.getZ() != b2.getZ()) return false;
        double dx = b1.getX() - b2.getX();
        double dy = b1.getY() - b2.getY();
        double distSq = dx * dx + dy * dy;
        double threshold = GameConstants.TILE_SIZE * 1.5;
        return distSq <= threshold * threshold;
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
