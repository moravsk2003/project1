package com.rustbuilder.ai.rl;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.service.evaluator.HouseEvaluator;
import com.rustbuilder.config.GameConstants;
import java.util.ArrayList;
import java.util.List;

public class EpisodeEvaluator {

    private final HouseEvaluator evaluator;
    private final RLRewardConfig rewardConfig;
    private final boolean useMultiDiscreteFlow;

    public EpisodeEvaluator(HouseEvaluator evaluator, RLRewardConfig rewardConfig, boolean useMultiDiscreteFlow) {
        this.evaluator = evaluator;
        this.rewardConfig = rewardConfig;
        this.useMultiDiscreteFlow = useMultiDiscreteFlow;
    }

    public void evaluate(EpisodeResult result, int maxStepsPerEpisode, int epoch, int ep, int totalEpisodes, boolean useMultiDiscreteLearning, com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay multiDiscreteMemory) {
        GridModel grid = result.grid;
        int missedSteps = maxStepsPerEpisode - result.totalActions;
        double earlyStopPenalty = useMultiDiscreteFlow ? 0.0 : -Math.pow(Math.abs(missedSteps * rewardConfig.earlyStopPenaltyMult), 1.2);

        if (grid.getAllBlocks().size() > 5) {
            result.evaluationResult = evaluator.evaluate(grid);
            
            List<BuildingBlock> allBlocks = grid.getAllBlocks();
            int blockCount = allBlocks.size();
            
            // Structural connectivity analysis (DFS-based components)
            boolean[] visited = new boolean[blockCount];
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
                    for (int j = 0; j < blockCount; j++) {
                        if (visited[j]) continue;
                        if (areBlocksConnected(allBlocks.get(cur), allBlocks.get(j))) {
                            visited[j] = true;
                            queue.add(j);
                        }
                    }
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
                    double minDist = Double.MAX_VALUE;
                    for (int fi : components.get(c)) {
                        for (int mi : components.get(mainIdx)) {
                            double d = Math.hypot(
                                allBlocks.get(fi).getX() - allBlocks.get(mi).getX(),
                                allBlocks.get(fi).getY() - allBlocks.get(mi).getY());
                            if (d < minDist) minDist = d;
                        }
                    }
                    double minDistTiles = minDist / GameConstants.TILE_SIZE;
                    fragmentPenalty += rewardConfig.fragmentBasePenalty + (minDistTiles * minDistTiles) * rewardConfig.fragmentDistPenaltyMult;
                }
            }
            
            double tcPenalty = 0.0;
            BuildingBlock tcBlock = null;
            for (BuildingBlock b : allBlocks) {
                if (b.getType() == com.rustbuilder.model.core.BuildingType.TC) { tcBlock = b; break; }
            }
            if (tcBlock != null) {
                int tcCompIdx = -1;
                for (int c = 0; c < components.size(); c++) {
                    for (int idx : components.get(c)) {
                        if (allBlocks.get(idx) == tcBlock) { tcCompIdx = c; break; }
                    }
                    if (tcCompIdx >= 0) break;
                }
                for (int i = 0; i < blockCount; i++) {
                    BuildingBlock b = allBlocks.get(i);
                    if (b == tcBlock) continue;
                    int bComp = -1;
                    for (int c = 0; c < components.size(); c++) {
                        if (components.get(c).contains(i)) { bComp = c; break; }
                    }
                    if (bComp != tcCompIdx) tcPenalty += rewardConfig.tcConnectivityPenalty;
                    double distToTC = Math.hypot(b.getX() - tcBlock.getX(), b.getY() - tcBlock.getY()) / GameConstants.TILE_SIZE;
                    tcPenalty += distToTC * rewardConfig.tcDistancePenaltyMult;
                }
            }
            
            double rawScore = result.evaluationResult.finalScore * rewardConfig.finalScoreMultiplier;
            double logisticsBonus = (result.evaluationResult.logistics.score > 0) ? rewardConfig.logisticsBonus : 0.0;
            double raidBonus = (result.evaluationResult.raid.sulfurToTC > 0) ? (result.evaluationResult.raid.score * rewardConfig.raidBonusMultiplier) : 0.0;
            
            result.finalEvalReward = rawScore + logisticsBonus + raidBonus + connectivityBonus
                        + earlyStopPenalty + fragmentPenalty + tcPenalty;
            
            // Distribute final reward backwards in neural memory (Multi-discrete)
            if (useMultiDiscreteLearning && multiDiscreteMemory != null && multiDiscreteMemory.size() > 0) {
                // terminal reward shaping: distribute a portion of final Eval back to useful steps
                double shapedTailReward = result.finalEvalReward * 0.25;
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
        double dist = Math.hypot(b1.getX() - b2.getX(), b1.getY() - b2.getY());
        return dist <= GameConstants.TILE_SIZE * 1.5;
    }
}
