package com.rustbuilder.ai.rl.domain;


import com.rustbuilder.ai.rl.application.EpisodeRunner;
import com.rustbuilder.ai.rl.domain.reward.RewardFormulaSet;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import com.rustbuilder.ai.rl.domain.reward.RewardFormulaScope;
import com.rustbuilder.core.action.BuildAction;
import com.rustbuilder.core.placement.PlacementError;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.Socket;
import com.rustbuilder.config.GameConstants;
import com.rustbuilder.service.evaluator.LogisticsEvaluator;
import com.rustbuilder.service.graph.HouseGraph;
import com.rustbuilder.service.graph.HouseGraph.TileNode;

/**
 * Calculates intermediate rewards for each RL step to guide the agent.
 * Rewards are kept small (mostly 0.0-0.5 range) to avoid Q-value explosion.
 */
public class StepRewardFunction {
    private static final int SPATIAL_REWARD_SEARCH_THRESHOLD = 96;

    public static class Breakdown {
        public double invalidPenalty = 0;
        public double basePlacement = 0;
        public double socketConnection = 0;
        public double disconnectedPenalty = 0;
        public double stabilityReward = 0;
        public double floatingPenalty = 0;
        public double typeBonus = 0;
        public double foundationBonus = 0;
        public double spatialCompactness = 0;
        public double spatialScatteredPenalty = 0;
        public double formulaReward = 0;

        public double total() {
            return invalidPenalty
                + basePlacement
                + socketConnection
                + disconnectedPenalty
                + stabilityReward
                + floatingPenalty
                + typeBonus
                + foundationBonus
                + spatialCompactness
                + spatialScatteredPenalty
                + formulaReward;
        }
    }

    public static double calculate(GridModel gridModel,
                                   BuildAction action,
                                   boolean inserted,
                                   boolean survived,
                                   BuildingBlock placed,
                                   PlacementError error) {
        return calculate(
            gridModel,
            action,
            inserted,
            survived,
            placed,
            error,
            RLRewardConfig.createDefault()
        );
    }

    public static double calculate(GridModel gridModel, BuildAction action, boolean inserted, boolean survived, 
                                 BuildingBlock placed, PlacementError error, RLRewardConfig config) {
        return calculateBreakdown(gridModel, action, inserted, survived, placed, error, config).total();
    }

    public static Breakdown calculateBreakdown(GridModel gridModel, BuildAction action, boolean inserted, boolean survived,
                                               BuildingBlock placed, PlacementError error, RLRewardConfig config) {
        Breakdown breakdown = new Breakdown();
        if (!inserted || error != PlacementError.NONE) {
            breakdown.invalidPenalty = switch (error) {
                case NO_SUPPORT -> config.penaltyNoSupport;
                case BAD_SOCKET_IS_FIRST, BAD_SOCKET_NO_TARGET, BAD_SOCKET_WRONG_TARGET_TYPE, BAD_SOCKET_NO_SOCKET_ALIGNMENT, BAD_SOCKET_CENTERDIST_REJECT -> config.penaltyBadSocket;
                case COLLISION -> config.penaltyCollision;
                case FLOOR_CONSTRAINT -> config.penaltyFloorConstraint;
                default -> config.penaltyGenericInvalid;
            };
            breakdown.formulaReward = evaluateStepFormulas(config, gridModel, action, inserted, survived, placed, error, 0);
            return breakdown;
        }
        if (!survived || placed == null) {
            breakdown.invalidPenalty = config.penaltyNoSupport;
            breakdown.formulaReward = evaluateStepFormulas(config, gridModel, action, inserted, survived, placed, error, 0);
            return breakdown;
        }

        breakdown.basePlacement = config.basePlacementReward;

        List<BuildingBlock> blocks = gridModel.getAllBlocks();
        if (blocks.isEmpty()) return breakdown;

        // ===== 1. Socket Connection Reward =====
        int socketConnections = countSocketConnections(gridModel, placed, blocks);
        boolean isFurniture = placed.getType() == BuildingType.TC || 
                              placed.getType() == BuildingType.WORKBENCH || 
                              placed.getType() == BuildingType.LOOT_ROOM;
        
        if (socketConnections > 0) {
            breakdown.socketConnection = Math.min(socketConnections * config.socketConnectionReward, config.socketConnectionMax);
        } else if (blocks.size() > 1 && !isFurniture) {
            breakdown.disconnectedPenalty = config.disconnectedSegmentPenalty;
        }

        // ===== 2. Structural Stability Reward =====
        // EpisodeRunner calls GridModel.finalizeLoad() immediately before reward calculation.
        double stability = placed.getStability();
        if (stability > 0) {
            breakdown.stabilityReward = stability * config.stabilityRewardMult;
        } else if (!isFoundation(placed)) {
            breakdown.floatingPenalty = config.floatingBlockPenalty;
        }

        // ===== 3. Type-specific bonuses =====
        if (action.actionType == BuildAction.ActionType.TC) {
            breakdown.typeBonus = config.tcPlacementBonus;
        } else if (action.actionType == BuildAction.ActionType.WORKBENCH) {
            if (countBlocksOfType(blocks, BuildingType.WORKBENCH) == 1 &&
                    countOpenDeployablesOfType(gridModel, "workbench") <= 1) {
                breakdown.typeBonus = config.secondaryDeployableBonus;
            }
        } else if (action.actionType == BuildAction.ActionType.LOOT_ROOM) {
            if (countOpenDeployablesOfType(gridModel, "loot_room") <= 1) {
                breakdown.typeBonus = config.secondaryDeployableBonus;
            }
        }

        // Foundation count bonus
        if (isFoundation(placed)) {
            int foundationCount = 0;
            for (BuildingBlock b : blocks) {
                if (isFoundation(b)) foundationCount++;
            }
            switch (foundationCount) {
                case 1 -> breakdown.foundationBonus = config.foundationCountBonus1;
                case 2 -> breakdown.foundationBonus = config.foundationCountBonus2;
                case 3 -> breakdown.foundationBonus = config.foundationCountBonus3;
                case 4 -> breakdown.foundationBonus = config.foundationCountBonus4;
                case 5 -> breakdown.foundationBonus = config.foundationCountBonus5;
            }
        }

        // ===== 4. Spatial compactness =====
        if (blocks.size() > 1) {
            double tileSize = GameConstants.TILE_SIZE;
            double scatteredThreshold = tileSize * 3;
            double minDistSq = Double.MAX_VALUE;
            for (BuildingBlock b : blocks) {
                if (b == placed) continue;
                double dx = b.getX() - placed.getX();
                double dy = b.getY() - placed.getY();
                double dSq = dx * dx + dy * dy;
                if (dSq < minDistSq) minDistSq = dSq;
            }
            double tileSizeSq = tileSize * tileSize;
            if (minDistSq <= tileSizeSq) {
                breakdown.spatialCompactness = config.spatialCompactnessBonus;
            } else if (minDistSq > scatteredThreshold * scatteredThreshold) {
                breakdown.spatialScatteredPenalty = config.spatialScatteredPenalty;
            }
        }

        breakdown.formulaReward = evaluateStepFormulas(config, gridModel, action, inserted, survived, placed, error, socketConnections);
        return breakdown;
    }

    private static double evaluateStepFormulas(RLRewardConfig config,
                                               GridModel gridModel,
                                               BuildAction action,
                                               boolean inserted,
                                               boolean survived,
                                               BuildingBlock placed,
                                               PlacementError error,
                                               int socketConnections) {
        if (config == null || config.rewardFormulaSet == null || config.rewardFormulaSet.isEmpty()) {
            return 0.0;
        }

        Map<String, Double> vars = new HashMap<>();
        int blockCount = gridModel != null ? gridModel.getAllBlocks().size() : 0;
        vars.put("inserted", inserted ? 1.0 : 0.0);
        vars.put("survived", survived ? 1.0 : 0.0);
        vars.put("invalid", (!inserted || error != PlacementError.NONE) ? 1.0 : 0.0);
        vars.put("block_count", (double) blockCount);
        vars.put("socket_connections", (double) socketConnections);
        vars.put("stability", placed != null ? placed.getStability() : 0.0);
        vars.put("is_foundation", placed != null && isFoundation(placed) ? 1.0 : 0.0);
        vars.put("is_tc_action", action != null && action.actionType == BuildAction.ActionType.TC ? 1.0 : 0.0);
        vars.put("is_workbench_action", action != null && action.actionType == BuildAction.ActionType.WORKBENCH ? 1.0 : 0.0);
        vars.put("is_loot_room_action", action != null && action.actionType == BuildAction.ActionType.LOOT_ROOM ? 1.0 : 0.0);
        vars.put("error_no_support", error == PlacementError.NO_SUPPORT ? 1.0 : 0.0);
        vars.put("error_collision", error == PlacementError.COLLISION ? 1.0 : 0.0);
        vars.put("error_bad_socket", isBadSocketError(error) ? 1.0 : 0.0);

        return config.rewardFormulaSet.evaluate(RewardFormulaScope.STEP, vars);
    }

    private static boolean isBadSocketError(PlacementError error) {
        return error == PlacementError.BAD_SOCKET
            || error == PlacementError.BAD_SOCKET_IS_FIRST
            || error == PlacementError.BAD_SOCKET_NO_TARGET
            || error == PlacementError.BAD_SOCKET_WRONG_TARGET_TYPE
            || error == PlacementError.BAD_SOCKET_NO_SOCKET_ALIGNMENT
            || error == PlacementError.BAD_SOCKET_CENTERDIST_REJECT;
    }

    /**
     * Count how many other blocks this block connects to via socket proximity.
     */
    private static int countSocketConnections(GridModel gridModel, BuildingBlock placed, List<BuildingBlock> allBlocks) {
        int connections = 0;
        List<Socket> placedSockets = placed.getSockets();
        List<BuildingBlock> candidates = allBlocks.size() > SPATIAL_REWARD_SEARCH_THRESHOLD
            ? gridModel.getNearbyBlocks(placed.getX(), placed.getY(), placed.getZ(), GameConstants.TILE_SIZE * 2.5)
            : allBlocks;
        
        for (BuildingBlock other : candidates) {
            if (other == placed) continue;
            
            // Quick distance check first
            if (Math.abs(placed.getX() - other.getX()) > GameConstants.TILE_SIZE * 2.5 ||
                Math.abs(placed.getY() - other.getY()) > GameConstants.TILE_SIZE * 2.5) {
                continue;
            }
            
            // Check socket-to-socket proximity
            boolean connected = false;
            for (Socket s1 : placedSockets) {
                for (Socket s2 : other.getSockets()) {
                    if (com.rustbuilder.util.SocketCompatibilityUtils.areEdgeSocketsConnected(s1, s2, 1.3)) {
                        connected = true;
                        break;
                    }
                }
                if (connected) break;
            }
            if (connected) connections++;
        }
        return connections;
    }

    private static boolean isFoundation(BuildingBlock b) {
        return b.getType() == BuildingType.FOUNDATION || b.getType() == BuildingType.TRIANGLE_FOUNDATION;
    }

    private static int countBlocksOfType(List<BuildingBlock> blocks, BuildingType type) {
        int count = 0;
        for (BuildingBlock block : blocks) {
            if (block.getType() == type) {
                count++;
            }
        }
        return count;
    }

    private static int countOpenDeployablesOfType(GridModel gridModel, String graphNodeType) {
        HouseGraph graph = new HouseGraph();
        graph.buildGraph(gridModel.getAllBlocks());

        int openCount = 0;
        for (TileNode node : graph.findAllNodesByType(graphNodeType)) {
            if (LogisticsEvaluator.isOpenToOutside(graph, node)) {
                openCount++;
            }
        }
        return openCount;
    }
}
