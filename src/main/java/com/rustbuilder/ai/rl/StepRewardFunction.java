package com.rustbuilder.ai.rl;

import java.util.List;
import com.rustbuilder.ai.ea.BaseGenome.BuildAction;
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
        if (!inserted || error != PlacementError.NONE) {
            return switch (error) {
                case NO_SUPPORT -> config.penaltyNoSupport;
                case BAD_SOCKET_IS_FIRST, BAD_SOCKET_NO_TARGET, BAD_SOCKET_WRONG_TARGET_TYPE, BAD_SOCKET_NO_SOCKET_ALIGNMENT, BAD_SOCKET_CENTERDIST_REJECT -> config.penaltyBadSocket;
                case COLLISION -> config.penaltyCollision;
                case FLOOR_CONSTRAINT -> config.penaltyFloorConstraint;
                default -> config.penaltyGenericInvalid;
            };
        }
        if (!survived || placed == null) {
            return config.penaltyNoSupport;
        }

        double reward = config.basePlacementReward;

        List<BuildingBlock> blocks = gridModel.getAllBlocks();
        if (blocks.isEmpty()) return reward;

        // ===== 1. Socket Connection Reward =====
        int socketConnections = countSocketConnections(gridModel, placed, blocks);
        boolean isFurniture = placed.getType() == BuildingType.TC || 
                              placed.getType() == BuildingType.WORKBENCH || 
                              placed.getType() == BuildingType.LOOT_ROOM;
        
        if (socketConnections > 0) {
            reward += Math.min(socketConnections * config.socketConnectionReward, config.socketConnectionMax);
        } else if (blocks.size() > 1 && !isFurniture) {
            reward += config.disconnectedSegmentPenalty;
        }

        // ===== 2. Structural Stability Reward =====
        // EpisodeRunner calls GridModel.finalizeLoad() immediately before reward calculation.
        double stability = placed.getStability();
        if (stability > 0) {
            reward += stability * config.stabilityRewardMult;
        } else if (!isFoundation(placed)) {
            reward += config.floatingBlockPenalty;
        }

        // ===== 3. Type-specific bonuses =====
        if (action.actionType == BuildAction.ActionType.TC) {
            reward += config.tcPlacementBonus;
        } else if (action.actionType == BuildAction.ActionType.WORKBENCH) {
            if (countBlocksOfType(blocks, BuildingType.WORKBENCH) == 1 &&
                    countOpenDeployablesOfType(gridModel, "workbench") <= 1) {
                reward += config.secondaryDeployableBonus;
            }
        } else if (action.actionType == BuildAction.ActionType.LOOT_ROOM) {
            if (countOpenDeployablesOfType(gridModel, "loot_room") <= 1) {
                reward += config.secondaryDeployableBonus;
            }
        }

        // Foundation count bonus
        if (isFoundation(placed)) {
            int foundationCount = 0;
            for (BuildingBlock b : blocks) {
                if (isFoundation(b)) foundationCount++;
            }
            switch (foundationCount) {
                case 1 -> reward += config.foundationCountBonus1;
                case 2 -> reward += config.foundationCountBonus2;
                case 3 -> reward += config.foundationCountBonus3;
                case 4 -> reward += config.foundationCountBonus4;
                case 5 -> reward += config.foundationCountBonus5;
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
                reward += config.spatialCompactnessBonus;
            } else if (minDistSq > scatteredThreshold * scatteredThreshold) {
                reward += config.spatialScatteredPenalty;
            }
        }

        return reward;
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
                    double dx = s1.getX() - s2.getX();
                    double dy = s1.getY() - s2.getY();
                    if (dx * dx + dy * dy < 1.3) {
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
