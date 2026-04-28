package com.rustbuilder.ai.rl;

import java.util.List;
import com.rustbuilder.ai.ea.BaseGenome.BuildAction;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.Socket;
import com.rustbuilder.service.physics.StabilityService;
import com.rustbuilder.config.GameConstants;

/**
 * Calculates intermediate rewards for each RL step to guide the agent.
 * Rewards are kept small (mostly 0.0-0.5 range) to avoid Q-value explosion.
 */
public class StepRewardFunction {

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
        int socketConnections = countSocketConnections(placed, blocks);
        boolean isFurniture = placed.getType() == BuildingType.TC || 
                              placed.getType() == BuildingType.WORKBENCH || 
                              placed.getType() == BuildingType.LOOT_ROOM;
        
        if (socketConnections > 0) {
            reward += Math.min(socketConnections * config.socketConnectionReward, config.socketConnectionMax);
        } else if (blocks.size() > 1 && !isFurniture) {
            reward += config.disconnectedSegmentPenalty;
        }

        // ===== 2. Structural Stability Reward =====
        StabilityService.recalculateAll(gridModel);
        double stability = placed.getStability();
        if (stability > 0) {
            reward += stability * config.stabilityRewardMult;
        } else if (!isFoundation(placed)) {
            reward += config.floatingBlockPenalty;
        }

        // ===== 3. Type-specific bonuses =====
        if (action.actionType == BuildAction.ActionType.TC) {
            reward += config.tcPlacementBonus;
        } else if (action.actionType == BuildAction.ActionType.WORKBENCH || 
                   action.actionType == BuildAction.ActionType.LOOT_ROOM) {
            reward += config.secondaryDeployableBonus;
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
            double minDist = Double.MAX_VALUE;
            for (BuildingBlock b : blocks) {
                if (b == placed) continue;
                double d = Math.hypot(b.getX() - placed.getX(), b.getY() - placed.getY());
                if (d < minDist) minDist = d;
            }
            double tileSize = GameConstants.TILE_SIZE;
            if (minDist <= tileSize) {
                reward += config.spatialCompactnessBonus;
            } else if (minDist > tileSize * 3) {
                reward += config.spatialScatteredPenalty;
            }
        }

        return reward;
    }

    /**
     * Count how many other blocks this block connects to via socket proximity.
     */
    private static int countSocketConnections(BuildingBlock placed, List<BuildingBlock> allBlocks) {
        int connections = 0;
        List<Socket> placedSockets = placed.getSockets();
        
        for (BuildingBlock other : allBlocks) {
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
}
