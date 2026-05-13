package com.rustbuilder.model.stability;

import com.rustbuilder.config.GameConstants;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.core.Socket;
import com.rustbuilder.model.structure.Door;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.util.BuildingTypeUtils;
import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.List;
import java.util.Queue;
import java.util.Set;

public class StabilityService {

    // Legacy method for list-based callers.
    public static void recalculateAll(List<BuildingBlock> blocks) {
        Queue<BuildingBlock> queue = new ArrayDeque<>();
        Set<BuildingBlock> inQueue = new HashSet<>();
        for (BuildingBlock b : blocks) {
            if (isFoundation(b) && b.getZ() == 0) {
                b.setStability(1.0);
                queue.add(b);
                inQueue.add(b);
            } else {
                b.setStability(0.0);
            }
        }

        // BFS propagation for the list-based compatibility path; O(N^2) per wave.
        while (!queue.isEmpty()) {
            BuildingBlock supporter = queue.poll();
            inQueue.remove(supporter);
            double supporterStability = supporter.getStability();

            if (supporterStability <= 0) continue;

            for (BuildingBlock supported : blocks) {
                if (supported == supporter) continue;
                if (isFoundation(supported)) continue;

                double factor = getSupportFactor(supported, supporter);
                if (factor > 0) {
                    double newStability = supporterStability * factor;
                    if (newStability > supported.getStability() + 0.001) {
                        supported.setStability(newStability);
                        if (inQueue.add(supported)) {
                            queue.add(supported);
                        }
                    }
                }
            }
        }
    }

    // Optimized method using GridModel spatial lookup.
    public static void recalculateAll(GridModel grid) {
        List<BuildingBlock> allBlocks = grid.getAllBlocks();

        Queue<BuildingBlock> queue = new ArrayDeque<>();
        Set<BuildingBlock> inQueue = new HashSet<>();
        for (BuildingBlock b : allBlocks) {
            if (isFoundation(b) && b.getZ() == 0) {
                b.setStability(1.0);
                queue.add(b);
                inQueue.add(b);
            } else {
                b.setStability(0.0);
            }
        }

        // BFS propagation: O(N) with spatial lookup.
        while (!queue.isEmpty()) {
            BuildingBlock supporter = queue.poll();
            inQueue.remove(supporter);
            double supporterStability = supporter.getStability();

            if (supporterStability <= 0) continue;

            List<BuildingBlock> candidates = grid.getNearbyBlocks(
                    supporter.getX(), supporter.getY(), supporter.getZ(), GameConstants.TILE_SIZE * 1.5
            );

            for (BuildingBlock supported : candidates) {
                if (supported == supporter) continue;
                if (isFoundation(supported)) continue;

                double factor = getSupportFactor(supported, supporter);
                if (factor > 0) {
                    double newStability = supporterStability * factor;
                    if (newStability > supported.getStability() + 0.001) {
                        supported.setStability(newStability);
                        if (inQueue.add(supported)) {
                            queue.add(supported);
                        }
                    }
                }
            }
        }
    }

    private static double getSupportFactor(BuildingBlock supported, BuildingBlock supporter) {
        double dx = supported.getX() - supporter.getX();
        double dy = supported.getY() - supporter.getY();
        double distSq = dx * dx + dy * dy;
        double zDiff = supported.getZ() - supporter.getZ();

        // 1. Wall on Foundation (vertical).
        if (isWall(supported) && isFoundation(supporter)) {
            if (zDiff == 0 && areSocketsConnected(supported, supporter, true)) {
                return 1.0;
            }
        }

        // 2. Wall on Wall (vertical stack).
        if (isWall(supported) && isWall(supporter)) {
            if (zDiff == 1 && areSocketsConnected(supported, supporter, false)) {
                return 0.9;
            }
        }

        // 3. Floor on Wall (ceiling).
        if (isFloor(supported) && isWall(supporter)) {
            if (zDiff == 1 && areSocketsConnected(supported, supporter, false)) {
                return 0.8;
            }
        }

        // 4. Floor on Floor (horizontal side connection).
        if (isFloor(supported) && isFloor(supporter)) {
            if (zDiff == 0 && areSocketsConnected(supported, supporter, false)) {
                return 0.5;
            }
        }

        // 5. Wall on Floor (vertical).
        if (isWall(supported) && isFloor(supporter)) {
            if (zDiff == 0 && areSocketsConnected(supported, supporter, false)) {
                return 0.9;
            }
        }

        // 6. TC / Workbench / LootRoom on Foundation or Floor.
        if ((supported.getType() == BuildingType.TC ||
                supported.getType() == BuildingType.WORKBENCH ||
                supported.getType() == BuildingType.LOOT_ROOM) &&
                (isFoundation(supporter) || isFloor(supporter))) {
            if (zDiff == 0 && distSq < 1.0) {
                return 1.0;
            }
        }

        // 7. Door inside a Doorway.
        if (supported.getType() == BuildingType.DOOR && supporter.getType() == BuildingType.DOORWAY) {
            if (zDiff == 0 && distSq < 1.0 && isMatchingDoorway(supported, supporter)) {
                return 1.0;
            }
        }

        return 0.0;
    }

    private static boolean isMatchingDoorway(BuildingBlock doorBlock, BuildingBlock doorwayBlock) {
        if (!(doorBlock instanceof Door) || !(doorwayBlock instanceof Wall)) {
            return false;
        }
        Door door = (Door) doorBlock;
        Wall doorway = (Wall) doorwayBlock;
        return door.getOrientation() == doorway.getOrientation();
    }

    private static boolean areSocketsConnected(BuildingBlock b1, BuildingBlock b2, boolean allowCenterConnection) {
        List<Socket> sockets1 = b1.getSockets();
        List<Socket> sockets2 = b2.getSockets();

        if (Math.abs(b1.getX() - b2.getX()) > GameConstants.TILE_SIZE * 2.5 ||
                Math.abs(b1.getY() - b2.getY()) > GameConstants.TILE_SIZE * 2.5) {
            return false;
        }

        for (Socket s1 : sockets1) {
            for (Socket s2 : sockets2) {
                // Increased tolerance by about 10% for easier floating-point snapping.
                if (com.rustbuilder.util.SocketCompatibilityUtils.areEdgeSocketsConnected(s1, s2, 1.3)) {
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

    public static boolean hasSupport(BuildingBlock block, List<BuildingBlock> potentialSupporters) {
        if (isFoundation(block) && block.getZ() == 0) {
            return true;
        }

        for (BuildingBlock supporter : potentialSupporters) {
            if (supporter == block) continue;
            if (supporter.getStability() > 0.0 && getSupportFactor(block, supporter) > 0) {
                return true;
            }
        }
        return false;
    }

    private static boolean isFoundation(BuildingBlock b) {
        return BuildingTypeUtils.isFoundation(b.getType());
    }

    private static boolean isWall(BuildingBlock b) {
        return BuildingTypeUtils.isWall(b.getType());
    }

    private static boolean isFloor(BuildingBlock b) {
        return BuildingTypeUtils.isFloor(b.getType());
    }
}
