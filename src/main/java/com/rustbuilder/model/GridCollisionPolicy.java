package com.rustbuilder.model;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.core.Socket;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.util.BuildingTypeUtils;
import com.rustbuilder.util.CollisionUtils;
import com.rustbuilder.util.SocketCompatibilityUtils;

final class GridCollisionPolicy {

    boolean allowsOverlap(BuildingBlock newBlock, BuildingBlock block) {
        if (block.getZ() != newBlock.getZ()) {
            int zDiff = newBlock.getZ() - block.getZ();

            if (zDiff == 1 && isHorizontalSurface(newBlock) && isWall(block)) {
                if (hasEdgeSocketConnection(block, newBlock)) {
                    return true;
                }
            } else if (zDiff == -1 && isHorizontalSurface(block) && isWall(newBlock)) {
                if (hasEdgeSocketConnection(newBlock, block)) {
                    return true;
                }
            } else {
                return true;
            }
        }

        boolean isNewWall = isWall(newBlock);
        boolean isExistingWall = isWall(block);
        boolean isNewFoundation = isFoundation(newBlock);
        boolean isExistingFoundation = isFoundation(block);
        boolean isNewCeiling = isCeiling(newBlock);
        boolean isExistingCeiling = isCeiling(block);

        if (block.getZ() == newBlock.getZ()
                && ((isNewWall && isExistingFoundation) || (isNewFoundation && isExistingWall))) {
            return true;
        }

        if ((isNewWall && isExistingCeiling) || (isNewCeiling && isExistingWall)) {
            BuildingBlock wallBlock = isNewWall ? newBlock : block;
            BuildingBlock ceilingBlock = isNewCeiling ? newBlock : block;
            if (hasEdgeSocketConnection(wallBlock, ceilingBlock)) {
                return true;
            }
        }

        boolean isNewDeployable = isDeployable(newBlock);
        boolean isExistingDeployable = isDeployable(block);
        boolean isExistingFloorBase = isFoundation(block) || isFloor(block);
        boolean isNewFloorBase = isFoundation(newBlock) || isFloor(newBlock);

        if ((isNewDeployable && isExistingFloorBase) || (isExistingDeployable && isNewFloorBase)) {
            return true;
        }

        boolean isNewDoor = newBlock.getType() == BuildingType.DOOR;
        boolean isExistingDoor = block.getType() == BuildingType.DOOR;
        boolean isExistingDoorway = block.getType() == BuildingType.DOORWAY;
        boolean isNewDoorway = newBlock.getType() == BuildingType.DOORWAY;

        if ((isNewDoor && (isExistingDoorway || isExistingWall))
                || (isExistingDoor && (isNewDoorway || isNewWall))) {
            return true;
        }

        if (isNewWall && isExistingWall
                && Math.abs(block.getX() - newBlock.getX()) < 1.0
                && Math.abs(block.getY() - newBlock.getY()) < 1.0) {
            if (block instanceof Wall && newBlock instanceof Wall) {
                if (((Wall) block).getOrientation() != ((Wall) newBlock).getOrientation()) {
                    return true;
                }
            } else {
                return true;
            }
        }

        return !CollisionUtils.checkCollision(newBlock.getCollisionPoints(), block.getCollisionPoints());
    }

    private boolean isDeployable(BuildingBlock block) {
        return BuildingTypeUtils.isFurniture(block.getType());
    }

    private boolean isWall(BuildingBlock block) {
        return BuildingTypeUtils.isWall(block.getType());
    }

    private boolean isFoundation(BuildingBlock block) {
        return BuildingTypeUtils.isFoundation(block.getType());
    }

    private boolean isFloor(BuildingBlock block) {
        return BuildingTypeUtils.isFloor(block.getType());
    }

    private boolean isCeiling(BuildingBlock block) {
        return block.getType() == BuildingType.FLOOR || block.getType() == BuildingType.TRIANGLE_FLOOR;
    }

    private boolean isHorizontalSurface(BuildingBlock block) {
        return isFoundation(block) || isFloor(block);
    }

    private boolean hasEdgeSocketConnection(BuildingBlock a, BuildingBlock b) {
        for (Socket sa : a.getSockets()) {
            if (sa.isCenter()) {
                continue;
            }
            for (Socket sb : b.getSockets()) {
                if (sb.isCenter()) {
                    continue;
                }
                if (SocketCompatibilityUtils.areEdgeSocketsConnected(sa, sb, 9.0)) {
                    return true;
                }
            }
        }
        return false;
    }
}
