package com.rustbuilder.model;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.util.BuildingTypeUtils;
import java.util.List;

final class GridPlacementRules {

    boolean allows(BuildingBlock newBlock, List<BuildingBlock> neighbors, List<BuildingBlock> allBlocks) {
        return hasRequiredBase(newBlock, neighbors) && respectsGlobalLimits(newBlock, allBlocks);
    }

    private boolean hasRequiredBase(BuildingBlock newBlock, List<BuildingBlock> neighbors) {
        if (!BuildingTypeUtils.isFurniture(newBlock.getType())) {
            return true;
        }

        for (BuildingBlock block : neighbors) {
            if ((BuildingTypeUtils.isFoundation(block.getType()) || BuildingTypeUtils.isFloor(block.getType()))
                    && block.getZ() == newBlock.getZ()
                    && Math.abs(block.getX() - newBlock.getX()) < 0.1
                    && Math.abs(block.getY() - newBlock.getY()) < 0.1) {
                return true;
            }
        }
        return false;
    }

    private boolean respectsGlobalLimits(BuildingBlock newBlock, List<BuildingBlock> allBlocks) {
        if (newBlock.getType() != BuildingType.TC) {
            return true;
        }

        for (BuildingBlock block : allBlocks) {
            if (block.getType() == BuildingType.TC) {
                return false;
            }
        }
        return true;
    }
}
