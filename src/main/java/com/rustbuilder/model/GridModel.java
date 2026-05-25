package com.rustbuilder.model;

import com.rustbuilder.config.GameConstants;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.spatial.LongBlockListMap;
import com.rustbuilder.model.stability.StabilityService;
import com.rustbuilder.model.structure.Door;
import com.rustbuilder.model.structure.Wall;
import java.util.ArrayList;
import java.util.List;

public class GridModel {
    private final List<BuildingBlock> blocks = new ArrayList<>();
    private final LongBlockListMap spatialMap = new LongBlockListMap();
    private final GridCollisionPolicy collisionPolicy = new GridCollisionPolicy();
    private final GridPlacementRules placementRules = new GridPlacementRules();

    private static long getSpatialKey(double x, double y, int z) {
        int gx = (int) Math.floor(x / GameConstants.TILE_SIZE);
        int gy = (int) Math.floor(y / GameConstants.TILE_SIZE);
        long lx = (gx + 500000L) & 0xFFFFF;
        long ly = (gy + 500000L) & 0xFFFFF;
        long lz = (z + 128L) & 0xFF;
        return lx | (ly << 20) | (lz << 40);
    }

    public boolean addBlock(BuildingBlock block) {
        List<BuildingBlock> candidates = getNearbyBlocks(block.getX(), block.getY(), block.getZ(), 1.0);

        boolean exists = false;
        for (BuildingBlock candidate : candidates) {
            if (isDuplicate(candidate, block)) {
                exists = true;
                break;
            }
        }

        if (!exists) {
            blocks.add(block);
            addToSpatialMap(block);
            updateStability();
            return blocks.contains(block);
        }
        return false;
    }

    /**
     * Adds a block without triggering stability recalculation.
     * Use this for bulk loading and call {@link #finalizeLoad()} once afterward.
     * This avoids O(N^2) behavior where each addBlock causes a full recalculate.
     */
    public boolean addBlockSilent(BuildingBlock block) {
        List<BuildingBlock> candidates = getNearbyBlocks(block.getX(), block.getY(), block.getZ(), 1.0);

        boolean exists = false;
        for (BuildingBlock candidate : candidates) {
            if (isDuplicate(candidate, block)) {
                exists = true;
                break;
            }
        }

        if (!exists) {
            blocks.add(block);
            addToSpatialMap(block);
            return true;
        }
        return false;
    }

    /**
     * Runs stability recalculation once after a bulk {@link #addBlockSilent} session,
     * then removes any blocks that ended up unsupported.
     */
    public void finalizeLoad() {
        StabilityService.recalculateAll(this);
        blocks.removeIf(block -> {
            if (block.getStability() < 0.1) {
                removeFromSpatialMap(block);
                return true;
            }
            return false;
        });
    }

    private void addToSpatialMap(BuildingBlock block) {
        long key = getSpatialKey(block.getX(), block.getY(), block.getZ());
        spatialMap.getOrCreate(key).add(block);
    }

    public void removeBlock(BuildingBlock block) {
        blocks.remove(block);
        removeFromSpatialMap(block);
        updateStability();
    }

    private void removeFromSpatialMap(BuildingBlock block) {
        long key = getSpatialKey(block.getX(), block.getY(), block.getZ());
        List<BuildingBlock> cell = spatialMap.get(key);
        if (cell != null) {
            cell.remove(block);
            if (cell.isEmpty()) {
                spatialMap.remove(key);
            }
        }
    }

    private void updateStability() {
        StabilityService.recalculateAll(this);
        blocks.removeIf(block -> {
            if (block.getStability() < 0.1) {
                removeFromSpatialMap(block);
                return true;
            }
            return false;
        });
    }

    public List<BuildingBlock> getAllBlocks() {
        return java.util.Collections.unmodifiableList(blocks);
    }

    private boolean isDuplicate(BuildingBlock existing, BuildingBlock block) {
        if (existing.getZ() != block.getZ()) {
            return false;
        }
        boolean samePos = Math.abs(existing.getX() - block.getX()) < 0.1
            && Math.abs(existing.getY() - block.getY()) < 0.1;
        if (!samePos) {
            return false;
        }

        if (com.rustbuilder.util.BuildingTypeUtils.isWall(existing.getType())
                && com.rustbuilder.util.BuildingTypeUtils.isWall(block.getType())
                && existing instanceof Wall && block instanceof Wall) {
            return ((Wall) existing).getOrientation() == ((Wall) block).getOrientation();
        }

        if (existing.getType() == BuildingType.DOOR && block.getType() == BuildingType.DOOR
                && existing instanceof Door && block instanceof Door) {
            return ((Door) existing).getOrientation() == ((Door) block).getOrientation();
        }
        return existing.getType() == block.getType();
    }

    public List<BuildingBlock> getNearbyBlocks(double x, double y, int z, double radius) {
        if (blocks.isEmpty()) {
            return java.util.Collections.emptyList();
        }
        List<BuildingBlock> nearby = new ArrayList<>();
        int gxMin = (int) Math.floor((x - radius) / GameConstants.TILE_SIZE);
        int gxMax = (int) Math.floor((x + radius) / GameConstants.TILE_SIZE);
        int gyMin = (int) Math.floor((y - radius) / GameConstants.TILE_SIZE);
        int gyMax = (int) Math.floor((y + radius) / GameConstants.TILE_SIZE);

        for (int gx = gxMin; gx <= gxMax; gx++) {
            for (int gy = gyMin; gy <= gyMax; gy++) {
                for (int kz = z - 2; kz <= z + 2; kz++) {
                    long key = (((long) (gx + 500000L) & 0xFFFFF))
                        | (((long) (gy + 500000L) & 0xFFFFF) << 20)
                        | (((long) (kz + 128L) & 0xFF) << 40);

                    List<BuildingBlock> cell = spatialMap.get(key);
                    if (cell != null) {
                        nearby.addAll(cell);
                    }
                }
            }
        }
        return nearby;
    }

    /**
     * Efficiently checks whether a block collides with existing geometry.
     * This bypasses the more expensive stability support check.
     */
    public boolean hasCollision(BuildingBlock newBlock) {
        double checkRadius = GameConstants.TILE_SIZE * 1.5;
        List<BuildingBlock> neighbors = getNearbyBlocks(newBlock.getX(), newBlock.getY(), newBlock.getZ(), checkRadius);

        double[] newPoly = newBlock.getPolygonPoints();
        if (newPoly.length > 0) {
            for (BuildingBlock block : neighbors) {
                if (block == newBlock) {
                    continue;
                }
                if (!collisionPolicy.allowsOverlap(newBlock, block)) {
                    return true;
                }
            }
        }
        return false;
    }

    public boolean canPlace(BuildingBlock newBlock) {
        double checkRadius = GameConstants.TILE_SIZE * 1.5;
        List<BuildingBlock> neighbors = getNearbyBlocks(newBlock.getX(), newBlock.getY(), newBlock.getZ(), checkRadius);

        for (BuildingBlock block : neighbors) {
            if (block != newBlock && isDuplicate(block, newBlock)) {
                return false;
            }
        }

        double[] newPoly = newBlock.getPolygonPoints();
        if (newPoly.length > 0) {
            for (BuildingBlock block : neighbors) {
                if (block == newBlock) {
                    continue;
                }
                if (!collisionPolicy.allowsOverlap(newBlock, block)) {
                    return false;
                }
            }
        }

        if (!StabilityService.hasSupport(newBlock, neighbors)) {
            return false;
        }

        return placementRules.allows(newBlock, neighbors, blocks);
    }

    @Override
    public GridModel clone() {
        GridModel clone = new GridModel();
        for (BuildingBlock block : this.blocks) {
            BuildingBlock blockCopy = block.clone();
            if (blockCopy != null) {
                clone.addBlockSilent(blockCopy);
            }
        }
        return clone;
    }

    public void clear() {
        blocks.clear();
        spatialMap.clear();
    }
}
