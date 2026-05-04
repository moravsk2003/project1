package com.rustbuilder.model;

import com.rustbuilder.model.core.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.util.BuildingTypeUtils;
import com.rustbuilder.config.GameConstants;

public class GridModel {
    private final List<BuildingBlock> blocks = new ArrayList<>();
    // Spatial index: key -> list of blocks in that grid cell
    private final LongBlockListMap spatialMap = new LongBlockListMap();

    // Key generation: Pack quantized coordinates into a long.
    // TILE_SIZE is 60. We use it as the grid cell size.
    private static long getSpatialKey(double x, double y, int z) {
        int gx = (int) Math.floor(x / GameConstants.TILE_SIZE);
        int gy = (int) Math.floor(y / GameConstants.TILE_SIZE);
        // Offset to handle negative coordinates and pack into positive range
        long lx = (gx + 500000L) & 0xFFFFF; 
        long ly = (gy + 500000L) & 0xFFFFF;
        long lz = (z + 128L) & 0xFF; // Z is usually 0-20
        return (lx) | (ly << 20) | (lz << 40);
    }

    public void addBlock(BuildingBlock block) {
        // Optimized duplicate check: only check nearby blocks
        // We use a small radius around the new block to find candidates for duplicate check
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
        }
    }

    /**
     * Adds a block without triggering stability recalculation.
     * Use this for bulk loading (e.g., during AI genome decode) and call
     * {@link #finalizeLoad()} once after all blocks are inserted.
     * This avoids O(N²) behaviour where each addBlock causes a full recalculate.
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
        com.rustbuilder.service.physics.StabilityService.recalculateAll(this);
        blocks.removeIf(b -> {
            if (b.getStability() < 0.1) {
                removeFromSpatialMap(b);
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
        // Use optimized spatial lookup
        com.rustbuilder.service.physics.StabilityService.recalculateAll(this); 
        
        // Remove unstable blocks
        blocks.removeIf(b -> {
            if (b.getStability() < 0.1) {
                removeFromSpatialMap(b);
                return true;
            }
            return false;
        });
    }

    public List<BuildingBlock> getAllBlocks() {
        return java.util.Collections.unmodifiableList(blocks);
    }

    private boolean isDuplicate(BuildingBlock b, BuildingBlock block) {
        if (b.getZ() != block.getZ()) return false;
        boolean samePos = Math.abs(b.getX() - block.getX()) < 0.1 &&
                          Math.abs(b.getY() - block.getY()) < 0.1;
        if (!samePos) return false;

        if (com.rustbuilder.util.BuildingTypeUtils.isWall(b.getType()) && 
            com.rustbuilder.util.BuildingTypeUtils.isWall(block.getType()) && 
            b instanceof Wall && block instanceof Wall) {
            return ((Wall) b).getOrientation() == ((Wall) block).getOrientation();
        }
        return b.getType() == block.getType();
    }
    
    /**
     * Efficiently get blocks within a certain radius.
     */
    public List<BuildingBlock> getNearbyBlocks(double x, double y, int z, double radius) {
        if (blocks.isEmpty()) {
            return java.util.Collections.emptyList();
        }
        List<BuildingBlock> nearby = new ArrayList<>();
        int gxMin = (int) Math.floor((x - radius) / GameConstants.TILE_SIZE);
        int gxMax = (int) Math.floor((x + radius) / GameConstants.TILE_SIZE);
        int gyMin = (int) Math.floor((y - radius) / GameConstants.TILE_SIZE);
        int gyMax = (int) Math.floor((y + radius) / GameConstants.TILE_SIZE);
        
        // Check Z range: z-2 to z+2 covers most interactions
        for (int gx = gxMin; gx <= gxMax; gx++) {
            for (int gy = gyMin; gy <= gyMax; gy++) {
                for (int kz = z - 2; kz <= z + 2; kz++) { 
                     long key = (((long)(gx + 500000L) & 0xFFFFF)) | 
                                (((long)(gy + 500000L) & 0xFFFFF) << 20) | 
                                (((long)(kz + 128L) & 0xFF) << 40);
                                
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
     * Efficiently checks if a new block strictly collides with any existing block.
     * Bypasses the expensive stability support check.
     */
    public boolean hasCollision(BuildingBlock newBlock) {
        double checkRadius = GameConstants.TILE_SIZE * 1.5;
        List<BuildingBlock> neighbors = getNearbyBlocks(newBlock.getX(), newBlock.getY(), newBlock.getZ(), checkRadius);

        double[] newPoly = newBlock.getPolygonPoints();
        if (newPoly.length > 0) {
            for (BuildingBlock block : neighbors) {
                if (block == newBlock) continue;
                if (!checkCollision(newBlock, block)) return true; // Collision detected
            }
        }
        return false;
    }

    /**
     * Optimized placement check.
     */
    public boolean canPlace(BuildingBlock newBlock) {
        // 1. Collision Check: only check nearby blocks
        double checkRadius = GameConstants.TILE_SIZE * 1.5;
        List<BuildingBlock> neighbors = getNearbyBlocks(newBlock.getX(), newBlock.getY(), newBlock.getZ(), checkRadius);

        double[] newPoly = newBlock.getPolygonPoints();
        if (newPoly.length > 0) {
            for (BuildingBlock block : neighbors) {
                if (block == newBlock) continue;
                if (!checkCollision(newBlock, block)) return false;
            }
        }

        // 2. Stability Check
        // Use neighbors list for support check!
        // This is O(K) instead of O(N)
        if (!com.rustbuilder.service.physics.StabilityService.hasSupport(newBlock, neighbors)) {
            return false;
        }

        // 3. Strict Rules (TC/Loot on Foundation)
        if (newBlock.getType() == BuildingType.TC ||
            newBlock.getType() == BuildingType.WORKBENCH ||
            newBlock.getType() == BuildingType.LOOT_ROOM) {
            
            boolean hasBase = false;
            for (BuildingBlock b : neighbors) {
                if ((isFoundation(b) || isFloor(b)) && b.getZ() == newBlock.getZ()) {
                    if (Math.abs(b.getX() - newBlock.getX()) < 0.1 &&
                        Math.abs(b.getY() - newBlock.getY()) < 0.1) {
                        hasBase = true;
                        break;
                    }
                }
            }
            if (!hasBase) return false;
        }

        // 4. Global TC Limit
        if (newBlock.getType() == BuildingType.TC) {
            for (BuildingBlock b : blocks) {
                if (b.getType() == BuildingType.TC) return false;
            }
        }

        return true;
    }

    private boolean checkCollision(BuildingBlock newBlock, BuildingBlock block) {
        // Vertical separation: ceilings may touch walls one floor below only when
        // an edge socket lines up. Otherwise that lower wall is treated as a
        // real collision instead of being silently accepted as support.
        if (block.getZ() != newBlock.getZ()) {
            if (isCeiling(newBlock) && isWall(block) && block.getZ() == newBlock.getZ() - 1) {
                return isValidCeilingOverLowerWall(newBlock, block);
            }
            if (isCeiling(block) && isWall(newBlock) && newBlock.getZ() == block.getZ() - 1) {
                return isValidCeilingOverLowerWall(block, newBlock);
            }
            return true; // No collision possible (different Z)
        }

        // Ignore collision between Wall and any horizontal surface (Foundation/Floor)
        boolean isNewWall = isWall(newBlock);
        boolean isExistingWall = isWall(block);
        boolean isNewHorizontal = isFoundation(newBlock) || isFloor(newBlock);
        boolean isExistingHorizontal = isFoundation(block) || isFloor(block);

        if ((isNewWall && isExistingHorizontal) || (isNewHorizontal && isExistingWall)) {
            return true;
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

        if ((isNewDoor && (isExistingDoorway || isExistingWall)) || 
            (isExistingDoor && (isNewDoorway || isNewWall))) {
            return true;
        }

        if (isNewWall && isExistingWall) {
            if (Math.abs(block.getX() - newBlock.getX()) < 1.0 &&
                    Math.abs(block.getY() - newBlock.getY()) < 1.0) {
                if (block instanceof Wall && newBlock instanceof Wall) {
                     if (((Wall)block).getOrientation() != ((Wall)newBlock).getOrientation()) {
                         return true;
                     }
                } else {
                    return true;
                }
            }
        }

        if (com.rustbuilder.util.CollisionUtils.checkCollision(newBlock.getCollisionPoints(), block.getCollisionPoints())) {
            return false;
        }
        return true;
    }

    private boolean isDeployable(BuildingBlock b) {
        return BuildingTypeUtils.isFurniture(b.getType());
    }
    private boolean isWall(BuildingBlock b) {
        return BuildingTypeUtils.isWall(b.getType());
    }

    public GridModel clone() {
        GridModel clone = new GridModel();
        for (BuildingBlock block : this.blocks) {
            BuildingBlock bc = cloneBlock(block);
            if (bc != null) {
                clone.addBlockSilent(bc);
            }
        }
        return clone;
    }

    private BuildingBlock cloneBlock(BuildingBlock b) {
        BuildingBlock clone = null;
        if (b instanceof com.rustbuilder.model.structure.Foundation) clone = new com.rustbuilder.model.structure.Foundation(b.getX(), b.getY(), b.getZ());
        else if (b instanceof com.rustbuilder.model.structure.TriangleFoundation) clone = new com.rustbuilder.model.structure.TriangleFoundation(b.getX(), b.getY(), b.getZ(), b.getRotation());
        else if (b instanceof Wall) {
            Wall w = (Wall)b;
            clone = new Wall(w.getX(), w.getY(), w.getZ(), w.getOrientation());
            clone.setType(w.getType());
            if (w.getType() == BuildingType.DOORWAY) ((Wall)clone).setDoorType(w.getDoorType());
        }
        else if (b instanceof com.rustbuilder.model.structure.Floor) clone = new com.rustbuilder.model.structure.Floor(b.getX(), b.getY(), b.getZ(), b.getRotation());
        else if (b instanceof com.rustbuilder.model.structure.TriangleFloor) clone = new com.rustbuilder.model.structure.TriangleFloor(b.getX(), b.getY(), b.getZ(), b.getRotation());
        else if (b instanceof com.rustbuilder.model.structure.Door) clone = new com.rustbuilder.model.structure.Door(b.getX(), b.getY(), b.getZ(), ((com.rustbuilder.model.structure.Door)b).getOrientation(), ((com.rustbuilder.model.structure.Door)b).getDoorType());
        else if (b instanceof com.rustbuilder.model.deployable.ToolCupboard) clone = new com.rustbuilder.model.deployable.ToolCupboard(b.getX(), b.getY(), b.getZ(), b.getRotation());
        else if (b instanceof com.rustbuilder.model.deployable.Workbench) clone = new com.rustbuilder.model.deployable.Workbench(b.getX(), b.getY(), b.getZ(), b.getRotation());
        else if (b instanceof com.rustbuilder.model.deployable.LootRoom) clone = new com.rustbuilder.model.deployable.LootRoom(b.getX(), b.getY(), b.getZ(), b.getRotation());

        if (clone != null) {
            clone.setRotation(b.getRotation());
            clone.setTier(b.getTier());
        }
        return clone;
    }

    private boolean isFoundation(BuildingBlock b) {
        return BuildingTypeUtils.isFoundation(b.getType());
    }

    private boolean isFloor(BuildingBlock b) {
        return BuildingTypeUtils.isFloor(b.getType());
    }

    private boolean isCeiling(BuildingBlock b) {
        return b.getType() == BuildingType.FLOOR || b.getType() == BuildingType.TRIANGLE_FLOOR;
    }

    private boolean isValidCeilingOverLowerWall(BuildingBlock ceiling, BuildingBlock wallBelow) {
        if (!isCeiling(ceiling) || !isWall(wallBelow) || wallBelow.getZ() != ceiling.getZ() - 1) {
            return false;
        }

        for (Socket ceilingSocket : ceiling.getSockets()) {
            if (ceilingSocket.getSide() == 10) {
                continue;
            }
            for (Socket wallSocket : wallBelow.getSockets()) {
                if (wallSocket.getSide() == 10) {
                    continue;
                }
                double dx = ceilingSocket.getX() - wallSocket.getX();
                double dy = ceilingSocket.getY() - wallSocket.getY();
                if (dx * dx + dy * dy < 1.3) {
                    return true;
                }
            }
        }
        return false;
    }

    public void clear() {
        blocks.clear();
        spatialMap.clear();
    }

    private static final class LongBlockListMap {
        private static final int DEFAULT_CAPACITY = 32;
        private static final float MAX_LOAD = 0.65f;

        private long[] keys;
        private List<BuildingBlock>[] values;
        private byte[] states; // 0 = empty, 1 = occupied, 2 = deleted
        private int size;
        private int tombstones;
        private int resizeThreshold;

        LongBlockListMap() {
            allocate(DEFAULT_CAPACITY);
        }

        List<BuildingBlock> get(long key) {
            int mask = keys.length - 1;
            int index = mix(key) & mask;
            while (true) {
                byte state = states[index];
                if (state == 0) {
                    return null;
                }
                if (state == 1 && keys[index] == key) {
                    return values[index];
                }
                index = (index + 1) & mask;
            }
        }

        List<BuildingBlock> getOrCreate(long key) {
            if (size + tombstones + 1 > resizeThreshold) {
                rehash(keys.length * 2);
            }

            int mask = keys.length - 1;
            int index = mix(key) & mask;
            int firstDeleted = -1;
            while (true) {
                byte state = states[index];
                if (state == 0) {
                    int target = firstDeleted >= 0 ? firstDeleted : index;
                    if (firstDeleted >= 0) {
                        tombstones--;
                    }
                    keys[target] = key;
                    states[target] = 1;
                    values[target] = new ArrayList<>();
                    size++;
                    return values[target];
                }
                if (state == 1 && keys[index] == key) {
                    return values[index];
                }
                if (state == 2 && firstDeleted < 0) {
                    firstDeleted = index;
                }
                index = (index + 1) & mask;
            }
        }

        void remove(long key) {
            int mask = keys.length - 1;
            int index = mix(key) & mask;
            while (true) {
                byte state = states[index];
                if (state == 0) {
                    return;
                }
                if (state == 1 && keys[index] == key) {
                    states[index] = 2;
                    values[index] = null;
                    size--;
                    tombstones++;
                    if (tombstones > size && keys.length > DEFAULT_CAPACITY) {
                        rehash(keys.length);
                    }
                    return;
                }
                index = (index + 1) & mask;
            }
        }

        void clear() {
            Arrays.fill(states, (byte) 0);
            Arrays.fill(values, null);
            size = 0;
            tombstones = 0;
        }

        private void rehash(int capacity) {
            long[] oldKeys = keys;
            List<BuildingBlock>[] oldValues = values;
            byte[] oldStates = states;
            allocate(capacity);

            for (int i = 0; i < oldKeys.length; i++) {
                if (oldStates[i] == 1) {
                    putRehashed(oldKeys[i], oldValues[i]);
                }
            }
        }

        private void putRehashed(long key, List<BuildingBlock> value) {
            int mask = keys.length - 1;
            int index = mix(key) & mask;
            while (states[index] == 1) {
                index = (index + 1) & mask;
            }
            keys[index] = key;
            values[index] = value;
            states[index] = 1;
            size++;
        }

        @SuppressWarnings("unchecked")
        private void allocate(int requestedCapacity) {
            int capacity = 1;
            while (capacity < requestedCapacity) {
                capacity <<= 1;
            }
            keys = new long[capacity];
            values = (List<BuildingBlock>[]) new List[capacity];
            states = new byte[capacity];
            size = 0;
            tombstones = 0;
            resizeThreshold = Math.max(1, (int) (capacity * MAX_LOAD));
        }

        private static int mix(long value) {
            value ^= value >>> 33;
            value *= 0xff51afd7ed558ccdL;
            value ^= value >>> 33;
            value *= 0xc4ceb9fe1a85ec53L;
            value ^= value >>> 33;
            return (int) value;
        }
    }
}
