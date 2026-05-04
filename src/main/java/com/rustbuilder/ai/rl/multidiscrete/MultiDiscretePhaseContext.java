package com.rustbuilder.ai.rl.multidiscrete;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.config.GameConstants;
import com.rustbuilder.util.BuildingTypeUtils;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * [RL REDESIGN]
 * Immutable context containing all data required for a multi-discrete phase decision.
 */
public class MultiDiscretePhaseContext {
    private final GridModel grid;
    private final boolean hasTC;
    private final boolean hasLootRoom;
    private final int step;
    private final int maxSteps;
    private final int blockCount;
    private final boolean hasAnyWall;
    private final boolean hasAnyBlockAtFloor0;
    private final boolean[] hasHorizontalAtFloor;
    private final boolean[] hasWallAtFloor;
    private final boolean[][] occupiedTiles;
    private final boolean[][] horizontalTiles;
    private final boolean[][] wallTiles;
    private final List<Integer>[] surfaceTilesByFloor;
    private final List<Integer>[] nearSurfaceTilesByFloor;
    private final List<Integer>[] nearWallTilesByFloor;
    private final List<Integer>[] wallPlacementTilesByFloor;
    private final List<Integer>[] ceilingPlacementTilesByFloor;

    public MultiDiscretePhaseContext(GridModel grid, boolean hasTC, boolean hasLootRoom, int step, int maxSteps) {
        this.grid = grid;
        this.hasTC = hasTC;
        this.hasLootRoom = hasLootRoom;
        this.step = step;
        this.maxSteps = maxSteps;

        boolean anyWall = false;
        boolean anyBlockAtFloor0 = false;
        boolean[] horizontalAtFloor = new boolean[MultiDiscreteActionSpace.FLOOR_COUNT];
        boolean[] wallAtFloor = new boolean[MultiDiscreteActionSpace.FLOOR_COUNT];
        boolean[][] occupied = new boolean[MultiDiscreteActionSpace.FLOOR_COUNT][MultiDiscreteActionSpace.TILE_COUNT];
        boolean[][] horizontal = new boolean[MultiDiscreteActionSpace.FLOOR_COUNT][MultiDiscreteActionSpace.TILE_COUNT];
        boolean[][] wall = new boolean[MultiDiscreteActionSpace.FLOOR_COUNT][MultiDiscreteActionSpace.TILE_COUNT];
        int count = 0;

        if (grid != null) {
            for (BuildingBlock block : grid.getAllBlocks()) {
                count++;
                int z = block.getZ();
                if (z >= 0 && z < wallAtFloor.length) {
                    if (z == 0) {
                        anyBlockAtFloor0 = true;
                    }
                    int tile = worldToTileIndex(block.getX(), block.getY());
                    if (tile >= 0) {
                        occupied[z][tile] = true;
                    }
                    if (BuildingTypeUtils.isHorizontalSurface(block.getType())) {
                        horizontalAtFloor[z] = true;
                        if (tile >= 0) {
                            horizontal[z][tile] = true;
                        }
                    }
                    if (BuildingTypeUtils.isWall(block.getType())) {
                        anyWall = true;
                        wallAtFloor[z] = true;
                        if (tile >= 0) {
                            wall[z][tile] = true;
                        }
                    }
                }
            }
        }

        this.blockCount = count;
        this.hasAnyWall = anyWall;
        this.hasAnyBlockAtFloor0 = anyBlockAtFloor0;
        this.hasHorizontalAtFloor = horizontalAtFloor;
        this.hasWallAtFloor = wallAtFloor;
        this.occupiedTiles = occupied;
        this.horizontalTiles = horizontal;
        this.wallTiles = wall;
        this.surfaceTilesByFloor = buildExactTileLists(horizontal);
        this.nearSurfaceTilesByFloor = buildNearTileLists(horizontal);
        this.nearWallTilesByFloor = buildNearTileLists(wall);
        this.wallPlacementTilesByFloor = buildWallPlacementTileLists();
        this.ceilingPlacementTilesByFloor = buildCeilingPlacementTileLists();
    }

    public GridModel getGrid() {
        return grid;
    }

    public boolean isHasTC() {
        return hasTC;
    }

    public boolean isHasLootRoom() {
        return hasLootRoom;
    }

    public int getStep() {
        return step;
    }

    public int getMaxSteps() {
        return maxSteps;
    }

    public int getBlockCount() {
        return blockCount;
    }

    public boolean hasAnyWall() {
        return hasAnyWall;
    }

    public boolean hasAnyBlockAtFloor0() {
        return hasAnyBlockAtFloor0;
    }

    public boolean hasWallAtFloor(int floor) {
        return floor >= 0 && floor < hasWallAtFloor.length && hasWallAtFloor[floor];
    }

    public boolean hasHorizontalAtFloor(int floor) {
        return floor >= 0 && floor < hasHorizontalAtFloor.length && hasHorizontalAtFloor[floor];
    }

    public boolean isOccupiedTile(int floor, int tileIndex) {
        return isValidFloorTile(floor, tileIndex) && occupiedTiles[floor][tileIndex];
    }

    public boolean isHorizontalTile(int floor, int tileIndex) {
        return isValidFloorTile(floor, tileIndex) && horizontalTiles[floor][tileIndex];
    }

    public boolean isWallTile(int floor, int tileIndex) {
        return isValidFloorTile(floor, tileIndex) && wallTiles[floor][tileIndex];
    }

    public List<Integer> getSurfaceTiles(int floor) {
        return getFloorList(surfaceTilesByFloor, floor);
    }

    public List<Integer> getNearSurfaceTiles(int floor) {
        return getFloorList(nearSurfaceTilesByFloor, floor);
    }

    public List<Integer> getNearWallTiles(int floor) {
        return getFloorList(nearWallTilesByFloor, floor);
    }

    public List<Integer> getWallPlacementTiles(int floor) {
        return getFloorList(wallPlacementTilesByFloor, floor);
    }

    public List<Integer> getCeilingPlacementTiles(int floor) {
        return getFloorList(ceilingPlacementTilesByFloor, floor);
    }

    private static boolean isValidFloorTile(int floor, int tileIndex) {
        return floor >= 0
            && floor < MultiDiscreteActionSpace.FLOOR_COUNT
            && tileIndex >= 0
            && tileIndex < MultiDiscreteActionSpace.TILE_COUNT;
    }

    private static int worldToTileIndex(double x, double y) {
        int tx = (int) Math.round((x - GameConstants.GRID_ORIGIN_X) / GameConstants.TILE_SIZE);
        int ty = (int) Math.round((y - GameConstants.GRID_ORIGIN_Y) / GameConstants.TILE_SIZE);
        if (tx < 0 || tx >= MultiDiscreteActionSpace.GRID_SIZE || ty < 0 || ty >= MultiDiscreteActionSpace.GRID_SIZE) {
            return -1;
        }
        return tx * MultiDiscreteActionSpace.GRID_SIZE + ty;
    }

    @SuppressWarnings("unchecked")
    private static List<Integer>[] buildExactTileLists(boolean[][] source) {
        List<Integer>[] result = new List[MultiDiscreteActionSpace.FLOOR_COUNT];
        for (int floor = 0; floor < result.length; floor++) {
            List<Integer> tiles = new ArrayList<>();
            for (int tile = 0; tile < MultiDiscreteActionSpace.TILE_COUNT; tile++) {
                if (source[floor][tile]) {
                    tiles.add(tile);
                }
            }
            result[floor] = Collections.unmodifiableList(tiles);
        }
        return result;
    }

    @SuppressWarnings("unchecked")
    private static List<Integer>[] buildNearTileLists(boolean[][] source) {
        List<Integer>[] result = new List[MultiDiscreteActionSpace.FLOOR_COUNT];
        for (int floor = 0; floor < result.length; floor++) {
            boolean[] mask = new boolean[MultiDiscreteActionSpace.TILE_COUNT];
            for (int tile = 0; tile < MultiDiscreteActionSpace.TILE_COUNT; tile++) {
                if (source[floor][tile]) {
                    addNeighborhood(mask, tile);
                }
            }
            result[floor] = maskToList(mask);
        }
        return result;
    }

    @SuppressWarnings("unchecked")
    private List<Integer>[] buildWallPlacementTileLists() {
        List<Integer>[] result = new List[MultiDiscreteActionSpace.FLOOR_COUNT];
        for (int floor = 0; floor < result.length; floor++) {
            boolean[] mask = new boolean[MultiDiscreteActionSpace.TILE_COUNT];
            addAll(mask, nearSurfaceTilesByFloor[floor]);
            if (floor > 0) {
                addAll(mask, nearWallTilesByFloor[floor - 1]);
            }
            result[floor] = maskToList(mask);
        }
        return result;
    }

    @SuppressWarnings("unchecked")
    private List<Integer>[] buildCeilingPlacementTileLists() {
        List<Integer>[] result = new List[MultiDiscreteActionSpace.FLOOR_COUNT];
        for (int floor = 0; floor < result.length; floor++) {
            if (floor > 0) {
                result[floor] = nearWallTilesByFloor[floor - 1];
            } else {
                result[floor] = Collections.emptyList();
            }
        }
        return result;
    }

    private static void addNeighborhood(boolean[] mask, int tile) {
        int tx = tile / MultiDiscreteActionSpace.GRID_SIZE;
        int ty = tile % MultiDiscreteActionSpace.GRID_SIZE;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = tx + dx;
                int ny = ty + dy;
                if (nx >= 0 && nx < MultiDiscreteActionSpace.GRID_SIZE
                        && ny >= 0 && ny < MultiDiscreteActionSpace.GRID_SIZE) {
                    mask[nx * MultiDiscreteActionSpace.GRID_SIZE + ny] = true;
                }
            }
        }
    }

    private static void addAll(boolean[] mask, List<Integer> tiles) {
        for (Integer tile : tiles) {
            if (tile != null && tile >= 0 && tile < mask.length) {
                mask[tile] = true;
            }
        }
    }

    private static List<Integer> maskToList(boolean[] mask) {
        List<Integer> tiles = new ArrayList<>();
        for (int i = 0; i < mask.length; i++) {
            if (mask[i]) {
                tiles.add(i);
            }
        }
        return Collections.unmodifiableList(tiles);
    }

    private static List<Integer> getFloorList(List<Integer>[] lists, int floor) {
        if (floor < 0 || floor >= lists.length) {
            return Collections.emptyList();
        }
        return lists[floor];
    }
}
