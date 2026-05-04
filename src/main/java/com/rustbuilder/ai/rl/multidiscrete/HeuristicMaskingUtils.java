package com.rustbuilder.ai.rl.multidiscrete;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import com.rustbuilder.ai.ea.BaseGenome.BuildAction;
import com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.util.GridPlacementUtils;

/**
 * Provides CHEAP heuristic masking to prune obviously invalid actions during multi-discrete selection.
 * 
 * REFACTOR GOAL:
 * - Phase 1 (Type), Phase 2 (Floor), Phase 3 (Tile), Phase 4 (Rotation) use ONLY cheap heuristics.
 * - Phase 5 (Aim Sector) is the ONLY phase that performs exact physics feasibility checks (dry-runs).
 * - Avoids "continuation search" (don't check if downstream phases are feasible from an upstream phase).
 * 
 * SEMANTICS:
 * - Rotation (Phase 4): Defines discrete block orientation (critical for Walls/Doors).
 * - Aim (Phase 5): Defines fine-grained sub-tile placement offset (critical for all types).
 */
public class HeuristicMaskingUtils {

    private static final int GRID_SIZE = MultiDiscreteActionSpace.GRID_SIZE;
    private static final int MAX_FLOORS = MultiDiscreteActionSpace.FLOOR_COUNT;
    private static final List<Integer> ROTATION_ZERO = Collections.singletonList(0);
    private static final List<Integer> CARDINAL_ROTATIONS = Collections.unmodifiableList(Arrays.asList(0, 1, 2, 3));
    private static final List<Integer> TRIANGLE_ROTATIONS = Collections.unmodifiableList(Arrays.asList(0, 1, 2, 3, 4, 5));
    private static final List<Integer> FLOOR_ZERO = Collections.singletonList(0);
    private static final List<Integer> STOP_TILE = Collections.singletonList(0);
    private static final List<Integer> STOP_AIM = Collections.singletonList(MultiDiscreteActionSpace.DEFAULT_AIM_SECTOR);
    private static final int FALLBACK_AIM_CHECK_BUDGET = 96;
    private static final List<Integer> CENTER_TILES = Collections.unmodifiableList(Arrays.asList(
        (GRID_SIZE / 2 - 1) * GRID_SIZE + (GRID_SIZE / 2 - 1),
        (GRID_SIZE / 2 - 1) * GRID_SIZE + (GRID_SIZE / 2),
        (GRID_SIZE / 2) * GRID_SIZE + (GRID_SIZE / 2 - 1),
        (GRID_SIZE / 2) * GRID_SIZE + (GRID_SIZE / 2)
    ));

    public static boolean DEBUG_MODE = false;

    // Debug counters for dead branches
    public static int prunedTypeCount = 0;
    public static int emptyTilesCount = 0;
    public static int emptyRotationsCount = 0;
    public static int rotationsRejectedByAimCount = 0;
    public static int tilesRejectedByNearStructureRule = 0;

    /**
     * Phase 1: Get globally valid building types for the current step.
     * Uses CHEAP gating rules (TC/Loot limits, ceiling-wall dependencies).
     */
    public static List<Integer> getValidTypes(GridModel grid, boolean hasTC, boolean hasLootRoom, int step) {
        return getValidTypes(new MultiDiscretePhaseContext(grid, hasTC, hasLootRoom, step, step + 1));
    }

    public static List<Integer> getValidTypes(MultiDiscretePhaseContext context) {
        List<Integer> baseTypes = MultiDiscreteActionSpace.getValidTypeActions(
            context.isHasTC(),
            context.isHasLootRoom(),
            context.getStep()
        );
        List<Integer> gated = new ArrayList<>();
        int blocks = context.getBlockCount();
        boolean hasAtLeastOneWall = context.hasAnyWall();
        
        for (Integer tIndex : baseTypes) {
            if (tIndex == MultiDiscreteActionSpace.STOP_TYPE_INDEX) {
                if (blocks >= MultiDiscreteActionSpace.MIN_BLOCKS_BEFORE_STOP) {
                    gated.add(tIndex);
                }
                continue;
            }
            ActionType type = MultiDiscreteActionSpace.decodeType(tIndex);
            if (type == null) {
                gated.add(tIndex);
                continue;
            }
            
            // Rule: No ceilings (floors) without at least one wall in the base
            if (isCeilingType(type) && !hasAtLeastOneWall) {
                continue;
            }
            
            // Rule: Start with foundations
            if (blocks < 2) {
                if (isFoundationType(type)) {
                    gated.add(tIndex);
                }
            } else {
                gated.add(tIndex);
            }
        }
        return gated;
    }

    /**
     * Phase 1 (Feasible): Alias for getValidTypes.
     * No longer performs expensive continuation scanning.
     */
    public static List<Integer> getFeasibleTypes(GridModel grid, boolean hasTC, boolean hasLootRoom, int step) {
        // [PERF] Removed hasFeasibleContinuation loop. Just use cheap gating.
        return getValidTypes(grid, hasTC, hasLootRoom, step);
    }

    public static List<Integer> getFeasibleTypes(MultiDiscretePhaseContext context) {
        return getValidTypes(context);
    }

    /**
     * Phase 2: Get valid floors for the selected type.
     * Uses CHEAP structural heuristics (walls below, foundations at floor 0).
     */
    public static List<Integer> getValidFloors(GridModel grid, int typeIndex, int step) {
        // [PERF] Removed hasFeasibleTile scan. Just return basic structurally-valid floors.
        return getBasicFloors(new MultiDiscretePhaseContext(grid, false, false, step, step + 1), typeIndex);
    }

    public static List<Integer> getValidFloors(MultiDiscretePhaseContext context, int typeIndex) {
        return getBasicFloors(context, typeIndex);
    }

    /**
     * Phase 3: Get valid tiles based on proximity and structural heuristics.
     * Uses CHEAP pre-filtering without rotation/aim dry-runs.
     */
    public static List<Integer> getValidTiles(GridModel grid, int typeIndex, int floorIndex, int step) {
        return getValidTiles(new MultiDiscretePhaseContext(grid, false, false, step, step + 1), typeIndex, floorIndex);
    }

    public static List<Integer> getValidTiles(MultiDiscretePhaseContext context, int typeIndex, int floorIndex) {
        List<Integer> valid = new ArrayList<>();
        ActionType type = MultiDiscreteActionSpace.decodeType(typeIndex);
        if (type == null) return STOP_TILE; // STOP

        if (context.getStep() == 0) {
            return CENTER_TILES;
        }

        if (isFoundationType(type)) {
            for (int tileIdx = 0; tileIdx < MultiDiscreteActionSpace.TILE_COUNT; tileIdx++) {
                valid.add(tileIdx);
            }
            return valid;
        }

        if (isWallLikeType(type)) {
            valid.addAll(context.getWallPlacementTiles(floorIndex));
        } else if (isCeilingType(type)) {
            valid.addAll(context.getCeilingPlacementTiles(floorIndex));
        } else if (isFurnitureType(type)) {
            valid.addAll(context.getSurfaceTiles(floorIndex));
        } else {
            valid.addAll(context.getNearSurfaceTiles(floorIndex));
        }

        if (valid.isEmpty()) {
            emptyTilesCount++;
        } else {
            tilesRejectedByNearStructureRule += MultiDiscreteActionSpace.TILE_COUNT - valid.size();
        }

        return valid;
    }

    /**
     * Phase 4: Get valid rotations.
     * No longer scans aim sectors or performs dry-runs.
     */
    public static List<Integer> getValidRotations(GridModel grid, int typeIndex, int floorIndex, int tileIndex) {
        ActionType type = MultiDiscreteActionSpace.decodeType(typeIndex);
        if (type == null) return ROTATION_ZERO; // STOP

        // [PERF] For structural and deployable types, the orientation (rotation index)
        // does not determine placement feasibility in the current engine. 
        // Foundation/floor snapping and deployable centering are handled internally.
        if (isRotationInvariant(type)) {
            return ROTATION_ZERO;
        }

        if (isTriangleType(type)) {
            return TRIANGLE_ROTATIONS;
        }

        return CARDINAL_ROTATIONS;
    }

    /**
     * Phase 5: Get valid aim sectors for the selected type, floor, tile, and rotation.
     * THIS IS THE ONLY PHASE that performs exact physics feasibility checks (dry-runs).
     */
    public static List<Integer> getValidAimSectors(GridModel grid, int typeIndex, int floorIndex, int tileIndex, int rotationIndex) {
        List<Integer> valid = new ArrayList<>();
        ActionType type = MultiDiscreteActionSpace.decodeType(typeIndex);
        if (type == null) return STOP_AIM; // Default center for STOP

        int tx = tileIndex / GRID_SIZE;
        int ty = tileIndex % GRID_SIZE;

        // Final gate: EXACT check via placement engine.
        // Check sectors in optimized order: center -> ring1 -> edges.
        for (int sector : SECTOR_OPTIMIZED_ORDER) {
            BuildAction trial = 
                new BuildAction(type, tx, ty, floorIndex, rotationIndex, 2, 0, sector);
            
            if (GridPlacementUtils.isActionActuallyFeasible(grid, trial)) {
                valid.add(sector);
            }
        }

        return valid;
    }

    public static int getFirstValidAimSector(GridModel grid, int typeIndex, int floorIndex, int tileIndex, int rotationIndex) {
        ActionType type = MultiDiscreteActionSpace.decodeType(typeIndex);
        if (type == null) return MultiDiscreteActionSpace.DEFAULT_AIM_SECTOR;

        int tx = tileIndex / GRID_SIZE;
        int ty = tileIndex % GRID_SIZE;

        for (int sector : SECTOR_OPTIMIZED_ORDER) {
            BuildAction trial = new BuildAction(type, tx, ty, floorIndex, rotationIndex, 2, 0, sector);
            if (GridPlacementUtils.isActionActuallyFeasible(grid, trial)) {
                return sector;
            }
        }

        return -1;
    }

    public static MultiDiscreteAction findFeasibleBuildAction(GridModel grid, boolean hasTC, boolean hasLootRoom, int step, java.util.Random random) {
        return findFeasibleBuildAction(new MultiDiscretePhaseContext(grid, hasTC, hasLootRoom, step, step + 1), random);
    }

    public static MultiDiscreteAction findFeasibleBuildAction(MultiDiscretePhaseContext context, java.util.Random random) {
        GridModel grid = context.getGrid();
        List<Integer> types = shuffled(getFeasibleTypes(context), random);
        types.remove(Integer.valueOf(MultiDiscreteActionSpace.STOP_TYPE_INDEX));
        int aimChecks = 0;

        for (int type : types) {
            List<Integer> floors = shuffled(getValidFloors(context, type), random);
            for (int floor : floors) {
                List<Integer> tiles = shuffled(getValidTiles(context, type, floor), random);
                for (int tile : tiles) {
                    List<Integer> rotations = shuffled(getValidRotations(grid, type, floor, tile), random);
                    for (int rotation : rotations) {
                        if (aimChecks++ >= FALLBACK_AIM_CHECK_BUDGET) {
                            return null;
                        }
                        int aim = getFirstValidAimSector(grid, type, floor, tile, rotation);
                        if (aim >= 0) {
                            return new MultiDiscreteAction(type, floor, tile, rotation, aim);
                        }
                    }
                }
            }
        }

        return null;
    }

    private static List<Integer> shuffled(List<Integer> values, java.util.Random random) {
        List<Integer> copy = new ArrayList<>(values);
        if (random != null && copy.size() > 1) {
            Collections.shuffle(copy, random);
        }
        return copy;
    }

    private static final int[] SECTOR_OPTIMIZED_ORDER = {
        5, 6, 9, 10, // Central 2x2 sectors
        1, 2, 4, 7, 8, 11, 13, 14, // Edge-adjacent sectors
        0, 3, 12, 15 // Corners
    };

    // --- Private Helpers ---

    private static List<Integer> getBasicFloors(MultiDiscretePhaseContext context, int typeIndex) {
        ActionType type = MultiDiscreteActionSpace.decodeType(typeIndex);
        if (type == null) return FLOOR_ZERO;

        if (isFoundationType(type)) return FLOOR_ZERO;

        List<Integer> valid = new ArrayList<>();

        if (isCeilingType(type)) {
            for (int f = 1; f < MAX_FLOORS; f++) {
                if (context.hasWallAtFloor(f - 1)) {
                    valid.add(f);
                }
            }
            return valid;
        }

        if (isWallLikeType(type)) {
            for (int f = 0; f < MAX_FLOORS; f++) {
                if (context.hasHorizontalAtFloor(f)
                        || (f > 0 && context.hasWallAtFloor(f - 1))) {
                    valid.add(f);
                }
            }
            return valid;
        }

        if (isFurnitureType(type)) {
            for (int f = 0; f < MAX_FLOORS; f++) {
                if (context.hasHorizontalAtFloor(f)) {
                    valid.add(f);
                }
            }
            return valid;
        }

        return valid;
    }

    private static boolean isFoundationType(ActionType type) {
        return type == ActionType.FOUNDATION || 
               type == ActionType.TRIANGLE_FOUNDATION;
    }

    private static boolean isCeilingType(ActionType type) {
        return type == ActionType.FLOOR || 
               type == ActionType.TRIANGLE_FLOOR;
    }

    private static boolean isWallLikeType(ActionType type) {
        return type == ActionType.WALL ||
               type == ActionType.DOORWAY ||
               type == ActionType.WINDOW_FRAME;
    }

    private static boolean isFurnitureType(ActionType type) {
        return type == ActionType.TC ||
               type == ActionType.WORKBENCH ||
               type == ActionType.LOOT_ROOM;
    }

    private static boolean isRotationInvariant(ActionType type) {
        // These types either have 4-fold symmetry or their placement engine 
        // ignores the input rotationIndex in favor of internal snapping/centering.
        return type == ActionType.FOUNDATION || 
               type == ActionType.FLOOR ||
               type == ActionType.TC ||
               type == ActionType.WORKBENCH ||
               type == ActionType.LOOT_ROOM;
    }

    private static boolean isTriangleType(ActionType type) {
        return type == ActionType.TRIANGLE_FOUNDATION ||
               type == ActionType.TRIANGLE_FLOOR;
    }
}
