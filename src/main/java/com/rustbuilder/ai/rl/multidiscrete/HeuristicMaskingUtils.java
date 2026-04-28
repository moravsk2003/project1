package com.rustbuilder.ai.rl.multidiscrete;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import com.rustbuilder.ai.ea.BaseGenome.BuildAction;
import com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType;
import com.rustbuilder.ai.rl.legacy.ActionSpace;
import com.rustbuilder.config.GameConstants;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
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
        List<Integer> baseTypes = ActionSpace.getValidTypeActions(hasTC, hasLootRoom, step);
        List<Integer> gated = new ArrayList<>();
        int blocks = grid.getAllBlocks().size();
        
        // Fast pre-check for ceiling rule
        boolean hasAtLeastOneWall = false;
        for (BuildingBlock b : grid.getAllBlocks()) {
            BuildingType bt = b.getType();
            if (bt == BuildingType.WALL || bt == BuildingType.DOORWAY || bt == BuildingType.WINDOW_FRAME) {
                hasAtLeastOneWall = true;
                break;
            }
        }
        
        for (Integer tIndex : baseTypes) {
            if (tIndex == com.rustbuilder.ai.rl.legacy.ActionSpace.STOP_TYPE_INDEX) {
                gated.add(tIndex);
                continue;
            }
            ActionType type = ActionSpace.decodeType(tIndex);
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

    /**
     * Phase 2: Get valid floors for the selected type.
     * Uses CHEAP structural heuristics (walls below, foundations at floor 0).
     */
    public static List<Integer> getValidFloors(GridModel grid, int typeIndex, int step) {
        // [PERF] Removed hasFeasibleTile scan. Just return basic structurally-valid floors.
        return getBasicFloors(grid, typeIndex);
    }

    /**
     * Phase 3: Get valid tiles based on proximity and structural heuristics.
     * Uses CHEAP pre-filtering without rotation/aim dry-runs.
     */
    public static List<Integer> getValidTiles(GridModel grid, int typeIndex, int floorIndex, int step) {
        List<Integer> valid = new ArrayList<>();
        ActionType type = ActionSpace.decodeType(typeIndex);
        if (type == null) return Collections.singletonList(0); // STOP

        if (step == 0) {
            // First step must be in the center
            addCenterTiles(valid);
        } else {
            // Check all tiles for structural proximity
            for (int tileIdx = 0; tileIdx < MultiDiscreteActionSpace.TILE_COUNT; tileIdx++) {
                int tx = tileIdx / GRID_SIZE;
                int ty = tileIdx % GRID_SIZE;

                if (passesNearStructureRule(grid, type, tx, ty, floorIndex)) {
                    valid.add(tileIdx);
                }
            }
        }

        if (valid.isEmpty()) {
            emptyTilesCount++;
        }

        return valid;
    }

    /**
     * Phase 4: Get valid rotations (0..3).
     * No longer scans aim sectors or performs dry-runs.
     */
    public static List<Integer> getValidRotations(GridModel grid, int typeIndex, int floorIndex, int tileIndex) {
        ActionType type = ActionSpace.decodeType(typeIndex);
        if (type == null) return Collections.singletonList(0); // STOP

        // [PERF] For structural and deployable types, the orientation (rotation index)
        // does not determine placement feasibility in the current engine. 
        // Foundation/floor snapping and deployable centering are handled internally.
        if (isRotationInvariant(type)) {
            return Collections.singletonList(0);
        }

        // Return all 4 rotations for anything else (walls, doors, etc.)
        List<Integer> rots = new ArrayList<>();
        for (int i = 0; i < 4; i++) rots.add(i);
        return rots;
    }

    /**
     * Phase 5: Get valid aim sectors (0..24) for the selected type, floor, tile, and rotation.
     * THIS IS THE ONLY PHASE that performs exact physics feasibility checks (dry-runs).
     */
    public static List<Integer> getValidAimSectors(GridModel grid, int typeIndex, int floorIndex, int tileIndex, int rotationIndex) {
        List<Integer> valid = new ArrayList<>();
        ActionType type = ActionSpace.decodeType(typeIndex);
        if (type == null) return Collections.singletonList(12); // Default center for STOP

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

    private static final int[] SECTOR_OPTIMIZED_ORDER = {
        12, // Center
        6, 7, 8, 11, 13, 16, 17, 18, // Ring 1 (neighbors of 12)
        0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24 // Ring 2 (edges)
    };

    // --- Private Helpers ---

    private static void addCenterTiles(List<Integer> list) {
        list.add((GRID_SIZE / 2 - 1) * GRID_SIZE + (GRID_SIZE / 2 - 1));
        list.add((GRID_SIZE / 2 - 1) * GRID_SIZE + (GRID_SIZE / 2));
        list.add((GRID_SIZE / 2) * GRID_SIZE + (GRID_SIZE / 2 - 1));
        list.add((GRID_SIZE / 2) * GRID_SIZE + (GRID_SIZE / 2));
    }

    /**
     * Cheap structural tile pruning.
     * Blocks must be near other blocks on the same floor or a wall below.
     */
    private static boolean passesNearStructureRule(GridModel grid, ActionType type, int tx, int ty, int floorIndex) {
        if (isFoundationType(type)) return true; // Foundations start the base

        double x = 200.0 + tx * GameConstants.TILE_SIZE;
        double y = 200.0 + ty * GameConstants.TILE_SIZE;
        double radius = GameConstants.TILE_SIZE * 1.2;

        List<BuildingBlock> near = grid.getNearbyBlocks(x, y, floorIndex, radius);
        
        boolean hasSameFloorAnyBlock = false;
        for (BuildingBlock b : near) {
            if (b.getZ() == floorIndex) {
                hasSameFloorAnyBlock = true;
                break;
            }
        }
        if (hasSameFloorAnyBlock) return true;

        // If no same-floor neighbors, we MUST have a wall/support below
        if (floorIndex > 0) {
            // Reuse the nearby query if possible (getNearbyBlocks with z=floorIndex usually gets +/- 1 floor range depending on implementation, 
            // but here we assume it's exact or we need another query).
            // Optimization: if GridModel.getNearbyBlocks is already floor-aware, we might need a separate call for below.
            List<BuildingBlock> below = grid.getNearbyBlocks(x, y, floorIndex - 1, radius);
            for (BuildingBlock b : below) {
                if (b.getZ() == floorIndex - 1) {
                    BuildingType bt = b.getType();
                    if (bt == BuildingType.WALL || bt == BuildingType.DOORWAY || bt == BuildingType.WINDOW_FRAME) {
                        return true;
                    }
                }
            }
        }

        tilesRejectedByNearStructureRule++;
        return false;
    }

    private static List<Integer> getBasicFloors(GridModel grid, int typeIndex) {
        ActionType type = ActionSpace.decodeType(typeIndex);
        if (type == null) return Collections.singletonList(0);

        if (isFoundationType(type)) return Collections.singletonList(0);

        // Pre-scan grid to avoid multiple passes
        boolean[] hasWallAtFloor = new boolean[MAX_FLOORS];
        boolean hasAnyBlockAtFloor0 = false;

        for (BuildingBlock b : grid.getAllBlocks()) {
            int z = b.getZ();
            if (z >= 0 && z < MAX_FLOORS) {
                if (z == 0) hasAnyBlockAtFloor0 = true;
                BuildingType bt = b.getType();
                if (bt == BuildingType.WALL || bt == BuildingType.DOORWAY || bt == BuildingType.WINDOW_FRAME) {
                    hasWallAtFloor[z] = true;
                }
            }
        }

        List<Integer> valid = new ArrayList<>();
        // Floor 0 is always base floor
        if (!isCeilingType(type) && hasAnyBlockAtFloor0) {
            valid.add(0);
        }
        // Higher floors allowed if there's a wall below
        for (int f = 1; f < MAX_FLOORS; f++) {
            if (hasWallAtFloor[f - 1]) valid.add(f);
        }

        if (valid.isEmpty()) valid.add(0);
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

    private static boolean isRotationInvariant(ActionType type) {
        // These types either have 4-fold symmetry or their placement engine 
        // ignores the input rotationIndex in favor of internal snapping/centering.
        return type == ActionType.FOUNDATION || 
               type == ActionType.TRIANGLE_FOUNDATION ||
               type == ActionType.FLOOR ||
               type == ActionType.TRIANGLE_FLOOR ||
               type == ActionType.TC ||
               type == ActionType.WORKBENCH ||
               type == ActionType.LOOT_ROOM;
    }
}
