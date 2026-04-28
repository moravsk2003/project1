package com.rustbuilder.ai.rl.legacy;

import com.rustbuilder.model.*;
import com.rustbuilder.model.core.*;
import com.rustbuilder.model.structure.*;
import com.rustbuilder.model.deployable.*;
import com.rustbuilder.ai.ea.BaseGenome.BuildAction;
import com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType;
import com.rustbuilder.util.GridPlacementUtils;
import java.util.ArrayList;
import java.util.List;

/**
 * Two-phase hierarchical action space for the RL Agent.
 * 
 * Phase 1 (Type Selection):  indices [0..10]  — 10 block types + STOP
 * Phase 2 (Position Selection): indices [11..2058] — 8x * 8y * 8z * 4orient = 2048
 * Total output neurons: 2059
 */
public class ActionSpace {

    // Grid dimensions
    private static final int GRID_SIZE = 8;
    private static final int MAX_FLOORS = 8;
    private static final int ORIENT_COUNT = 4;

    // Phase 1: Type selection (10 types + STOP)
    public static final int TYPE_COUNT = 11;  // ActionType.values().length + 1 for STOP
    public static final int STOP_TYPE_INDEX = 10;

    // Phase 2: Position selection
    public static final int POSITION_COUNT = GRID_SIZE * GRID_SIZE * MAX_FLOORS * ORIENT_COUNT; // 2048
    public static final int POSITION_OFFSET = TYPE_COUNT; // positions start at index 11

    // Total action space
    public static final int TOTAL_ACTIONS = TYPE_COUNT + POSITION_COUNT; // 2059

    // ==================== Phase 1: Type ====================

    /**
     * Decode a type index [0..10] to an ActionType, or null for STOP.
     */
    public static ActionType decodeType(int typeIndex) {
        if (typeIndex == STOP_TYPE_INDEX) return null; // STOP
        ActionType[] types = ActionType.values();
        if (typeIndex >= 0 && typeIndex < types.length) {
            return types[typeIndex];
        }
        return null;
    }

    /**
     * Encode an ActionType to its type index [0..9], or STOP_TYPE_INDEX for null/STOP.
     */
    public static int encodeType(ActionType type) {
        if (type == null) return STOP_TYPE_INDEX;
        return type.ordinal();
    }

    /**
     * Get valid type indices for Phase 1.
     */
    public static List<Integer> getValidTypeActions(boolean hasTC, boolean hasLootRoom, int currentStep) {
        List<Integer> valid = new ArrayList<>(TYPE_COUNT);

        for (int i = 0; i < TYPE_COUNT - 1; i++) { // 0..9 (skip STOP, add it at end)
            ActionType type = ActionType.values()[i];

            if (currentStep == 0) {
                // First step: only foundations
                if (type != ActionType.FOUNDATION && type != ActionType.TRIANGLE_FOUNDATION) continue;
            } else {
                if (type == ActionType.TC && hasTC) continue;
                if (type == ActionType.LOOT_ROOM && hasLootRoom) continue;
            }

            valid.add(i);
        }

        // Always allow STOP (except on first step)
        if (currentStep > 0) {
            valid.add(STOP_TYPE_INDEX);
        }

        return valid;
    }

    // ==================== Phase 2: Position ====================

    /**
     * Decode a position index [11..2058] to a BuildAction with the given type.
     */
    public static BuildAction decodePosition(int posIndex, ActionType type) {
        int local = posIndex - POSITION_OFFSET; // Convert to 0-based [0..2047]

        int orient = local % ORIENT_COUNT; local /= ORIENT_COUNT;
        int z = local % MAX_FLOORS; local /= MAX_FLOORS;
        int y = local % GRID_SIZE; local /= GRID_SIZE;
        int x = local % GRID_SIZE;

        int tier = 2; // Default STONE
        int doorType = 0; // Default SHEET_METAL

        return new BuildAction(type, x, y, z, orient, tier, doorType);
    }

    /**
     * Encode a BuildAction's position to its position index [11..2058].
     */
    public static int encodePosition(BuildAction action) {
        return POSITION_OFFSET +
               (((action.gridX * GRID_SIZE + action.gridY)
                 * MAX_FLOORS + action.floor)
                 * ORIENT_COUNT + action.orientation);
    }

    /**
     * Get truly valid position indices for Phase 2, using GridPlacementUtils for real validation.
     * Falls back to heuristic filtering if no positions pass real validation.
     */
    public static List<Integer> getValidPositionActions(ActionType type, int currentStep, GridModel grid) {
        List<Integer> valid = new ArrayList<>(256);

        for (int posIdx = POSITION_OFFSET; posIdx < TOTAL_ACTIONS; posIdx++) {
            int local = posIdx - POSITION_OFFSET;
            int z = (local / ORIENT_COUNT) % MAX_FLOORS;

            // Quick heuristic pre-filters (cheap, before expensive placement check)
            if (type == ActionType.FOUNDATION || type == ActionType.TRIANGLE_FOUNDATION) {
                if (z != 0) continue; // Foundations only on ground
                if (currentStep == 0) {
                    int y = (local / (ORIENT_COUNT * MAX_FLOORS)) % GRID_SIZE;
                    int x = (local / (ORIENT_COUNT * MAX_FLOORS * GRID_SIZE)) % GRID_SIZE;
                    if (x < 3 || x > 4 || y < 3 || y > 4) continue; // First step: center only
                }
            }
            if (z > 1 && currentStep < 5) continue; // No high builds early

            // Real validation via GridPlacementUtils
            BuildAction trial = decodePosition(posIdx, type);
            GridPlacementUtils.Placement placement = GridPlacementUtils.calculatePlacement(grid, trial);
            if (placement.valid) {
                // Final definitive check: does it actually fit in the grid without collision?
                if (simulateCanPlace(grid, trial, placement)) {
                    valid.add(posIdx);
                }
            }
        }

        // If no valid positions found, return a small subset for exploration (agent will get -0.05 penalty)
        if (valid.isEmpty()) {
            for (int posIdx = POSITION_OFFSET; posIdx < POSITION_OFFSET + 32; posIdx++) {
                valid.add(posIdx);
            }
        }

        return valid;
    }

    /**
     * Check if an action index belongs to Phase 1 (type selection).
     */
    public static boolean isTypeAction(int index) {
        return index >= 0 && index < TYPE_COUNT;
    }

    /**
     * Check if an action index belongs to Phase 2 (position selection).
     */
    public static boolean isPositionAction(int index) {
        return index >= POSITION_OFFSET && index < TOTAL_ACTIONS;
    }

    /**
     * Dry-run simulation of RLTrainingService.placeBlock to guarantee a placement
     * won't immediately fail due to collision or invalid support.
     */
    private static boolean simulateCanPlace(GridModel gridModel, BuildAction action, GridPlacementUtils.Placement placement) {
        double finalX = placement.x;
        double finalY = placement.y;
        double finalRotation = placement.rotation;
        Orientation finalOrientation = placement.orientation;
        int z = (action.actionType == ActionType.FOUNDATION || action.actionType == ActionType.TRIANGLE_FOUNDATION) ? 0 : action.floor;
        
        BuildingTier tier = BuildingTier.STONE;
        DoorType doorType = DoorType.SHEET_METAL;
        BuildingBlock block = null;

        switch (action.actionType) {
            case FOUNDATION: block = new Foundation(finalX, finalY, z); block.setRotation(finalRotation); break;
            case TRIANGLE_FOUNDATION: block = new TriangleFoundation(finalX, finalY, z, finalRotation); break;
            case WALL: block = new Wall(finalX, finalY, z, finalOrientation); break;
            case DOORWAY:
                Wall dw = new Wall(finalX, finalY, z, finalOrientation);
                dw.setType(BuildingType.DOORWAY);
                dw.setDoorType(doorType);
                block = dw;
                break;
            case WINDOW_FRAME:
                Wall wf = new Wall(finalX, finalY, z, finalOrientation);
                wf.setType(BuildingType.WINDOW_FRAME);
                block = wf;
                break;
            case FLOOR: block = new Floor(finalX, finalY, z, finalRotation); break;
            case TRIANGLE_FLOOR: block = new TriangleFloor(finalX, finalY, z, finalRotation); break;
            case TC: block = new ToolCupboard(finalX, finalY, z, finalRotation); break;
            case WORKBENCH: block = new Workbench(finalX, finalY, z, finalRotation); break;
            case LOOT_ROOM: block = new LootRoom(finalX, finalY, z, finalRotation); break;
        }

        if (block != null) {
            if (!(block instanceof Wall)) {
                block.setRotation(finalRotation);
            }
            block.setTier(tier);
            return gridModel.canPlace(block);
        }
        
        return false;
    }
}
