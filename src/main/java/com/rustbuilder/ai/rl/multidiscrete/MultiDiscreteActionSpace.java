package com.rustbuilder.ai.rl.multidiscrete;

import com.rustbuilder.ai.ea.BaseGenome.BuildAction;
import com.rustbuilder.ai.ea.BaseGenome.BuildAction.ActionType;
import java.util.ArrayList;
import java.util.List;

/**
 * Specification and constants for the new multi-discrete action space.
 * 
 * PHASES:
 * 1. Type      (11)
 * 2. Floor     (8)
 * 3. Tile      (64)
 * 4. Rotation  (6) - triangle-capable 60 degree steps
 * 5. AimSector (16) - 4x4 grid
 */
public class MultiDiscreteActionSpace {

    public static final int PHASE_COUNT = 5;

    public static final int TYPE_COUNT = 11;
    public static final int STOP_TYPE_INDEX = 10;
    public static final int MIN_BLOCKS_BEFORE_STOP = 5;
    public static final int FLOOR_COUNT = 8;
    public static final int GRID_SIZE = 8;
    public static final int TILE_COUNT = GRID_SIZE * GRID_SIZE; // 64
    public static final int ROTATION_COUNT = 6; // 0..5, step 60 degrees for triangle surfaces
    public static final int AIM_GRID_SIZE = 4;
    public static final int AIM_SECTOR_COUNT = AIM_GRID_SIZE * AIM_GRID_SIZE; // 4x4 grid
    public static final int DEFAULT_AIM_SECTOR = 5;

    /**
     * Get sizes for each phase of the multi-discrete action space.
     */
    public static int[] getPhaseSizes() {
        return new int[] {
            TYPE_COUNT,
            FLOOR_COUNT,
            TILE_COUNT,
            ROTATION_COUNT,
            AIM_SECTOR_COUNT
        };
    }

    /**
     * Map a rotation index (0..5) to its degree equivalent (0..300).
     */
    public static double rotationIndexToDegrees(int index) {
        if (index < 0 || index >= ROTATION_COUNT) return 0;
        return index * 60.0;
    }

    /**
     * Check if a full MultiDiscreteAction is within valid bounds.
     */
    public static boolean isValid(MultiDiscreteAction action) {
        if (action == null) return false;
        return action.isValid();
    }

    /**
     * Check if a specific phase value is within bounds.
     */
    public static boolean isPhaseValueValid(int phaseIndex, int value) {
        int[] sizes = getPhaseSizes();
        if (phaseIndex < 0 || phaseIndex >= sizes.length) return false;
        return value >= 0 && value < sizes[phaseIndex];
    }

    // ==================== Type Encoding (migrated from legacy ActionSpace) ====================

    /**
     * Decode a type index [0..10] to an ActionType, or null for STOP.
     */
    public static ActionType decodeType(int typeIndex) {
        if (typeIndex == STOP_TYPE_INDEX) return null;
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
     * Get valid type indices for Phase 1 (type selection).
     */
    public static List<Integer> getValidTypeActions(boolean hasTC, boolean hasLootRoom, int currentStep) {
        List<Integer> valid = new ArrayList<>(TYPE_COUNT);

        for (int i = 0; i < TYPE_COUNT - 1; i++) {
            ActionType type = ActionType.values()[i];

            if (currentStep == 0) {
                if (type != ActionType.FOUNDATION && type != ActionType.TRIANGLE_FOUNDATION) continue;
            } else {
                if (type == ActionType.TC && hasTC) continue;
                if (type == ActionType.LOOT_ROOM && hasLootRoom) continue;
            }

            valid.add(i);
        }

        if (currentStep >= MIN_BLOCKS_BEFORE_STOP) {
            valid.add(STOP_TYPE_INDEX);
        }

        return valid;
    }
}
