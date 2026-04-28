package com.rustbuilder.ai.rl.multidiscrete;

/**
 * Specification and constants for the new multi-discrete action space.
 * 
 * PHASES:
 * 1. Type      (11)
 * 2. Floor     (8)
 * 3. Tile      (64)
 * 4. Rotation  (4) - 90 degree steps
 * 5. AimSector (25) - 5x5 grid
 */
public class MultiDiscreteActionSpace {

    public static final int PHASE_COUNT = 5;

    public static final int TYPE_COUNT = 11;
    public static final int FLOOR_COUNT = 8;
    public static final int GRID_SIZE = 8;
    public static final int TILE_COUNT = GRID_SIZE * GRID_SIZE; // 64
    public static final int ROTATION_COUNT = 4; // 0..3, step 90°
    public static final int AIM_SECTOR_COUNT = 25; // 5x5 grid

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
     * Map a rotation index (0..3) to its degree equivalent (0..270).
     */
    public static double rotationIndexToDegrees(int index) {
        if (index < 0 || index >= ROTATION_COUNT) return 0;
        return index * 90.0;
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
}
