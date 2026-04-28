package com.rustbuilder.util;

import com.rustbuilder.model.core.BuildingType;

/**
 * Utility class for categorizing BuildingType values.
 * Centralizes type-check logic that was previously duplicated across
 * GameController, SnappingService, GridModel, HouseGraph, and StabilityService.
 */
public final class BuildingTypeUtils {

    private BuildingTypeUtils() {}

    /** Returns true for WALL, DOORWAY, and WINDOW_FRAME. */
    public static boolean isWall(BuildingType type) {
        return type == BuildingType.WALL
            || type == BuildingType.DOORWAY
            || type == BuildingType.WINDOW_FRAME;
    }

    /** Returns true for FOUNDATION and TRIANGLE_FOUNDATION. */
    public static boolean isFoundation(BuildingType type) {
        return type == BuildingType.FOUNDATION
            || type == BuildingType.TRIANGLE_FOUNDATION;
    }

    /** Returns true for FLOOR and TRIANGLE_FLOOR. */
    public static boolean isFloor(BuildingType type) {
        return type == BuildingType.FLOOR
            || type == BuildingType.TRIANGLE_FLOOR;
    }

    /** Returns true for any horizontal surface: foundation or floor. */
    public static boolean isHorizontalSurface(BuildingType type) {
        return isFoundation(type) || isFloor(type);
    }

    /** Returns true for utility/furniture blocks: TC, WORKBENCH, LOOT_ROOM. */
    public static boolean isFurniture(BuildingType type) {
        return type == BuildingType.TC
            || type == BuildingType.WORKBENCH
            || type == BuildingType.LOOT_ROOM;
    }

    /**
     * Returns true if the given tool string represents a wall-type placement.
     * Used for placement logic in GameController and SnappingService.
     */
    public static boolean isWallTool(String tool) {
        return "WALL".equals(tool)
            || "DOOR_FRAME".equals(tool)
            || "WINDOW_FRAME".equals(tool);
    }
}
