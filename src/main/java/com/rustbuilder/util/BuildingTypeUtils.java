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
     * Converts UI tool identifiers to their model type. Some older UI labels are
     * aliases for enum values and are normalized here.
     */
    public static BuildingType fromToolId(String toolId) {
        if (toolId == null) {
            return null;
        }

        switch (toolId) {
            case "TRIANGLE":
                return BuildingType.TRIANGLE_FOUNDATION;
            case "DOOR_FRAME":
                return BuildingType.DOORWAY;
            default:
                try {
                    return BuildingType.valueOf(toolId);
                } catch (IllegalArgumentException ex) {
                    return null;
                }
        }
    }

    /**
     * Converts a model type back to the currently expected UI/snapping tool id.
     */
    public static String toToolId(BuildingType type) {
        if (type == null) {
            return null;
        }

        switch (type) {
            case TRIANGLE_FOUNDATION:
                return "TRIANGLE";
            case DOORWAY:
                return "DOOR_FRAME";
            default:
                return type.name();
        }
    }

    /**
     * Returns true if the given tool string represents a wall-type placement.
     * Used for placement logic in GameController and SnappingService.
     */
    public static boolean isWallTool(String tool) {
        return isWall(fromToolId(tool));
    }
}
