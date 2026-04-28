package com.rustbuilder.util;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

import com.rustbuilder.model.core.BuildingType;

class BuildingTypeUtilsTest {

    @Test
    void isWall_returnsTrue_forWallTypes() {
        assertTrue(BuildingTypeUtils.isWall(BuildingType.WALL));
        assertTrue(BuildingTypeUtils.isWall(BuildingType.DOORWAY));
        assertTrue(BuildingTypeUtils.isWall(BuildingType.WINDOW_FRAME));
    }

    @Test
    void isWall_returnsFalse_forNonWallTypes() {
        assertFalse(BuildingTypeUtils.isWall(BuildingType.FOUNDATION));
        assertFalse(BuildingTypeUtils.isWall(BuildingType.FLOOR));
        assertFalse(BuildingTypeUtils.isWall(BuildingType.TC));
    }

    @Test
    void isFoundation_returnsTrue_forFoundationTypes() {
        assertTrue(BuildingTypeUtils.isFoundation(BuildingType.FOUNDATION));
        assertTrue(BuildingTypeUtils.isFoundation(BuildingType.TRIANGLE_FOUNDATION));
    }

    @Test
    void isFoundation_returnsFalse_forOtherTypes() {
        assertFalse(BuildingTypeUtils.isFoundation(BuildingType.WALL));
        assertFalse(BuildingTypeUtils.isFoundation(BuildingType.FLOOR));
    }

    @Test
    void isFloor_returnsTrue_forFloorTypes() {
        assertTrue(BuildingTypeUtils.isFloor(BuildingType.FLOOR));
        assertTrue(BuildingTypeUtils.isFloor(BuildingType.TRIANGLE_FLOOR));
    }

    @Test
    void isHorizontalSurface_returnsTrueForFoundationsAndFloors() {
        assertTrue(BuildingTypeUtils.isHorizontalSurface(BuildingType.FOUNDATION));
        assertTrue(BuildingTypeUtils.isHorizontalSurface(BuildingType.TRIANGLE_FOUNDATION));
        assertTrue(BuildingTypeUtils.isHorizontalSurface(BuildingType.FLOOR));
        assertTrue(BuildingTypeUtils.isHorizontalSurface(BuildingType.TRIANGLE_FLOOR));
        assertFalse(BuildingTypeUtils.isHorizontalSurface(BuildingType.WALL));
    }

    @Test
    void isFurniture_returnsTrue_forFurnitureBlocks() {
        assertTrue(BuildingTypeUtils.isFurniture(BuildingType.TC));
        assertTrue(BuildingTypeUtils.isFurniture(BuildingType.WORKBENCH));
        assertTrue(BuildingTypeUtils.isFurniture(BuildingType.LOOT_ROOM));
    }

    @Test
    void isWallTool_recognizesWallToolStrings() {
        assertTrue(BuildingTypeUtils.isWallTool("WALL"));
        assertTrue(BuildingTypeUtils.isWallTool("DOOR_FRAME"));
        assertTrue(BuildingTypeUtils.isWallTool("WINDOW_FRAME"));
        assertFalse(BuildingTypeUtils.isWallTool("FOUNDATION"));
        assertFalse(BuildingTypeUtils.isWallTool(null));
        assertFalse(BuildingTypeUtils.isWallTool(""));
    }
}
