package com.rustbuilder.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.structure.TriangleFoundation;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.service.physics.SnappingService;

public class SnappingServiceTest {

    private GridModel gridModel;
    private SnappingService snappingService;

    @BeforeEach
    void setUp() {
        gridModel = new GridModel();
        snappingService = new SnappingService(gridModel);
    }

    @Test
    void testSnapToFoundation() {
        // Place a foundation at 0,0
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);

        double mouseX = 30;
        double mouseY = 0;
        
        SnappingService.SnapResult result = snappingService.calculateSnap(mouseX, mouseY, "WALL", 0);
        
        assertTrue(result.valid, "Should find a valid snap");
        assertEquals(Orientation.NORTH, result.orientation, "Should snap to North orientation");
    }

    @Test
    void testSnapFloorToWall() {
        // Place a foundation so the wall has support
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);

        // Place a wall at 0,0
        Wall wall = new Wall(0, 0, 0, Orientation.NORTH);
        gridModel.addBlock(wall);

        // Try to snap a floor to the center of the wall tile
        double mouseX = 30;
        double mouseY = 30;

        // Snap Floor at currentFloor=1 (floors go on top of walls)
        SnappingService.SnapResult result = snappingService.calculateSnap(mouseX, mouseY, "FLOOR", 1);

        assertTrue(result.valid, "Should find a valid snap for floor");
        assertEquals(180.0, result.rotation, 0.01, "Floor should be rotated 180 degrees");
    }

    @Test
    void testSnapCeilingInsideWallOnFoundation() {
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);

        Wall wall = new Wall(0, 0, 0, Orientation.NORTH);
        gridModel.addBlock(wall);

        SnappingService.SnapResult result = snappingService.calculateSnap(30, 30, "FLOOR", 1);

        assertTrue(result.valid, "Should find a valid snap for inside ceiling");
        assertEquals(0, result.x, 0.01, "Ghost X should stay inside the wall tile");
        assertEquals(0, result.y, 0.01, "Ghost Y should stay inside the wall tile");
    }

    @Test
    void testSnapFloorToEastWall() {
        // Place a foundation so the wall has support
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);

        // Place a wall at 0,0 East
        Wall wall = new Wall(0, 0, 0, Orientation.EAST);
        gridModel.addBlock(wall);

        // Snap Floor at currentFloor=1
        SnappingService.SnapResult result = snappingService.calculateSnap(30, 30, "FLOOR", 1);

        assertTrue(result.valid);
        assertEquals(270.0, result.rotation, 0.01, "Floor on East wall should be rotated 270 degrees");
    }
    @Test
    void testSnapCeilingOutsideWallOnFoundation() {
        // 1. Place Square Foundation at 0,0
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);

        // 2. Place Wall on Foundation (North)
        Wall wall = new Wall(0, 0, 0, Orientation.NORTH);
        gridModel.addBlock(wall);

        double mouseX = 30;
        double mouseY = -10;

        SnappingService.SnapResult result = snappingService.calculateSnap(mouseX, mouseY, "FLOOR", 1);

        assertTrue(result.valid, "Should find a valid snap for outside ceiling");
        assertEquals(0, result.x, 0.01, "Ghost X should be 0");
        assertEquals(-60, result.y, 0.01, "Ghost Y should be -60 (North of wall)");
    }

    @Test
    void testDoorIgnoresCloserNonDoorwaySocket() {
        Foundation leftFoundation = new Foundation(0, 0, 0, 0);
        Foundation doorwayFoundation = new Foundation(60, 0, 0, 0);
        gridModel.addBlock(leftFoundation);
        gridModel.addBlock(doorwayFoundation);

        Wall doorway = new Wall(60, 0, 0, Orientation.NORTH);
        doorway.setType(BuildingType.DOORWAY);
        gridModel.addBlock(doorway);

        SnappingService.SnapResult result = snappingService.calculateSnap(50, 30, "DOOR", 0);

        assertTrue(result.valid, "Door should snap to the nearest eligible doorway, not a closer foundation socket");
        assertEquals(60, result.x, 0.01);
        assertEquals(0, result.y, 0.01);
        assertEquals(Orientation.NORTH, result.orientation);
    }

    @Test
    void testWallOnTriangleUsesClosestSocketSide() {
        TriangleFoundation triangle = new TriangleFoundation(0, 0, 0, 0);
        gridModel.addBlock(triangle);

        double rightSocketX = 45;
        double rightSocketY = 30 + com.rustbuilder.config.GameConstants.TRIANGLE_OFFSET / 2.0;

        SnappingService.SnapResult result = snappingService.calculateSnap(rightSocketX, rightSocketY, "WALL", 0);

        assertTrue(result.valid);
        assertEquals(Orientation.TRIANGLE_RIGHT, result.orientation);
    }
}
