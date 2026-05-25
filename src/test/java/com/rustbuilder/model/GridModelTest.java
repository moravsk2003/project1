package com.rustbuilder.model;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.model.structure.Door;
import com.rustbuilder.model.structure.Floor;
import com.rustbuilder.model.structure.TriangleFloor;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.core.action.BuildAction;
import com.rustbuilder.config.GameConstants;
import com.rustbuilder.service.physics.PlacementResult;
import com.rustbuilder.service.physics.PlacementService;

public class GridModelTest {

    private GridModel gridModel;

    @BeforeEach
    void setUp() {
        gridModel = new GridModel();
    }

    @Test
    void testWallJunctionCollision() {
        // Place a foundation so walls have structural support (testing collision, not stability)
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);

        // Place first wall
        Wall wall1 = new Wall(0, 0, 0, Orientation.NORTH);
        gridModel.addBlock(wall1);

        // Place another wall at 90 degrees on the same tile.
        Wall wall2 = new Wall(0, 0, 0, Orientation.EAST);

        assertTrue(gridModel.canPlace(wall2), "Should allow placing wall at 90 degrees on same tile (Corner)");
    }

    @Test
    void canPlaceRejectsDuplicateWallOrientation() {
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);

        Wall wall = new Wall(0, 0, 0, Orientation.NORTH);
        gridModel.addBlock(wall);

        Wall duplicateWall = new Wall(0, 0, 0, Orientation.NORTH);
        assertFalse(gridModel.canPlace(duplicateWall), "Should reject duplicate wall at same tile and orientation");
    }

    @Test
    void canPlaceAllowsMultipleDoorsOnDifferentDoorwayOrientations() {
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);

        Wall northDoorway = new Wall(0, 0, 0, Orientation.NORTH);
        northDoorway.setType(BuildingType.DOORWAY);
        gridModel.addBlock(northDoorway);

        Wall eastDoorway = new Wall(0, 0, 0, Orientation.EAST);
        eastDoorway.setType(BuildingType.DOORWAY);
        gridModel.addBlock(eastDoorway);

        Door northDoor = new Door(0, 0, 0, Orientation.NORTH, DoorType.SHEET_METAL);
        assertTrue(gridModel.canPlace(northDoor), "First door should fit in matching north doorway");
        gridModel.addBlock(northDoor);

        Door eastDoor = new Door(0, 0, 0, Orientation.EAST, DoorType.SHEET_METAL);
        assertTrue(gridModel.canPlace(eastDoor), "Different doorway orientations on one tile should each allow a door");

        Door duplicateNorthDoor = new Door(0, 0, 0, Orientation.NORTH, DoorType.GARAGE);
        assertFalse(gridModel.canPlace(duplicateNorthDoor), "Duplicate door orientation should still be rejected");
    }

    @Test
    void testWallFoundationCollision() {
        // Place foundation
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);

        Wall wall = new Wall(0, 0, 0, Orientation.NORTH);

        assertTrue(gridModel.canPlace(wall), "Should allow placing wall on foundation");
    }

    @Test
    void testFloorWallCollision() {
        // Place a wall
        Wall wall = new Wall(0, 0, 0, Orientation.NORTH);
        gridModel.addBlock(wall);

        // Place a floor at the same location
        Floor floor = new Floor(0, 0, 0, 0);

        assertFalse(gridModel.canPlace(floor), "Should NOT allow placing floor intersecting wall");
    }
    @Test
    void testSquareFloorDiagonalWallCollision() {
        // Place a diagonal wall (Triangle Left) on a tile
        Wall wall = new Wall(0, 0, 0, Orientation.TRIANGLE_LEFT);
        gridModel.addBlock(wall);

        // Place a square floor on the same tile
        Floor floor = new Floor(0, 0, 0, 0);

        // They SHOULD collide because the diagonal wall cuts through the square floor
        assertFalse(gridModel.canPlace(floor), "Should NOT allow placing square floor on top of diagonal wall");
    }

    @Test
    void testFloorCollisionWithWallBelow() {
        // 1. Place a diagonal wall at Z=0
        Wall diagonalWall = new Wall(0, 0, 0, Orientation.TRIANGLE_LEFT);
        gridModel.addBlock(diagonalWall);

        // 2. Try to place a square floor at Z=1
        Floor floor = new Floor(0, 0, 1, 0);

        // A diagonal wall's polygon intersects the floor polygon even across Z levels.
        // This is intentional: diagonal walls cut through the space above them.
        assertFalse(gridModel.canPlace(floor), "Should NOT allow placing floor at Z=1 if diagonal wall at Z=0 intersects it");

        // 3. Now test with an edge wall; it should not block a floor above it.
        gridModel.clear();
        // Add a foundation so the wall has support
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);
        Wall edgeWall = new Wall(0, 0, 0, Orientation.NORTH);
        gridModel.addBlock(edgeWall);

        assertTrue(gridModel.canPlace(floor), "Should allow placing floor at Z=1 if wall at Z=0 is an edge-only wall");
    }

    @Test
    void testTriangleFloorOnTriangleWall() {
        // Place a Foundation so the wall has structural support (we're testing collision, not stability)
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);

        // Wall at 0,0,0 with TRIANGLE_LEFT orientation
        Wall wall = new Wall(0, 0, 0, Orientation.TRIANGLE_LEFT);
        gridModel.addBlock(wall);

        // Try to place a Triangle Floor at 0,0,1 with Rotation 0
        TriangleFloor floor = new TriangleFloor(0, 0, 1, 0);

        // This should be ALLOWED.
        // The triangle wall is a diagonal on the edge; the triangle floor above it
        // should not intersect the wall polygon when properly shrunk.
        assertTrue(gridModel.canPlace(floor), "Should allow placing Triangle Floor (Rot 0) on Triangle Wall (Left)");
    }

    @Test
    void triangleFloorCannotUseWallCenterSocketAsSupport() {
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);

        Wall wall = new Wall(0, 0, 0, Orientation.NORTH);
        gridModel.addBlock(wall);

        TriangleFloor floor = new TriangleFloor(0, 0, 1, 0);

        assertFalse(gridModel.canPlace(floor), "Triangle floor should not be supported by a wall center socket");
    }

    @Test
    void triangleFloorRejectsNonParallelWallEdgeSocket() {
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);

        Wall wall = new Wall(0, 0, 0, Orientation.NORTH);
        gridModel.addBlock(wall);

        double x = GameConstants.HALF_TILE / 2.0;
        double y = -GameConstants.HALF_TILE - GameConstants.TRIANGLE_OFFSET / 2.0;
        TriangleFloor floor = new TriangleFloor(x, y, 1, 0);

        assertFalse(gridModel.canPlace(floor), "Triangle floor should reject a close but non-parallel wall edge socket");
    }

    @Test
    void testFloatingWallRejected() {
        // Try to place a wall in the air (no foundation, no support)
        Wall wall = new Wall(0, 0, 0, Orientation.NORTH);
        
        // Should be rejected because it has no support
        assertFalse(gridModel.canPlace(wall), "Should NOT allow placing floating wall");
        
        // Place a foundation
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);
        
        // Now place the wall on the foundation
        // Note: Foundation is at 0,0,0. Wall is at 0,0,0.
        // Support factor logic: Wall on Foundation (Z diff 0, dist < 0.8 tile) -> Supported.
        assertTrue(gridModel.canPlace(wall), "Should allow placing wall on foundation");
    }

    @Test
    void testWallNextToWallRejected() {
        // 1. Place Foundation and Wall at 0,0
        Foundation foundation = new Foundation(0, 0, 0, 0);
        gridModel.addBlock(foundation);
        Wall wall1 = new Wall(0, 0, 0, Orientation.NORTH);
        gridModel.addBlock(wall1);

        // 2. Try to place a Wall at 0, -60 (North of existing wall)
        // There is NO foundation at 0, -60.
        // It should be rejected.
        Wall wall2 = new Wall(0, -60, 0, Orientation.NORTH);
        
        assertFalse(gridModel.canPlace(wall2), "Should NOT allow placing wall next to another wall without foundation");
    }

    @Test
    void aiPlacementRejectsSameFloorWallAsWallTarget() {
        double x = GameConstants.GRID_ORIGIN_X - GameConstants.HALF_TILE;
        double y = GameConstants.GRID_ORIGIN_Y - GameConstants.HALF_TILE;
        gridModel.addBlockSilent(new Wall(x, y, 0, Orientation.NORTH));

        BuildAction action = new BuildAction(BuildAction.ActionType.WALL, 0, 0, 0, 1, 2, 0, 12);

        PlacementResult placement = PlacementService.calculatePlacement(gridModel, action);

        assertFalse(placement.isValid(), "AI should not place a same-floor wall by targeting another wall");
    }

    @Test
    void aiPlacementAllowsWallStackingOnlyFromWallBelow() {
        double x = GameConstants.GRID_ORIGIN_X - GameConstants.HALF_TILE;
        double y = GameConstants.GRID_ORIGIN_Y - GameConstants.HALF_TILE;
        gridModel.addBlockSilent(new Wall(x, y, 0, Orientation.NORTH));

        BuildAction action = new BuildAction(BuildAction.ActionType.WALL, 0, 0, 1, 1, 2, 0, 12);

        PlacementResult placement = PlacementService.calculatePlacement(gridModel, action);

        assertTrue(placement.isValid(), "AI should allow vertical wall stacking from a wall one floor below");
    }

    @Test
    void aiPlacementAllowsCeilingInsideWallTile() {
        double x = GameConstants.GRID_ORIGIN_X - GameConstants.HALF_TILE;
        double y = GameConstants.GRID_ORIGIN_Y - GameConstants.HALF_TILE;
        gridModel.addBlock(new Foundation(x, y, 0, 0));
        gridModel.addBlock(new Wall(x, y, 0, Orientation.NORTH));

        BuildAction action = new BuildAction(BuildAction.ActionType.FLOOR, 0, 0, 1, 0, 2, 0, 5);

        PlacementResult placement = PlacementService.calculatePlacement(gridModel, action);

        assertTrue(placement instanceof PlacementResult.Valid,
                "AI should allow a ceiling on the inside side of the wall");
        PlacementResult.Valid validPlacement = (PlacementResult.Valid) placement;
        assertEquals(x, validPlacement.x(), 0.01);
        assertEquals(y, validPlacement.y(), 0.01);
        assertTrue(PlacementService.isActionActuallyFeasible(gridModel, action),
                "RL feasibility mask should see the inside ceiling as a valid action");
    }

    @Test
    void testSpatialIndexHashingAndTombstones() {
        int count = 100;
        java.util.List<Foundation> blocks = new java.util.ArrayList<>();
        for (int i = 0; i < count; i++) {
            Foundation foundation = new Foundation(i * 120.0, 0, 0, 0);
            blocks.add(foundation);
            assertTrue(gridModel.addBlockSilent(foundation));
        }

        for (int i = 0; i < count; i++) {
            double targetX = i * 120.0;
            java.util.List<BuildingBlock> found = gridModel.getNearbyBlocks(targetX, 0, 0, 1.0);
            assertEquals(1, found.size());
            assertEquals(blocks.get(i), found.get(0));
        }

        for (int i = 0; i < count; i++) {
            gridModel.removeBlock(blocks.get(i));
        }

        assertTrue(gridModel.getAllBlocks().isEmpty());
        for (int i = 0; i < count; i++) {
            double targetX = i * 120.0;
            java.util.List<BuildingBlock> found = gridModel.getNearbyBlocks(targetX, 0, 0, 1.0);
            assertTrue(found.isEmpty());
        }

        for (int i = 0; i < count; i++) {
            assertTrue(gridModel.addBlockSilent(blocks.get(i)));
        }

        for (int i = 0; i < count; i++) {
            double targetX = i * 120.0;
            java.util.List<BuildingBlock> found = gridModel.getNearbyBlocks(targetX, 0, 0, 1.0);
            assertEquals(1, found.size());
        }
    }
}
