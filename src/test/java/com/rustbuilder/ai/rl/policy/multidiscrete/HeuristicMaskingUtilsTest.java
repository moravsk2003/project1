package com.rustbuilder.ai.rl.policy.multidiscrete;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.rustbuilder.core.action.BuildAction.ActionType;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.structure.Wall;

public class HeuristicMaskingUtilsTest {

    private GridModel gridModel;

    @BeforeEach
    void setUp() {
        gridModel = new GridModel();
    }

    @Test
    void testGetValidTypesStartWithFoundations() {
        // Step 0: Should only allow foundations
        List<Integer> types = HeuristicMaskingUtils.getValidTypes(gridModel, false, false, 0);
        assertFalse(types.isEmpty(), "Should not be empty");
        
        for (int typeIndex : types) {
            if (typeIndex == MultiDiscreteActionSpace.STOP_TYPE_INDEX) {
                continue;
            }
            ActionType decoded = MultiDiscreteActionSpace.decodeType(typeIndex);
            assertTrue(decoded == ActionType.FOUNDATION || decoded == ActionType.TRIANGLE_FOUNDATION,
                    "Step 0 must only allow foundations, got: " + decoded);
        }
    }

    @Test
    void testGetValidTypesNoCeilingsWithoutWalls() {
        // Place two foundations (so blockCount = 2, meaning we can build other block types)
        gridModel.addBlock(new Foundation(0, 0, 0, 0));
        gridModel.addBlock(new Foundation(60, 0, 0, 0));

        // Step 2, no walls placed yet.
        List<Integer> types = HeuristicMaskingUtils.getValidTypes(gridModel, true, false, 2);

        // Ceiling/Floor (ActionType.FLOOR or TRIANGLE_FLOOR) should be masked out because there are no walls
        for (int typeIndex : types) {
            if (typeIndex == MultiDiscreteActionSpace.STOP_TYPE_INDEX) {
                continue;
            }
            ActionType decoded = MultiDiscreteActionSpace.decodeType(typeIndex);
            assertFalse(decoded == ActionType.FLOOR || decoded == ActionType.TRIANGLE_FLOOR,
                    "Ceilings/floors should not be allowed without walls");
        }

        // Add a wall
        gridModel.addBlock(new Wall(0, 0, 0, Orientation.NORTH));

        // Now ceilings/floors should be allowed
        List<Integer> typesWithWall = HeuristicMaskingUtils.getValidTypes(gridModel, true, false, 3);
        boolean foundFloor = false;
        for (int typeIndex : typesWithWall) {
            if (typeIndex == MultiDiscreteActionSpace.STOP_TYPE_INDEX) {
                continue;
            }
            ActionType decoded = MultiDiscreteActionSpace.decodeType(typeIndex);
            if (decoded == ActionType.FLOOR || decoded == ActionType.TRIANGLE_FLOOR) {
                foundFloor = true;
            }
        }
        assertTrue(foundFloor, "Ceilings/floors should be allowed when at least one wall is present");
    }

    @Test
    void testGetValidFloors() {
        // Foundation index
        int foundationTypeIndex = MultiDiscreteActionSpace.encodeType(ActionType.FOUNDATION);
        List<Integer> foundationFloors = HeuristicMaskingUtils.getValidFloors(gridModel, foundationTypeIndex, 0);
        assertEquals(1, foundationFloors.size());
        assertEquals(0, foundationFloors.get(0), "Foundations must only be allowed on floor 0");

        // Wall index
        int wallTypeIndex = MultiDiscreteActionSpace.encodeType(ActionType.WALL);
        // Initially no floor support (no foundation/walls)
        List<Integer> wallFloorsEmpty = HeuristicMaskingUtils.getValidFloors(gridModel, wallTypeIndex, 0);
        assertTrue(wallFloorsEmpty.isEmpty(), "Walls should not be allowed on any floor without supporting structures");

        // Add foundation
        gridModel.addBlock(new Foundation(0, 0, 0, 0));
        List<Integer> wallFloorsWithFoundation = HeuristicMaskingUtils.getValidFloors(gridModel, wallTypeIndex, 1);
        assertFalse(wallFloorsWithFoundation.isEmpty());
        assertTrue(wallFloorsWithFoundation.contains(0), "Walls should be allowed on floor 0 with foundation support");
    }

    @Test
    void testGetValidRotations() {
        int foundationTypeIndex = MultiDiscreteActionSpace.encodeType(ActionType.FOUNDATION);
        List<Integer> foundationRotations = HeuristicMaskingUtils.getValidRotations(gridModel, foundationTypeIndex, 0, 0);
        assertEquals(1, foundationRotations.size());
        assertEquals(0, foundationRotations.get(0), "Foundation should be rotation-invariant (only index 0)");

        int wallTypeIndex = MultiDiscreteActionSpace.encodeType(ActionType.WALL);
        List<Integer> wallRotations = HeuristicMaskingUtils.getValidRotations(gridModel, wallTypeIndex, 0, 0);
        assertEquals(4, wallRotations.size(), "Walls should have 4 rotations");

        int triFloorIndex = MultiDiscreteActionSpace.encodeType(ActionType.TRIANGLE_FLOOR);
        List<Integer> triFloorRotations = HeuristicMaskingUtils.getValidRotations(gridModel, triFloorIndex, 0, 0);
        assertEquals(6, triFloorRotations.size(), "Triangle floors should have 6 rotations");
    }
}
