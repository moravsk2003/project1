package com.rustbuilder.service;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.service.physics.StabilityService;

class StabilityServiceTest {

    /**
     * A single foundation at floor 0 must have stability = 1.0 after recalculation.
     */
    @Test
    void foundation_atFloor0_hasFullStability() {
        GridModel grid = new GridModel();
        Foundation f = new Foundation(0, 0, 0);
        grid.addBlock(f);

        StabilityService.recalculateAll(grid);

        assertEquals(1.0, f.getStability(), 0.001);
    }

    /**
     * A wall placed directly on a foundation should receive stability > 0.
     */
    @Test
    void wallOnFoundation_receivesPositiveStability() {
        GridModel grid = new GridModel();
        Foundation f = new Foundation(0, 0, 0);
        // A wall sits on the north edge of the foundation
        Wall wall = new Wall(0, 0, 0, Orientation.NORTH);
        grid.addBlock(f);
        grid.addBlock(wall);

        StabilityService.recalculateAll(grid);

        assertTrue(wall.getStability() > 0.0,
            "Wall on a foundation should have stability > 0, got: " + wall.getStability());
    }

    /**
     * A block with no support (floating wall) should have stability = 0
     * after recalculation via the list-based legacy method.
     */
    @Test
    void floatingWall_hasZeroStability_legacyMethod() {
        Wall wall = new Wall(500, 500, 5, Orientation.NORTH);

        List<BuildingBlock> blocks = List.of(wall);
        StabilityService.recalculateAll(blocks);

        assertEquals(0.0, wall.getStability(), 0.001,
            "A floating wall with no connecting foundation should have 0 stability");
    }

    /**
     * Multiple foundations at floor 0 should all have full stability.
     */
    @Test
    void multipleFoundations_allHaveFullStability() {
        GridModel grid = new GridModel();
        Foundation f1 = new Foundation(0, 0, 0);
        Foundation f2 = new Foundation(60, 0, 0);
        Foundation f3 = new Foundation(0, 60, 0);
        grid.addBlock(f1);
        grid.addBlock(f2);
        grid.addBlock(f3);

        StabilityService.recalculateAll(grid);

        assertEquals(1.0, f1.getStability(), 0.001);
        assertEquals(1.0, f2.getStability(), 0.001);
        assertEquals(1.0, f3.getStability(), 0.001);
    }

    /**
     * After clearing the grid, all previously-added blocks are gone and
     * re-running stability doesn't throw.
     */
    @Test
    void clearGrid_thenRecalculate_doesNotThrow() {
        GridModel grid = new GridModel();
        grid.addBlock(new Foundation(0, 0, 0));
        grid.clear();

        assertDoesNotThrow(() -> StabilityService.recalculateAll(grid));
    }
}
