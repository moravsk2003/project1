package com.rustbuilder.model;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.deployable.ToolCupboard;
import com.rustbuilder.model.structure.Floor;
import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.structure.Wall;
import org.junit.jupiter.api.Test;

class GridModelInvariantTest {

    @Test
    void acceptedMutationsPreserveCollisionAndStabilityInvariants() {
        GridModel grid = new GridModel();

        assertTrue(grid.addBlock(new Foundation(0, 0, 0, 0)));
        assertGridInvariants(grid);

        assertTrue(grid.addBlock(new Wall(0, 0, 0, Orientation.NORTH)));
        assertGridInvariants(grid);

        assertTrue(grid.addBlock(new Floor(0, 0, 1, 0)));
        assertGridInvariants(grid);

        assertTrue(grid.addBlock(new ToolCupboard(0, 0, 0, 0)));
        assertGridInvariants(grid);
    }

    @Test
    void removingSupportPrunesUnsupportedDependentBlocks() {
        GridModel grid = new GridModel();
        Foundation foundation = new Foundation(0, 0, 0, 0);
        Wall wall = new Wall(0, 0, 0, Orientation.NORTH);
        Floor roof = new Floor(0, 0, 1, 0);

        assertTrue(grid.addBlock(foundation));
        assertTrue(grid.addBlock(wall));
        assertTrue(grid.addBlock(roof));
        assertGridInvariants(grid);

        grid.removeBlock(foundation);

        assertTrue(grid.getAllBlocks().isEmpty(), "Removing the only support should prune dependents");
    }

    @Test
    void gridAllowsAtMostOneToolCupboard() {
        GridModel grid = new GridModel();

        assertTrue(grid.addBlock(new Foundation(0, 0, 0, 0)));
        assertTrue(grid.addBlock(new Foundation(60, 0, 0, 0)));
        assertTrue(grid.addBlock(new ToolCupboard(0, 0, 0, 0)));

        assertFalse(grid.canPlace(new ToolCupboard(60, 0, 0, 0)),
                "Grid placement rules should enforce the global TC limit");
        assertGridInvariants(grid);
    }

    private static void assertGridInvariants(GridModel grid) {
        long toolCupboards = grid.getAllBlocks().stream()
                .filter(block -> block.getType() == BuildingType.TC)
                .count();
        assertTrue(toolCupboards <= 1, "Grid must not contain more than one TC");

        for (BuildingBlock block : grid.getAllBlocks()) {
            assertFalse(grid.hasCollision(block), "Accepted grid contains a collision for " + block.getType());
            assertTrue(block.getStability() >= 0.1,
                    "Accepted grid contains an unsupported block: " + block.getType());
        }
    }
}
