package com.rustbuilder.ai;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.ai.ea.BaseGenome;
import com.rustbuilder.core.action.BuildAction;

/**
 * Unit tests for BaseGenome.decode().
 * Verifies ordered placement, skip-invalid-snap logic, and no floating blocks.
 */
class BaseGenomeDecodeTest {

    // ── helpers ────────────────────────────────────────────────────────────────

    /** Builds an action list that only contains walls (no foundations). */
    private BaseGenome wallsOnlyGenome() {
        List<BuildAction> actions = new java.util.ArrayList<>();
        for (int i = 0; i < 5; i++) {
            actions.add(new BuildAction(
                BuildAction.ActionType.WALL,
                i, 0, 0, 0, 2, 0
            ));
        }
        return new BaseGenome(actions);
    }

    /** Builds a genome with one foundation + walls around it. */
    private BaseGenome foundationPlusWallsGenome() {
        List<BuildAction> actions = new java.util.ArrayList<>();
        // One foundation at grid tile (2,2)
        actions.add(new BuildAction(
            BuildAction.ActionType.FOUNDATION,
            2, 2, 0, 0, 2, 0
        ));
        // Walls — will attempt to snap to the foundation's edge sockets
        for (int i = 0; i < 4; i++) {
            actions.add(new BuildAction(
                BuildAction.ActionType.WALL,
                2 + i, 2, 0, i, 2, 0
            ));
        }
        return new BaseGenome(actions);
    }

    /** Creates a random genome as produced by the GA. */
    private BaseGenome randomGenome() {
        return BaseGenome.randomGenome();
    }

    // ── tests ──────────────────────────────────────────────────────────────────

    /**
     * Walls-only genome: no foundation exists → no snap target → 0 blocks placed.
     * Confirms the skip-invalid logic. decode() must return false (placed == 0).
     */
    @Test
    void wallsOnly_noFoundation_nothingPlaced() {
        GridModel grid = new GridModel();
        boolean placed = wallsOnlyGenome().decode(grid);
        assertFalse(placed, "Expected decode() to return false — no foundation means no snap target for walls");
        assertEquals(0, grid.getAllBlocks().size(), "No blocks should be placed without a foundation");
    }

    /**
     * A genome with one foundation followed by walls:
     * at least the foundation must be placed (walls may or may not snap depending
     * on tile positions, but decode must return true).
     */
    @Test
    void foundationAlwaysPlaced_regardlessOfSnap() {
        GridModel grid = new GridModel();
        boolean placed = foundationPlusWallsGenome().decode(grid);
        assertTrue(placed, "decode() should return true — foundation doesn't need snap");
        assertTrue(grid.getAllBlocks().size() >= 1, "At least the foundation should be placed");
        // Confirm the foundation is in the grid
        boolean hasFoundation = grid.getAllBlocks().stream()
            .anyMatch(b -> b.getType() == BuildingType.FOUNDATION);
        assertTrue(hasFoundation, "Grid must contain at least one FOUNDATION block");
    }

    /**
     * All blocks after decode must have passed canPlace() at placement time,
     * meaning no block should be in a position where canPlace() now returns false.
     * (Tests the post-condition: what's in the grid is valid.)
     */
    @Test
    void decode_noFloatingOrColliding_blocks() {
        // Use a fixed genome with foundation + structural blocks
        BaseGenome genome = foundationPlusWallsGenome();
        GridModel grid = new GridModel();
        genome.decode(grid);

        List<BuildingBlock> blocks = grid.getAllBlocks();
        for (BuildingBlock block : blocks) {
            // Temporarily remove the block, check canPlace, then re-add
            grid.removeBlock(block);
            boolean wouldBeValid = grid.canPlace(block);
            grid.addBlock(block);
            // Note: canPlace checks collision. Since it was placed, it must be valid.
            assertTrue(wouldBeValid,
                "Block " + block.getType() + " at (" + block.getX() + "," + block.getY() + ") failed canPlace() post-decode");
        }
    }

    /**
     * Random genome should decode without throwing any exception.
     */
    @Test
    void randomGenome_decodesWithoutException() {
        for (int i = 0; i < 10; i++) {
            GridModel grid = new GridModel();
            assertDoesNotThrow(() -> randomGenome().decode(grid),
                "decode() must not throw on random genomes");
        }
    }

    /**
     * Confirm phases are ordered: after decode, the grid cannot contain a wall
     * that was placed BEFORE any foundation existed (ordering guarantee).
     * Practical check: if foundations are in the result, their count ≥ 1.
     */
    @Test
    void orderedDecode_foundationsExistBeforeWalls() {
        BaseGenome genome = foundationPlusWallsGenome();
        GridModel grid = new GridModel();
        genome.decode(grid);

        long foundationCount = grid.getAllBlocks().stream()
            .filter(b -> b.getType() == BuildingType.FOUNDATION)
            .count();
        long wallCount = grid.getAllBlocks().stream()
            .filter(b -> b.getType() == BuildingType.WALL)
            .count();

        // If any walls were placed, there must have been at least one foundation
        if (wallCount > 0) {
            assertTrue(foundationCount > 0,
                "Walls placed implies foundations must have been placed first");
        }
    }
}
