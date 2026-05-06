package com.rustbuilder.service;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.structure.Floor;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.deployable.ToolCupboard;
import com.rustbuilder.service.evaluator.HouseEvaluator;

class HouseEvaluatorTest {

    private HouseEvaluator evaluator;
    private GridModel grid;

    @BeforeEach
    void setUp() {
        evaluator = new HouseEvaluator();
        grid = new GridModel();
    }

    private <T extends BuildingBlock> T stone(T block) {
        block.setTier(BuildingTier.STONE);
        return block;
    }

    /**
     * An empty grid has no TC, so the logistics score should be 0
     * and the final score should also be 0.
     */
    @Test
    void emptyGrid_producesZeroLogisticsScore() {
        HouseEvaluator.EvaluationResult result = evaluator.evaluate(grid);
        // No TC means no logistics path -> logistics score must be 0
        assertEquals(0.0, result.logistics.score, 0.001,
            "Empty grid has no TC, so logistics score must be 0");
        // Raid score should also be 0 (the evaluator skips it when logistics == 0)
        assertEquals(0.0, result.raid.score, 0.001,
            "Raid score should be 0 when logistics are invalid");
    }

    /**
     * A grid with only a solo foundation but no TC should still score 0.
     */
    @Test
    void foundationOnly_noTC_hasZeroLogisticsScore() {
        grid.addBlock(new Foundation(0, 0, 0));
        HouseEvaluator.EvaluationResult result = evaluator.evaluate(grid);
        assertEquals(0.0, result.logistics.score, 0.001,
            "Grid with no TC should produce a 0 logistics score");
    }

    /**
     * Weights must sum to influence the final score proportionally.
     * Setting raid weight = 0 means the raid component does not contribute.
     */
    @Test
    void zeroRaidWeight_doesNotIncludeRaidScore() {
        evaluator.setWeights(0.5, 0.5, 0.0, 0.0);
        grid.addBlock(new Foundation(0, 0, 0));
        grid.addBlock(new ToolCupboard(0, 0, 0, 0));

        HouseEvaluator.EvaluationResult result = evaluator.evaluate(grid);
        // Raid should not influence when weight is 0
        double expectedMax = result.logistics.score * 0.5 + result.cost.score * 0.5;
        assertEquals(expectedMax, result.finalScore, 0.001,
            "Final score with 0 raid weight should equal logistics*0.5 + cost*0.5");
    }

    @Test
    void zeroOtherWeights_usesWorkingAreaOnly() {
        evaluator.setWeights(0.0, 0.0, 0.0, 1.0);
        grid.addBlock(new Foundation(0, 0, 0));

        HouseEvaluator.EvaluationResult result = evaluator.evaluate(grid);
        assertEquals(result.workingArea.score, result.finalScore, 0.001,
            "With only working-area weight enabled, final score must equal working-area score");
    }

    @Test
    void enclosedOneByOne_hasFourPerimeterEdgesAndScaledWorkingArea() {
        grid.addBlock(new Foundation(0, 0, 0));
        grid.addBlock(new Wall(0, 0, 0, Orientation.NORTH));
        grid.addBlock(new Wall(0, 0, 0, Orientation.EAST));
        grid.addBlock(new Wall(0, 0, 0, Orientation.SOUTH));
        grid.addBlock(new Wall(0, 0, 0, Orientation.WEST));
        grid.addBlock(new Floor(0, 0, 1, 0));

        HouseEvaluator.EvaluationResult result = evaluator.evaluate(grid);

        assertEquals(4, result.workingArea.openEdgePerimeter,
            "A fully enclosed 1x1 room should expose four perimeter edges");
        assertEquals(0.25, result.workingArea.rawScore, 0.001,
            "One protected tile over four perimeter edges should have raw score 0.25");
        assertEquals(0.2, result.workingArea.score, 0.001,
            "Working-area normalization should not turn a single enclosed tile into 0.5");
        assertEquals(1, result.safeZone.closedBlocks,
            "A fully enclosed 1x1 room should count as one closed safe-zone tile");
    }

    @Test
    void walledTcWithoutRoof_isOpenFromAbove() {
        grid.addBlockSilent(stone(new Foundation(0, 0, 0)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.NORTH)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.EAST)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.SOUTH)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.WEST)));
        grid.addBlockSilent(new ToolCupboard(0, 0, 0, 0));

        HouseEvaluator.EvaluationResult result = evaluator.evaluate(grid);

        assertEquals(0, result.raid.sulfurToTC,
            "A walled TC without a roof should be reachable from outside through the open top");
        assertEquals(0.0, result.raid.score, 0.001,
            "Open-roof TC should not receive raid-resistance score from side-wall coverage");
        assertEquals(0, result.safeZone.closedBlocks,
            "A walled TC without a roof should not count as a closed safe-zone tile");
        assertEquals(0, result.workingArea.protectedTiles,
            "A walled room without a roof should not count as protected working area");
    }

    @Test
    void walledTcWithRoof_requiresRaidCost() {
        grid.addBlockSilent(stone(new Foundation(0, 0, 0)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.NORTH)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.EAST)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.SOUTH)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.WEST)));
        grid.addBlockSilent(stone(new Floor(0, 0, 1, 0)));
        grid.addBlockSilent(new ToolCupboard(0, 0, 0, 0));

        HouseEvaluator.EvaluationResult result = evaluator.evaluate(grid);

        assertEquals(true, result.raid.sulfurToTC > 0,
            "A TC enclosed by walls and roof should require raid cost");
        assertEquals(1, result.safeZone.closedBlocks,
            "A TC enclosed by walls and roof should count as one closed safe-zone tile");
    }

    /**
     * Evaluating the same grid twice should produce the same result (determinism).
     */
    @Test
    void evaluate_isDeterministic() {
        grid.addBlock(new Foundation(0, 0, 0));
        grid.addBlock(new ToolCupboard(30, 30, 0, 0));

        HouseEvaluator.EvaluationResult r1 = evaluator.evaluate(grid);
        HouseEvaluator.EvaluationResult r2 = evaluator.evaluate(grid);

        assertEquals(r1.finalScore, r2.finalScore, 0.0001,
            "Evaluation should be deterministic for the same grid");
    }

    /**
     * EvaluationResult.toString() should not throw.
     */
    @Test
    void evaluationResult_toString_doesNotThrow() {
        HouseEvaluator.EvaluationResult result = evaluator.evaluate(grid);
        assertDoesNotThrow(result::toString);
    }
}
