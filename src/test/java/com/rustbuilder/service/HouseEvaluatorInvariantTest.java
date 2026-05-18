package com.rustbuilder.service;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.deployable.ToolCupboard;
import com.rustbuilder.model.structure.Floor;
import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.service.evaluator.HouseEvaluator;
import java.util.List;
import org.junit.jupiter.api.Test;

class HouseEvaluatorInvariantTest {

    @Test
    void defaultFinalScoreStaysNormalizedForRepresentativeGrids() {
        HouseEvaluator evaluator = new HouseEvaluator();

        for (GridModel grid : List.of(new GridModel(), foundationWithTc(), enclosedTcRoom())) {
            HouseEvaluator.EvaluationResult result = evaluator.evaluate(grid);
            assertTrue(result.finalScore >= 0.0 && result.finalScore <= 1.0,
                    "Default final score should stay in [0, 1], got " + result.finalScore);
        }
    }

    private static GridModel foundationWithTc() {
        GridModel grid = new GridModel();
        grid.addBlock(new Foundation(0, 0, 0, 0));
        grid.addBlock(new ToolCupboard(0, 0, 0, 0));
        return grid;
    }

    private static GridModel enclosedTcRoom() {
        GridModel grid = new GridModel();
        grid.addBlockSilent(stone(new Foundation(0, 0, 0, 0)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.NORTH)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.EAST)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.SOUTH)));
        grid.addBlockSilent(stone(new Wall(0, 0, 0, Orientation.WEST)));
        grid.addBlockSilent(stone(new Floor(0, 0, 1, 0)));
        grid.addBlockSilent(new ToolCupboard(0, 0, 0, 0));
        grid.finalizeLoad();
        return grid;
    }

    private static <T extends com.rustbuilder.model.core.BuildingBlock> T stone(T block) {
        block.setTier(BuildingTier.STONE);
        return block;
    }
}
