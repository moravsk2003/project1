package com.rustbuilder.service;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.deployable.ToolCupboard;
import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.service.evaluator.HouseEvaluator;
import org.junit.jupiter.api.Test;

class TestOpenBase {

    @Test
    void evaluatesOpenBaseWithoutThrowing() {
        HouseEvaluator evaluator = new HouseEvaluator();
        GridModel grid = new GridModel();

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                grid.addBlock(new Foundation(x * 60, y * 60, 0));
            }
        }

        grid.addBlock(new ToolCupboard(60, 60, 0, 0));

        HouseEvaluator.EvaluationResult result = assertDoesNotThrow(() -> evaluator.evaluate(grid));
        assertNotNull(result);
    }
}
