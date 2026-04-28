package com.rustbuilder.service;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.deployable.ToolCupboard;
import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.service.evaluator.HouseEvaluator;

public class TestOpenBase {
    public static void main(String[] args) {
        HouseEvaluator evaluator = new HouseEvaluator();
        GridModel grid = new GridModel();
        
        // Place 3x3 foundations
        for(int x=0; x<3; x++) {
            for(int y=0; y<3; y++) {
                grid.addBlock(new Foundation(x * 60, y * 60, 0));
            }
        }
        
        // Place TC in center
        grid.addBlock(new ToolCupboard(60, 60, 0, 0));
        
        HouseEvaluator.EvaluationResult result = evaluator.evaluate(grid);
    }
}
