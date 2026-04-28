package com.rustbuilder.service.evaluator;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.service.graph.HouseGraph;
import java.util.List;

/**
 * Combines all 4 evaluation criteria into a single score.
 * This will serve as the fitness function for the AI generator.
 */
public class HouseEvaluator {

    private double logisticsWeight = 0.25;
    private double costWeight = 0.20;
    private double raidWeight = 0.30;
    private double workingAreaWeight = 0.25;

    private final LogisticsEvaluator logisticsEvaluator = new LogisticsEvaluator();
    private final ResourceCostEvaluator costEvaluator = new ResourceCostEvaluator();
    private final RaidResistanceEvaluator raidEvaluator = new RaidResistanceEvaluator();
    private final WorkingAreaEvaluator workingAreaEvaluator = new WorkingAreaEvaluator();

    public static class EvaluationResult {
        public final LogisticsEvaluator.LogisticsResult logistics;
        public final ResourceCostEvaluator.CostResult cost;
        public final RaidResistanceEvaluator.RaidResult raid;
        public final WorkingAreaEvaluator.WorkingAreaResult workingArea;
        public final double finalScore;

        public EvaluationResult(LogisticsEvaluator.LogisticsResult logistics,
                                ResourceCostEvaluator.CostResult cost,
                                RaidResistanceEvaluator.RaidResult raid,
                                WorkingAreaEvaluator.WorkingAreaResult workingArea,
                                double finalScore) {
            this.logistics = logistics;
            this.cost = cost;
            this.raid = raid;
            this.workingArea = workingArea;
            this.finalScore = finalScore;
        }

        @Override
        public String toString() {
            return String.format(
                "=== House Evaluation ===\n%s\n%s\n%s\n%s\nFinal Score: %.2f",
                logistics, cost, raid, workingArea, finalScore
            );
        }
    }

    public void setWeights(double logistics, double cost, double raid, double workingArea) {
        this.logisticsWeight = logistics;
        this.costWeight = cost;
        this.raidWeight = raid;
        this.workingAreaWeight = workingArea;
    }

    /**
     * Evaluate the house described by the GridModel.
     */
    public EvaluationResult evaluate(GridModel gridModel) {
        List<BuildingBlock> blocks = gridModel.getAllBlocks();

        // 1. Build 3D graph
        HouseGraph graph = new HouseGraph();
        graph.buildGraph(blocks);

        // 2. Evaluate performance criteria first
        LogisticsEvaluator.LogisticsResult logResult = logisticsEvaluator.evaluate(graph);
        WorkingAreaEvaluator.WorkingAreaResult workingAreaResult = workingAreaEvaluator.evaluate(graph, blocks);
        
        // Always calculate raid score so the reward surface isn't flat when TC is missing
        RaidResistanceEvaluator.RaidResult computedRaid = raidEvaluator.evaluate(graph, blocks);
        RaidResistanceEvaluator.RaidResult raidResult;
        
        if (logResult.score < 0.0001) {
             // heavily penalize but retain gradient
             raidResult = new RaidResistanceEvaluator.RaidResult(computedRaid.sulfurToTC, computedRaid.sulfurToLootRoom, computedRaid.score * 0.1);
        } else {
             raidResult = computedRaid;
        }

        // 3. Resource cost now depends on base quality
        ResourceCostEvaluator.CostResult costResult = costEvaluator.evaluate(
            blocks, 
            logResult.score, 
            raidResult.score, 
            workingAreaResult.score
        );

        // 4. Combine scores
        // logistics: higher = better (shorter paths)
        // cost: higher = cheaper (less resources)
        // raid: higher = more raid-resistant
        // working area: higher = more protected usable area per open edge
        double finalScore = logisticsWeight * logResult.score +
                            costWeight * costResult.score +
                            raidWeight * raidResult.score +
                            workingAreaWeight * workingAreaResult.score;

        return new EvaluationResult(logResult, costResult, raidResult, workingAreaResult, finalScore);
    }
}
