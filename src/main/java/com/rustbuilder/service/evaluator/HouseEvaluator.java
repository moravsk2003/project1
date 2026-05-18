package com.rustbuilder.service.evaluator;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.service.graph.HouseGraph;
import java.util.List;
import java.util.Objects;

/**
 * Combines all 5 evaluation criteria into a single score.
 * This will serve as the fitness function for the AI generator.
 */
public class HouseEvaluator implements HouseEvaluationService {

    private static final double DEFAULT_LOGISTICS_WEIGHT = 0.22;
    private static final double DEFAULT_COST_WEIGHT = 0.18;
    private static final double DEFAULT_RAID_WEIGHT = 0.28;
    private static final double DEFAULT_WORKING_AREA_WEIGHT = 0.22;
    private static final double DEFAULT_SAFE_ZONE_WEIGHT = 0.10;

    private double logisticsWeight = DEFAULT_LOGISTICS_WEIGHT;
    private double costWeight = DEFAULT_COST_WEIGHT;
    private double raidWeight = DEFAULT_RAID_WEIGHT;
    private double workingAreaWeight = DEFAULT_WORKING_AREA_WEIGHT;
    private double safeZoneWeight = DEFAULT_SAFE_ZONE_WEIGHT;

    private final LogisticsEvaluator logisticsEvaluator;
    private final ResourceCostEvaluator costEvaluator;
    private final RaidResistanceEvaluator raidEvaluator;
    private final WorkingAreaEvaluator workingAreaEvaluator;
    private final SafeZoneEvaluator safeZoneEvaluator;
    private final HouseGraphFactory graphFactory;

    public HouseEvaluator() {
        this(new LogisticsEvaluator(),
                new ResourceCostEvaluator(),
                new RaidResistanceEvaluator(),
                new WorkingAreaEvaluator(),
                new SafeZoneEvaluator(),
                HouseGraph::new);
    }

    public HouseEvaluator(LogisticsEvaluator logisticsEvaluator,
                          ResourceCostEvaluator costEvaluator,
                          RaidResistanceEvaluator raidEvaluator,
                          WorkingAreaEvaluator workingAreaEvaluator,
                          SafeZoneEvaluator safeZoneEvaluator,
                          HouseGraphFactory graphFactory) {
        this.logisticsEvaluator = Objects.requireNonNull(logisticsEvaluator, "logisticsEvaluator");
        this.costEvaluator = Objects.requireNonNull(costEvaluator, "costEvaluator");
        this.raidEvaluator = Objects.requireNonNull(raidEvaluator, "raidEvaluator");
        this.workingAreaEvaluator = Objects.requireNonNull(workingAreaEvaluator, "workingAreaEvaluator");
        this.safeZoneEvaluator = Objects.requireNonNull(safeZoneEvaluator, "safeZoneEvaluator");
        this.graphFactory = Objects.requireNonNull(graphFactory, "graphFactory");
    }

    public static class EvaluationResult {
        public final LogisticsEvaluator.LogisticsResult logistics;
        public final ResourceCostEvaluator.CostResult cost;
        public final RaidResistanceEvaluator.RaidResult raid;
        public final WorkingAreaEvaluator.WorkingAreaResult workingArea;
        public final SafeZoneEvaluator.SafeZoneResult safeZone;
        public final double finalScore;

        public EvaluationResult(LogisticsEvaluator.LogisticsResult logistics,
                                ResourceCostEvaluator.CostResult cost,
                                RaidResistanceEvaluator.RaidResult raid,
                                WorkingAreaEvaluator.WorkingAreaResult workingArea,
                                SafeZoneEvaluator.SafeZoneResult safeZone,
                                double finalScore) {
            this.logistics = logistics;
            this.cost = cost;
            this.raid = raid;
            this.workingArea = workingArea;
            this.safeZone = safeZone;
            this.finalScore = finalScore;
        }

        @Override
        public String toString() {
            return String.format(
                "=== House Evaluation ===\n%s\n%s\n%s\n%s\n%s\nFinal Score: %.2f",
                logistics, cost, raid, workingArea, safeZone, finalScore
            );
        }
    }

    public void setWeights(double logistics, double cost, double raid, double workingArea) {
        setWeights(logistics, cost, raid, workingArea, 0.0);
    }

    @Override
    public void setWeights(double logistics, double cost, double raid, double workingArea, double safeZone) {
        this.logisticsWeight = logistics;
        this.costWeight = cost;
        this.raidWeight = raid;
        this.workingAreaWeight = workingArea;
        this.safeZoneWeight = safeZone;
    }

    /**
     * Evaluate the house described by the GridModel.
     */
    @Override
    public EvaluationResult evaluate(GridModel gridModel) {
        List<BuildingBlock> blocks = gridModel.getAllBlocks();

        // 1. Build 3D graph
        HouseGraph graph = graphFactory.create();
        graph.buildGraph(blocks);

        // 2. Evaluate performance criteria first
        LogisticsEvaluator.LogisticsResult logResult = logisticsEvaluator.evaluate(graph);
        WorkingAreaEvaluator.WorkingAreaResult workingAreaResult = workingAreaEvaluator.evaluate(graph, blocks);
        SafeZoneEvaluator.SafeZoneResult safeZoneResult = safeZoneEvaluator.evaluate(graph);
        
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
        // safe zone: higher = more closed blocks protected from outside walking access
        double finalScore = logisticsWeight * logResult.score +
                            costWeight * costResult.score +
                            raidWeight * raidResult.score +
                            workingAreaWeight * workingAreaResult.score +
                            safeZoneWeight * safeZoneResult.score;

        return new EvaluationResult(logResult, costResult, raidResult, workingAreaResult, safeZoneResult, finalScore);
    }
}
