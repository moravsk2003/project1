package com.rustbuilder.service.evaluator;

import com.rustbuilder.service.graph.HouseGraph;

public final class HouseEvaluatorFactory {

    public HouseEvaluator create() {
        return createDefault();
    }

    public static HouseEvaluator createDefault() {
        return new HouseEvaluator(
                new LogisticsEvaluator(),
                new ResourceCostEvaluator(),
                new RaidResistanceEvaluator(),
                new WorkingAreaEvaluator(),
                new SafeZoneEvaluator(),
                HouseGraph::new);
    }
}
