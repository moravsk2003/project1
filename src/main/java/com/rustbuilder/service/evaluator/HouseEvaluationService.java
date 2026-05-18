package com.rustbuilder.service.evaluator;

import com.rustbuilder.model.GridModel;

public interface HouseEvaluationService {

    default void setWeights(double logistics, double cost, double raid, double workingArea) {
        setWeights(logistics, cost, raid, workingArea, 0.0);
    }

    void setWeights(double logistics, double cost, double raid, double workingArea, double safeZone);

    HouseEvaluator.EvaluationResult evaluate(GridModel gridModel);
}
