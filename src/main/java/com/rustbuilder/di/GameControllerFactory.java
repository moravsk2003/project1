package com.rustbuilder.di;

import com.google.inject.Inject;
import com.google.inject.Provider;
import com.rustbuilder.controller.GameController;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.service.evaluator.HouseEvaluationService;
import com.rustbuilder.service.physics.SnappingService;
import com.rustbuilder.ui.GameCanvas;

public final class GameControllerFactory {
    private final Provider<HouseEvaluationService> houseEvaluatorProvider;

    @Inject
    public GameControllerFactory(Provider<HouseEvaluationService> houseEvaluatorProvider) {
        this.houseEvaluatorProvider = houseEvaluatorProvider;
    }

    public GameController create(GridModel gridModel, GameCanvas gameCanvas) {
        return new GameController(
                gridModel,
                gameCanvas,
                new SnappingService(gridModel),
                houseEvaluatorProvider.get());
    }
}
