package com.rustbuilder.di;

import com.google.inject.Guice;
import com.google.inject.Injector;
import com.rustbuilder.ai.ea.GeneticAlgorithmService;
import com.rustbuilder.ai.rl.RLTrainingService;
import com.rustbuilder.controller.GameController;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.GridModelFactory;
import com.rustbuilder.service.evaluator.HouseEvaluationService;
import com.rustbuilder.ui.GameCanvas;

/**
 * Production composition root backed by Guice.
 *
 * <p>Tests can still bypass it and inject fakes directly into constructors.</p>
 */
public final class AppComponent {

    private final Injector injector;

    public AppComponent() {
        this(Guice.createInjector(new AppModule()));
    }

    AppComponent(Injector injector) {
        this.injector = injector;
    }

    public GridModel createGridModel() {
        return injector.getInstance(GridModelFactory.class).create();
    }

    public HouseEvaluationService createHouseEvaluator() {
        return injector.getInstance(HouseEvaluationService.class);
    }

    public GameController createGameController(GridModel gridModel, GameCanvas gameCanvas) {
        return injector.getInstance(GameControllerFactory.class).create(gridModel, gameCanvas);
    }

    public GeneticAlgorithmService createGeneticAlgorithmService() {
        return injector.getInstance(GeneticAlgorithmService.class);
    }

    public RLTrainingService createRLTrainingService() {
        return injector.getInstance(RLTrainingService.class);
    }
}
