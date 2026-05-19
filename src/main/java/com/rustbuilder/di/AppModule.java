package com.rustbuilder.di;

import com.google.inject.AbstractModule;
import com.google.inject.Provider;
import com.google.inject.Provides;
import com.google.inject.Singleton;
import com.rustbuilder.ai.ea.application.GeneticAlgorithmService;
import com.rustbuilder.ai.rl.infrastructure.DefaultStateEncoderFactory;
import com.rustbuilder.ai.rl.infrastructure.DefaultTrainingAgentFactory;
import com.rustbuilder.ai.rl.application.EpisodeRunnerFactory;
import com.rustbuilder.ai.rl.application.RLTrainingService;
import com.rustbuilder.ai.rl.application.RLTrainingServiceFactory;
import com.rustbuilder.ai.rl.ports.StateEncoderFactory;
import com.rustbuilder.ai.rl.ports.TrainingAgentFactory;
import com.rustbuilder.ai.rl.supervisor.application.RLBranchComparator;
import com.rustbuilder.ai.rl.supervisor.application.RLDualTrainingCoordinator;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.GridModelFactory;
import com.rustbuilder.service.evaluator.HouseEvaluationService;
import com.rustbuilder.service.evaluator.HouseEvaluatorFactory;
import java.util.Random;

public final class AppModule extends AbstractModule {

    @Override
    protected void configure() {
        bind(GridModelFactory.class).toInstance(GridModel::new);
        bind(StateEncoderFactory.class).to(DefaultStateEncoderFactory.class).in(Singleton.class);
        bind(TrainingAgentFactory.class).to(DefaultTrainingAgentFactory.class).in(Singleton.class);
        bind(HouseEvaluatorFactory.class).in(Singleton.class);
        bind(RLBranchComparator.class);
        bind(GameControllerFactory.class).in(Singleton.class);
    }

    @Provides
    HouseEvaluationService provideHouseEvaluationService(HouseEvaluatorFactory factory) {
        return factory.create();
    }

    @Provides
    Random provideRandom() {
        return new Random();
    }

    @Provides
    GeneticAlgorithmService provideGeneticAlgorithmService(HouseEvaluationService evaluator,
                                                           GridModelFactory gridModelFactory,
                                                           Random random) {
        return new GeneticAlgorithmService(evaluator, gridModelFactory, random);
    }

    @Provides
    RLTrainingServiceFactory provideRLTrainingServiceFactory(Provider<RLTrainingService> provider) {
        return provider::get;
    }

    @Provides
    RLDualTrainingCoordinator provideRLDualTrainingCoordinator(RLTrainingServiceFactory trainingServiceFactory) {
        return new RLDualTrainingCoordinator(trainingServiceFactory);
    }

    @Provides
    RLTrainingService provideRLTrainingService(HouseEvaluationService evaluator,
                                               GridModelFactory gridModelFactory,
                                               Random random,
                                               StateEncoderFactory stateEncoderFactory,
                                               TrainingAgentFactory trainingAgentFactory,
                                               EpisodeRunnerFactory episodeRunnerFactory,
                                               RLTrainingServiceFactory branchTrainingServiceFactory,
                                               Provider<RLDualTrainingCoordinator> dualTrainingCoordinatorProvider,
                                               Provider<RLBranchComparator> branchComparatorProvider) {
        return new RLTrainingService(
                evaluator,
                gridModelFactory,
                random,
                stateEncoderFactory,
                trainingAgentFactory,
                episodeRunnerFactory,
                branchTrainingServiceFactory,
                dualTrainingCoordinatorProvider::get,
                branchComparatorProvider::get);
    }
}
