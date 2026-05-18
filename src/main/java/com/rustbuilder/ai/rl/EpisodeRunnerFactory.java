package com.rustbuilder.ai.rl;

import com.google.inject.Inject;
import com.rustbuilder.ai.rl.env.state.StateRepresentationEncoder;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteDQNAgent;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscretePhasePolicy;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteStateObserver;
import com.rustbuilder.model.GridModelFactory;
import com.rustbuilder.service.evaluator.HouseEvaluationService;
import java.util.Objects;
import java.util.Random;

public final class EpisodeRunnerFactory {

    private final GridModelFactory gridModelFactory;

    @Inject
    public EpisodeRunnerFactory(GridModelFactory gridModelFactory) {
        this.gridModelFactory = Objects.requireNonNull(gridModelFactory, "gridModelFactory");
    }

    public EpisodeEvaluator createEvaluator(HouseEvaluationService evaluator, RLRewardConfig rewardConfig) {
        return new EpisodeEvaluator(Objects.requireNonNull(evaluator, "evaluator"), rewardConfig);
    }

    public EpisodeRunner createRunner(MultiDiscreteDQNAgent multiDiscreteAgent,
                                      MultiDiscreteExperienceReplay multiDiscreteMemory,
                                      MultiDiscretePhasePolicy multiDiscretePolicy,
                                      MultiDiscreteStateObserver multiDiscreteObserver,
                                      Random random,
                                      RLRewardConfig rewardConfig,
                                      RLTrainingLogger logger,
                                      RLTrainingService rlService,
                                      StateRepresentationEncoder stateEncoder,
                                      HouseEvaluationService evaluator) {
        return new EpisodeRunner(
                multiDiscreteAgent,
                multiDiscreteMemory,
                multiDiscretePolicy,
                multiDiscreteObserver,
                random,
                rewardConfig,
                logger,
                rlService,
                stateEncoder,
                Objects.requireNonNull(evaluator, "evaluator"),
                gridModelFactory);
    }
}
