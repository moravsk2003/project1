package com.rustbuilder.ai.rl;

import com.rustbuilder.ai.rl.env.spec.EncodingRuntimeConfig;
import com.rustbuilder.ai.rl.env.state.StateRepresentationEncoder;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteDQNAgent;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscretePhaseDecisionProvider;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscretePhasePolicy;
import com.rustbuilder.ai.rl.multidiscrete.NeuralMultiDiscreteDecisionProvider;
import com.rustbuilder.ai.rl.multidiscrete.ProvidedPhaseMultiDiscretePolicy;

public final class DefaultTrainingAgentFactory implements TrainingAgentFactory {

    @Override
    public MultiDiscreteExperienceReplay createReplay(int capacity) {
        return new MultiDiscreteExperienceReplay(capacity);
    }

    @Override
    public MultiDiscreteDQNAgent createAgent(EncodingRuntimeConfig config, boolean use2dCnn) {
        return new MultiDiscreteDQNAgent(config.stateEncodingSpec, config.actionSpaceSpec, use2dCnn);
    }

    @Override
    public NeuralMultiDiscreteDecisionProvider createNeuralProvider(
            MultiDiscreteDQNAgent agent,
            StateRepresentationEncoder encoder) {
        return new NeuralMultiDiscreteDecisionProvider(agent, encoder);
    }

    @Override
    public MultiDiscretePhasePolicy createPolicy(MultiDiscretePhaseDecisionProvider provider) {
        return new ProvidedPhaseMultiDiscretePolicy(provider);
    }
}
