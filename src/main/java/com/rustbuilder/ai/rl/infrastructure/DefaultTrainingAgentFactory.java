package com.rustbuilder.ai.rl.infrastructure;


import com.rustbuilder.ai.rl.environment.spec.ActionSpaceSpec;
import com.rustbuilder.ai.rl.environment.spec.StateEncodingSpec;
import com.rustbuilder.ai.rl.ports.TrainingAgentFactory;
import com.rustbuilder.ai.rl.environment.spec.EncodingRuntimeConfig;
import com.rustbuilder.ai.rl.environment.state.StateRepresentationEncoder;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscreteDQNAgent;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscreteExperienceReplay;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscretePhaseDecisionProvider;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscretePhasePolicy;
import com.rustbuilder.ai.rl.policy.multidiscrete.NeuralMultiDiscreteDecisionProvider;
import com.rustbuilder.ai.rl.policy.multidiscrete.ProvidedPhaseMultiDiscretePolicy;

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
