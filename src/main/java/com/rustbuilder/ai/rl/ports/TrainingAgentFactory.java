package com.rustbuilder.ai.rl.ports;

import com.rustbuilder.ai.rl.environment.spec.EncodingRuntimeConfig;
import com.rustbuilder.ai.rl.environment.state.StateRepresentationEncoder;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscreteDQNAgent;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscreteExperienceReplay;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscretePhaseDecisionProvider;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscretePhasePolicy;
import com.rustbuilder.ai.rl.policy.multidiscrete.NeuralMultiDiscreteDecisionProvider;

public interface TrainingAgentFactory {
    MultiDiscreteExperienceReplay createReplay(int capacity);

    MultiDiscreteDQNAgent createAgent(EncodingRuntimeConfig config, boolean use2dCnn);

    NeuralMultiDiscreteDecisionProvider createNeuralProvider(
            MultiDiscreteDQNAgent agent,
            StateRepresentationEncoder encoder);

    MultiDiscretePhasePolicy createPolicy(MultiDiscretePhaseDecisionProvider provider);
}
