package com.rustbuilder.ai.rl;

import com.rustbuilder.ai.rl.env.spec.EncodingRuntimeConfig;
import com.rustbuilder.ai.rl.env.state.StateRepresentationEncoder;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteDQNAgent;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteExperienceReplay;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscretePhaseDecisionProvider;
import com.rustbuilder.ai.rl.multidiscrete.MultiDiscretePhasePolicy;
import com.rustbuilder.ai.rl.multidiscrete.NeuralMultiDiscreteDecisionProvider;

public interface TrainingAgentFactory {
    MultiDiscreteExperienceReplay createReplay(int capacity);

    MultiDiscreteDQNAgent createAgent(EncodingRuntimeConfig config, boolean use2dCnn);

    NeuralMultiDiscreteDecisionProvider createNeuralProvider(
            MultiDiscreteDQNAgent agent,
            StateRepresentationEncoder encoder);

    MultiDiscretePhasePolicy createPolicy(MultiDiscretePhaseDecisionProvider provider);
}
