package com.rustbuilder.ai.rl.supervisor;

/**
 * Safe default implementation used until a real LLM provider is connected.
 */
public class NoOpLlmSupervisor implements LlmSupervisor {
    @Override
    public SupervisorDecision review(SupervisorObservation observation) {
        return SupervisorDecision.keepGoing("No LLM supervisor provider configured.");
    }
}
