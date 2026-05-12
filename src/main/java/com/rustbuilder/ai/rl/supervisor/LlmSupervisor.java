package com.rustbuilder.ai.rl.supervisor;

/**
 * Boundary for an external LLM supervisor. Implementations should return only
 * structured decisions; RLTrainingService validates every decision before use.
 */
public interface LlmSupervisor {
    SupervisorDecision review(SupervisorObservation observation) throws Exception;
}
