package com.rustbuilder.ai.rl.supervisor.ports;


import com.rustbuilder.ai.rl.application.RLTrainingService;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorObservation;
/**
 * Boundary for an external LLM supervisor. Implementations should return only
 * structured decisions; RLTrainingService validates every decision before use.
 */
public interface LlmSupervisor {
    SupervisorDecision review(SupervisorObservation observation) throws Exception;
}
