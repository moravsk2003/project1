package com.rustbuilder.ai.rl.policy.multidiscrete;

/**
 * [RL REDESIGN]
 * Policy interface for making decisions in a multi-discrete 5-phase action context.
 */
public interface MultiDiscretePhasePolicy {
    
    /**
     * Executes the orchestration of all 5 decision phases 
     * to produce a single MultiDiscreteAction.
     * Optionally records phase progression if observer is provided.
     */
    MultiDiscreteAction chooseAction(MultiDiscretePhaseContext context, MultiDiscreteStateObserver observer);
}
