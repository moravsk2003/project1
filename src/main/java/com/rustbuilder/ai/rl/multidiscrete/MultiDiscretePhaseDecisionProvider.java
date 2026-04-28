package com.rustbuilder.ai.rl.multidiscrete;

/**
 * [RL REDESIGN]
 * A strategy contract for providing concrete multi-discrete phase indices 
 * for a specific development context. 
 * 
 * This enables decoupling between WHO makes the decision (heuristic vs learned model)
 * and HOW the policy layer builds the action.
 */
public interface MultiDiscretePhaseDecisionProvider {
    
    /**
     * Supplies chosen phase indices for a given situation.
     */
    MultiDiscretePhaseDecision provideDecision(MultiDiscretePhaseContext context);
}
