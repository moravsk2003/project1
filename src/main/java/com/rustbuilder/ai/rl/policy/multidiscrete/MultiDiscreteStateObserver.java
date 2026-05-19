package com.rustbuilder.ai.rl.policy.multidiscrete;

/**
 * [RL REDESIGN]
 * Observer used to monitor and record the individual phases of multi-discrete action assembly.
 * Enables the neural network to learn phase-by-phase dependencies and aids in debugging.
 */
public interface MultiDiscreteStateObserver {

    /**
     * Called when a specific phase of the action assembly is decided.
     *
     * @param context the current generation context
     * @param phaseName human-readable name of the phase (e.g., "TYPE", "POSITION_X")
     * @param phaseIndex the chronological index of the phase (0..5)
     * @param value the selected value for this phase
     */
    void observePhase(MultiDiscretePhaseContext context, String phaseName, int phaseIndex, int value);

    /**
     * Called when the entire 5-phase action has been assembled and is ready to be executed.
     *
     * @param context the context just before the action is taken
     * @param action the finalized multi-discrete action
     */
    void onActionAssembled(MultiDiscretePhaseContext context, MultiDiscreteAction action);
}
