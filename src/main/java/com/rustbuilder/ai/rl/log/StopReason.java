package com.rustbuilder.ai.rl.log;

/**
 * Reasons why a training episode terminated.
 */
public enum StopReason {
    /** Agent explicitly chose the STOP action. */
    AGENT_STOP,
    /** Maximum allowed steps reached. */
    MAX_STEPS,
    /** No valid actions found by masking/heuristics. */
    NO_VALID_ACTIONS,
    /** All mask probability became zero. */
    MASK_EMPTY,
    /** Action failed consistency/physics checks at runtime. */
    PHYSICS_FAILED,
    /** Runtime exception during execution. */
    ERROR,
    /** Agent stalled (no growth for many steps). */
    STAGNATION,
    /** Initial or fallback state. */
    UNKNOWN
}
