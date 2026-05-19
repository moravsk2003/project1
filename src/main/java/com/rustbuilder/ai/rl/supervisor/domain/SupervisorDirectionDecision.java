package com.rustbuilder.ai.rl.supervisor.domain;

/**
 * Result of the first LLM call: choose the broad direction before selecting a
 * concrete supervisor action.
 */
public final class SupervisorDirectionDecision {
    private final SupervisorActionDirection direction;
    private final String reason;
    private final Double confidence;

    public SupervisorDirectionDecision(SupervisorActionDirection direction,
                                       String reason,
                                       Double confidence) {
        this.direction = direction;
        this.reason = reason != null ? reason : "";
        this.confidence = confidence;
    }

    public SupervisorActionDirection getDirection() {
        return direction;
    }

    public String getReason() {
        return reason;
    }

    public Double getConfidence() {
        return confidence;
    }
}
