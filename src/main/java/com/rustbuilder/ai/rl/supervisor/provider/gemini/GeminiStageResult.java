package com.rustbuilder.ai.rl.supervisor.provider.gemini;

import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;

import java.util.Map;

/**
 * Result of one supervisor reasoning stage.
 */
public final class GeminiStageResult {
    private final SupervisorDecision decision;
    private final String decisionJson;
    private final Map<String, Object> diagnostics;

    GeminiStageResult(SupervisorDecision decision, String decisionJson, Map<String, Object> diagnostics) {
        this.decision = decision != null ? decision : SupervisorDecision.keepGoing("Supervisor returned no decision.");
        this.decisionJson = decisionJson != null ? decisionJson : "";
        this.diagnostics = diagnostics != null ? diagnostics : Map.of();
    }

    public SupervisorDecision getDecision() {
        return decision;
    }

    public String getDecisionJson() {
        return decisionJson;
    }

    public Map<String, Object> getDiagnostics() {
        return diagnostics;
    }
}
