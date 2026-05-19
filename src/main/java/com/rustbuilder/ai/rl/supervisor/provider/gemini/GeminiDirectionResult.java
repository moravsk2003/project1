package com.rustbuilder.ai.rl.supervisor.provider.gemini;

import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDirectionDecision;

import java.util.Map;

public final class GeminiDirectionResult {
    private final SupervisorDirectionDecision directionDecision;
    private final String directionJson;
    private final Map<String, Object> diagnostics;

    public GeminiDirectionResult(SupervisorDirectionDecision directionDecision,
                                 String directionJson,
                                 Map<String, Object> diagnostics) {
        this.directionDecision = directionDecision;
        this.directionJson = directionJson != null ? directionJson : "";
        this.diagnostics = diagnostics != null ? Map.copyOf(diagnostics) : Map.of();
    }

    public SupervisorDirectionDecision getDirectionDecision() {
        return directionDecision;
    }

    public String getDirectionJson() {
        return directionJson;
    }

    public Map<String, Object> getDiagnostics() {
        return diagnostics;
    }
}
