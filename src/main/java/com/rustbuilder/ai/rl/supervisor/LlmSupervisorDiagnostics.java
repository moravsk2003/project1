package com.rustbuilder.ai.rl.supervisor;

import java.util.Map;

/**
 * Optional provider-side diagnostics for supervisor calls.
 */
public interface LlmSupervisorDiagnostics {
    Map<String, Object> getLastDiagnostics();
}
