package com.rustbuilder.ai.rl.supervisor.provider.gemini;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Normalized result from a single Gemini model call.
 */
final class GeminiModelResult {
    private final String modelName;
    private final String decisionJson;
    private final String rawText;
    private final String error;

    private GeminiModelResult(String modelName, String decisionJson, String rawText, String error) {
        this.modelName = modelName != null ? modelName : "";
        this.decisionJson = decisionJson;
        this.rawText = rawText != null ? rawText : "";
        this.error = error;
    }

    static GeminiModelResult decision(String modelName, String decisionJson, String rawText) {
        return new GeminiModelResult(modelName, decisionJson, rawText, null);
    }

    static GeminiModelResult error(String modelName, String error) {
        return new GeminiModelResult(modelName, null, "", error != null ? error : "unknown error");
    }

    GeminiModelResult withError(String error) {
        return new GeminiModelResult(modelName, decisionJson, rawText, error != null ? error : "unknown error");
    }

    String getModelName() {
        return modelName;
    }

    String getDecisionJson() {
        return decisionJson;
    }

    String getError() {
        return error;
    }

    Map<String, Object> toDiagnostics() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("model", modelName);
        if (decisionJson != null) {
            map.put("decisionJson", decisionJson);
        }
        if (!rawText.isBlank()) {
            map.put("rawText", rawText);
        }
        if (error != null && !error.isBlank()) {
            map.put("error", error);
        }
        return map;
    }
}
