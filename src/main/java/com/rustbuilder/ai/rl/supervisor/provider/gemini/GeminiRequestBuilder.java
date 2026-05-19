package com.rustbuilder.ai.rl.supervisor.provider.gemini;

import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorActionDirection;
import com.rustbuilder.ai.rl.supervisor.serialization.SimpleJson;
import com.rustbuilder.ai.rl.supervisor.serialization.SupervisorJson;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorObservation;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Converts a supervisor observation into Gemini's generateContent request body.
 */
public final class GeminiRequestBuilder {
    private final LlmSupervisorConfig supervisorConfig;
    private final GeminiPromptCatalog promptCatalog;

    public GeminiRequestBuilder(LlmSupervisorConfig supervisorConfig, GeminiPromptCatalog promptCatalog) {
        this.supervisorConfig = supervisorConfig != null ? supervisorConfig.clone() : new LlmSupervisorConfig();
        this.promptCatalog = promptCatalog != null ? promptCatalog : new GeminiPromptCatalog();
    }

    public String buildRequestBody(String stage, SupervisorObservation observation, String priorProposalJson) {
        return buildRequestBody(stage, observation, priorProposalJson, null);
    }

    public String buildRequestBody(String stage,
                                   SupervisorObservation observation,
                                   String priorProposalJson,
                                   SupervisorActionDirection selectedDirection) {
        List<String> baseAllowedActions = SupervisorJson.allowedActionsFor(observation);
        List<String> allowedActions = selectedDirection != null
            ? selectedDirection.allowedActionNames(baseAllowedActions)
            : baseAllowedActions;
        Map<String, Object> observationJson = SupervisorJson.observationToMap(observation, allowedActions);
        if (selectedDirection != null) {
            observationJson.put("selectedActionDirection", selectedDirection.toMap(baseAllowedActions));
        }
        List<String> allowedDirections = SupervisorActionDirection.allowedDirectionsFor(baseAllowedActions)
            .stream()
            .map(Enum::name)
            .toList();

        Map<String, Object> userPayload = new LinkedHashMap<>();
        userPayload.put("task", promptCatalog.taskForStage(stage));
        userPayload.put("stage", stage);
        userPayload.put("cautionLevel", supervisorConfig.getCautionLevel().name());
        userPayload.put("decisionMode", supervisorConfig.getDecisionMode().name());
        userPayload.put("supervisorPolicy", promptCatalog.supervisorPolicy(supervisorConfig));
        userPayload.put("observation", observationJson);
        if ("SELECT_DIRECTION".equals(stage)) {
            userPayload.put("direction_schema", promptCatalog.directionSchemaText(allowedDirections));
        } else {
            userPayload.put("decision_schema", promptCatalog.decisionSchemaText());
        }
        if (priorProposalJson != null && !priorProposalJson.isBlank()) {
            userPayload.put("priorProposal", SimpleJson.parse(priorProposalJson));
        }

        Map<String, Object> request = new LinkedHashMap<>();
        request.put("systemInstruction", Map.of(
            "parts", List.of(Map.of("text", promptCatalog.systemPrompt(stage, supervisorConfig)))));
        request.put("contents", List.of(Map.of(
            "role", "user",
            "parts", List.of(Map.of("text", SimpleJson.stringify(userPayload))))));
        request.put("generationConfig", Map.of(
            "temperature", promptCatalog.temperatureFor(supervisorConfig),
            "responseMimeType", "application/json",
            "responseSchema", promptCatalog.responseSchema(stage, allowedActions, allowedDirections)));
        return SimpleJson.stringify(request);
    }
}
