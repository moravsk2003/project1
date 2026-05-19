package com.rustbuilder.ai.rl.supervisor.provider.gemini;

import com.rustbuilder.ai.rl.domain.RLRewardConfig;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorObservation;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Runs primary/fallback Gemini stages and converts them into supervisor decisions.
 */
public final class GeminiSupervisorStageRunner {
    private final String primaryModel;
    private final String fallbackModel;
    private final GeminiModelClient modelClient;
    private final GeminiRequestBuilder requestBuilder;
    private final GeminiResponseParser responseParser;
    private final Supplier<RLRewardConfig> currentRewardConfigSupplier;

    public GeminiSupervisorStageRunner(String primaryModel,
                                       String fallbackModel,
                                       GeminiModelClient modelClient,
                                       GeminiRequestBuilder requestBuilder,
                                       GeminiResponseParser responseParser,
                                       Supplier<RLRewardConfig> currentRewardConfigSupplier) {
        this.primaryModel = primaryModel != null ? primaryModel : "";
        this.fallbackModel = fallbackModel != null ? fallbackModel : "";
        this.modelClient = modelClient;
        this.requestBuilder = requestBuilder;
        this.responseParser = responseParser != null ? responseParser : new GeminiResponseParser();
        this.currentRewardConfigSupplier = currentRewardConfigSupplier != null
            ? currentRewardConfigSupplier
            : RLRewardConfig::createDefault;
    }

    public GeminiStageResult requestDecision(String stage,
                                             SupervisorObservation observation,
                                             String priorProposalJson) {
        String requestBody = requestBuilder.buildRequestBody(stage, observation, priorProposalJson);
        Map<String, Object> diagnostics = new LinkedHashMap<>();
        diagnostics.put("stage", stage);

        GeminiModelResult primary = modelClient.callModel(primaryModel, requestBody);
        diagnostics.put("primary", primary.toDiagnostics());
        if (primary.getDecisionJson() != null) {
            try {
                diagnostics.put("selectedModel", primary.getModelName());
                return new GeminiStageResult(parseDecision(primary.getDecisionJson()), primary.getDecisionJson(), diagnostics);
            } catch (RuntimeException e) {
                primary = primary.withError(primary.getModelName() + " returned invalid decision JSON: " + e.getMessage());
                diagnostics.put("primary", primary.toDiagnostics());
            }
        }

        if (fallbackModel != null && !fallbackModel.isBlank() && !fallbackModel.equals(primaryModel)) {
            GeminiModelResult fallback = modelClient.callModel(fallbackModel, requestBody);
            diagnostics.put("fallback", fallback.toDiagnostics());
            if (fallback.getDecisionJson() != null) {
                try {
                    diagnostics.put("selectedModel", fallback.getModelName());
                    return new GeminiStageResult(parseDecision(fallback.getDecisionJson()), fallback.getDecisionJson(), diagnostics);
                } catch (RuntimeException e) {
                    fallback = fallback.withError(fallback.getModelName() + " returned invalid decision JSON: " + e.getMessage());
                    diagnostics.put("fallback", fallback.toDiagnostics());
                }
            }
            diagnostics.put("error", "Gemini primary and fallback failed: " + primary.getError() + "; " + fallback.getError());
            return new GeminiStageResult(SupervisorDecision.keepGoing(
                "Gemini primary and fallback failed: " + primary.getError() + "; " + fallback.getError()),
                "",
                diagnostics);
        }

        diagnostics.put("error", "Gemini request failed: " + primary.getError());
        return new GeminiStageResult(SupervisorDecision.keepGoing("Gemini request failed: " + primary.getError()),
            "",
            diagnostics);
    }

    private SupervisorDecision parseDecision(String decisionJson) {
        return responseParser.parseDecision(decisionJson, currentRewardConfigSupplier);
    }
}
