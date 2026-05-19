package com.rustbuilder.ai.rl.supervisor.provider.gemini;

import com.rustbuilder.ai.rl.domain.RLRewardConfig;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorActionDirection;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDirectionDecision;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorObservation;
import com.rustbuilder.ai.rl.supervisor.serialization.SupervisorJson;

import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Supplier;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Runs primary/fallback Gemini stages and converts them into supervisor decisions.
 */
public final class GeminiSupervisorStageRunner {
    private static final AtomicInteger fallbackSkipCounter = new AtomicInteger(0);
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
        return requestDecision(stage, observation, priorProposalJson, null);
    }

    public GeminiStageResult requestDecision(String stage,
                                             SupervisorObservation observation,
                                             String priorProposalJson,
                                             SupervisorActionDirection selectedDirection) {
        String requestBody = requestBuilder.buildRequestBody(stage, observation, priorProposalJson, selectedDirection);
        Map<String, Object> diagnostics = new LinkedHashMap<>();
        diagnostics.put("stage", stage);
        if (selectedDirection != null) {
            diagnostics.put("selectedDirection", selectedDirection.name());
        }

        boolean skipPrimary = false;
        int currentSkip = fallbackSkipCounter.get();
        if (currentSkip > 0) {
            int remaining = fallbackSkipCounter.decrementAndGet();
            if (remaining >= 0) {
                diagnostics.put("rateLimitSkip", true);
                diagnostics.put("rateLimitSkipsRemaining", remaining);
                skipPrimary = true;
            } else {
                fallbackSkipCounter.compareAndSet(remaining, 0);
            }
        }

        GeminiModelResult primary = null;
        if (!skipPrimary) {
            primary = modelClient.callModel(primaryModel, requestBody);
            diagnostics.put("primary", primary.toDiagnostics());
            if (primary.getDecisionJson() != null) {
                try {
                    diagnostics.put("selectedModel", primary.getModelName());
                    return new GeminiStageResult(parseDecision(primary.getDecisionJson()), primary.getDecisionJson(), diagnostics);
                } catch (RuntimeException e) {
                    primary = primary.withError(primary.getModelName() + " returned invalid decision JSON: " + e.getMessage());
                    diagnostics.put("primary", primary.toDiagnostics());
                }
            } else if (primary.getError() != null && primary.getError().contains("HTTP 429")) {
                fallbackSkipCounter.set(10);
                diagnostics.put("rateLimitTriggered", true);
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
            String primaryErr = primary != null ? primary.getError() : "skipped (rate limit)";
            diagnostics.put("error", "Gemini primary and fallback failed: " + primaryErr + "; " + fallback.getError());
            return new GeminiStageResult(SupervisorDecision.keepGoing(
                "Gemini primary and fallback failed: " + primaryErr + "; " + fallback.getError()),
                "",
                diagnostics);
        }

        String primaryErr = primary != null ? primary.getError() : "skipped (rate limit)";
        diagnostics.put("error", "Gemini request failed: " + primaryErr);
        return new GeminiStageResult(SupervisorDecision.keepGoing("Gemini request failed: " + primaryErr),
            "",
            diagnostics);
    }

    public GeminiDirectionResult requestDirection(String stage,
                                                  SupervisorObservation observation) {
        String requestBody = requestBuilder.buildRequestBody(stage, observation, null, null);
        Map<String, Object> diagnostics = new LinkedHashMap<>();
        diagnostics.put("stage", stage);

        boolean skipPrimary = false;
        int currentSkip = fallbackSkipCounter.get();
        if (currentSkip > 0) {
            int remaining = fallbackSkipCounter.decrementAndGet();
            if (remaining >= 0) {
                diagnostics.put("rateLimitSkip", true);
                diagnostics.put("rateLimitSkipsRemaining", remaining);
                skipPrimary = true;
            } else {
                fallbackSkipCounter.compareAndSet(remaining, 0);
            }
        }

        GeminiModelResult primary = null;
        if (!skipPrimary) {
            primary = modelClient.callModel(primaryModel, requestBody);
            diagnostics.put("primary", primary.toDiagnostics());
            if (primary.getDecisionJson() != null) {
                try {
                    diagnostics.put("selectedModel", primary.getModelName());
                    return new GeminiDirectionResult(parseDirection(primary.getDecisionJson(), observation),
                        primary.getDecisionJson(),
                        diagnostics);
                } catch (RuntimeException e) {
                    primary = primary.withError(primary.getModelName() + " returned invalid direction JSON: " + e.getMessage());
                    diagnostics.put("primary", primary.toDiagnostics());
                }
            } else if (primary.getError() != null && primary.getError().contains("HTTP 429")) {
                fallbackSkipCounter.set(10);
                diagnostics.put("rateLimitTriggered", true);
            }
        }

        if (fallbackModel != null && !fallbackModel.isBlank() && !fallbackModel.equals(primaryModel)) {
            GeminiModelResult fallback = modelClient.callModel(fallbackModel, requestBody);
            diagnostics.put("fallback", fallback.toDiagnostics());
            if (fallback.getDecisionJson() != null) {
                try {
                    diagnostics.put("selectedModel", fallback.getModelName());
                    return new GeminiDirectionResult(parseDirection(fallback.getDecisionJson(), observation),
                        fallback.getDecisionJson(),
                        diagnostics);
                } catch (RuntimeException e) {
                    fallback = fallback.withError(fallback.getModelName() + " returned invalid direction JSON: " + e.getMessage());
                    diagnostics.put("fallback", fallback.toDiagnostics());
                }
            }
            String primaryErr = primary != null ? primary.getError() : "skipped (rate limit)";
            diagnostics.put("error", "Gemini primary and fallback failed: " + primaryErr + "; " + fallback.getError());
            return new GeminiDirectionResult(null, "", diagnostics);
        }

        String primaryErr = primary != null ? primary.getError() : "skipped (rate limit)";
        diagnostics.put("error", "Gemini request failed: " + primaryErr);
        return new GeminiDirectionResult(null, "", diagnostics);
    }

    private SupervisorDecision parseDecision(String decisionJson) {
        return responseParser.parseDecision(decisionJson, currentRewardConfigSupplier);
    }

    private SupervisorDirectionDecision parseDirection(String directionJson, SupervisorObservation observation) {
        List<SupervisorActionDirection> allowedDirections =
            SupervisorActionDirection.allowedDirectionsFor(SupervisorJson.allowedActionsFor(observation));
        return responseParser.parseDirection(directionJson, allowedDirections);
    }
}
