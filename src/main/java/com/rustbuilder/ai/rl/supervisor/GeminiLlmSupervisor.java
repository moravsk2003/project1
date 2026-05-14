package com.rustbuilder.ai.rl.supervisor;

import com.rustbuilder.ai.rl.RLRewardConfig;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Native Java Gemini adapter for the LLM supervisor.
 */
public class GeminiLlmSupervisor implements LlmSupervisor, LlmSupervisorDiagnostics {
    private static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(30);
    public static final String DEFAULT_MODEL = "gemini-2.5-flash";
    public static final String DEFAULT_FALLBACK_MODEL = "gemma-4-31b-it";
    private static final String DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

    private static final String BASE_SYSTEM_PROMPT = """
        You supervise reinforcement learning for a Rust base builder.
        Return exactly one JSON object and no markdown.
        Use only actions listed in observation.allowedActions.
        Use STOP_TRAINING only when the run is clearly wasting the remaining budget.
        Use START_NEW_RUN only when observation.trainingContext.trainingRunning is false and provide modelName.
        During active training, stopReason is the last episode stop reason, not proof that the whole run stopped.
        Never use START_NEW_RUN to continue an active run; use KEEP_GOING unless a listed training action is justified.
        Always set callFrequency to VERY_SOON, SOON, MEDIUM, or LONG to choose the next review cadence.
        Do not omit callFrequency, even when the action is KEEP_GOING.
        Use faster cadence while unstable or after branch changes; use longer cadence when learning is stable.
        Use REQUEST_HISTORICAL_REPORT only when idle and you need prior epoch trends before starting a run.
        Use PROMOTE_BRANCH only after observation.trendMetrics.latestBranchComparison shows a candidate that should replace the live reward config.
        Use JUMP_TO_BRANCH only with a modelName from observation.trendMetrics.knownBranchModelNames.
        Never invent fields outside rewardConfig or rewardTerms.
        Formula terms may use only variables and functions from observation.rewardFormulaContract.
        Use trainingRemainingSeconds and trainingDeadline to avoid disruptive changes near the end of a run.
        Include confidence, riskLevel, expectedEffect, rollbackPlan, changeMagnitude, and requiresBranchTest when useful.
        """;

    private final HttpClient httpClient;
    private final String apiKey;
    private final String model;
    private final String fallbackModel;
    private final String baseUrl;
    private final LlmSupervisorConfig supervisorConfig;
    private final Supplier<RLRewardConfig> currentRewardConfigSupplier;
    private volatile Map<String, Object> lastDiagnostics = Map.of();

    public GeminiLlmSupervisor(String apiKey, Supplier<RLRewardConfig> currentRewardConfigSupplier) {
        this(apiKey, null, currentRewardConfigSupplier);
    }

    public GeminiLlmSupervisor(String apiKey,
                               LlmSupervisorConfig supervisorConfig,
                               Supplier<RLRewardConfig> currentRewardConfigSupplier) {
        this(apiKey,
            firstNonBlank(
                supervisorConfig != null ? supervisorConfig.getPrimaryModel() : "",
                System.getenv("GEMINI_MODEL"),
                DEFAULT_MODEL),
            firstNonBlank(
                supervisorConfig != null ? supervisorConfig.getFallbackModel() : "",
                System.getenv("GEMINI_FALLBACK_MODEL"),
                DEFAULT_FALLBACK_MODEL),
            firstNonBlank(System.getenv("GEMINI_BASE_URL"), DEFAULT_BASE_URL),
            supervisorConfig,
            currentRewardConfigSupplier,
            HttpClient.newBuilder().connectTimeout(DEFAULT_TIMEOUT).build());
    }

    GeminiLlmSupervisor(String apiKey,
                        String model,
                        String fallbackModel,
                        String baseUrl,
                        LlmSupervisorConfig supervisorConfig,
                        Supplier<RLRewardConfig> currentRewardConfigSupplier,
                        HttpClient httpClient) {
        this.apiKey = firstNonBlank(apiKey, System.getenv("GEMINI_API_KEY"), System.getenv("GOOGLE_API_KEY"));
        this.model = firstNonBlank(model, DEFAULT_MODEL);
        this.fallbackModel = firstNonBlank(fallbackModel, "");
        this.baseUrl = firstNonBlank(baseUrl, DEFAULT_BASE_URL).replaceAll("/+$", "");
        this.supervisorConfig = supervisorConfig != null ? supervisorConfig.clone() : new LlmSupervisorConfig();
        this.currentRewardConfigSupplier = currentRewardConfigSupplier != null
            ? currentRewardConfigSupplier
            : RLRewardConfig::createDefault;
        this.httpClient = httpClient != null
            ? httpClient
            : HttpClient.newBuilder().connectTimeout(DEFAULT_TIMEOUT).build();
    }

    @Override
    public SupervisorDecision review(SupervisorObservation observation) throws Exception {
        Map<String, Object> diagnostics = new LinkedHashMap<>();
        diagnostics.put("provider", "builtin:gemini");
        diagnostics.put("primaryModel", model);
        diagnostics.put("fallbackModel", fallbackModel);
        diagnostics.put("decisionMode", supervisorConfig.getDecisionMode().name());
        diagnostics.put("cautionLevel", supervisorConfig.getCautionLevel().name());
        diagnostics.put("observationJson", SupervisorJson.observationToJson(observation));
        List<Map<String, Object>> stages = new java.util.ArrayList<>();
        diagnostics.put("stages", stages);

        if (apiKey == null || apiKey.isBlank()) {
            diagnostics.put("error", "No GEMINI_API_KEY/GOOGLE_API_KEY/UI API key configured.");
            lastDiagnostics = diagnostics;
            return SupervisorDecision.keepGoing("No GEMINI_API_KEY/GOOGLE_API_KEY/UI API key configured.");
        }

        LlmSupervisorConfig.DecisionMode decisionMode = supervisorConfig.getDecisionMode();
        StageResult proposal = requestDecision("ANALYZE_AND_PROPOSE", observation, null);
        stages.add(proposal.diagnostics);
        if (decisionMode == LlmSupervisorConfig.DecisionMode.SINGLE_STEP) {
            diagnostics.put("finalStage", "ANALYZE_AND_PROPOSE");
            lastDiagnostics = diagnostics;
            return proposal.decision;
        }

        boolean twoStage = decisionMode == LlmSupervisorConfig.DecisionMode.TWO_STAGE_ALWAYS
            || isHighImpactDecision(proposal.decision, observation);
        diagnostics.put("highImpact", twoStage);
        if (!twoStage) {
            diagnostics.put("finalStage", "ANALYZE_AND_PROPOSE");
            lastDiagnostics = diagnostics;
            return proposal.decision;
        }

        StageResult finalDecision = requestDecision("FINAL_DECISION", observation, proposal.decisionJson);
        stages.add(finalDecision.diagnostics);
        diagnostics.put("finalStage", "FINAL_DECISION");
        lastDiagnostics = diagnostics;
        return finalDecision.decision;
    }

    @Override
    public Map<String, Object> getLastDiagnostics() {
        return lastDiagnostics != null ? Map.copyOf(lastDiagnostics) : Map.of();
    }

    private StageResult requestDecision(String stage, SupervisorObservation observation, String priorProposalJson) {
        String requestBody = buildRequestBody(stage, observation, priorProposalJson);
        Map<String, Object> diagnostics = new LinkedHashMap<>();
        diagnostics.put("stage", stage);

        ModelResult primary = callModel(model, requestBody);
        diagnostics.put("primary", primary.toDiagnostics());
        if (primary.decisionJson != null) {
            try {
                diagnostics.put("selectedModel", primary.modelName);
                return new StageResult(parseDecision(primary.decisionJson), primary.decisionJson, diagnostics);
            } catch (RuntimeException e) {
                primary = primary.withError(primary.modelName + " returned invalid decision JSON: " + e.getMessage());
                diagnostics.put("primary", primary.toDiagnostics());
            }
        }

        if (fallbackModel != null && !fallbackModel.isBlank() && !fallbackModel.equals(model)) {
            ModelResult fallback = callModel(fallbackModel, requestBody);
            diagnostics.put("fallback", fallback.toDiagnostics());
            if (fallback.decisionJson != null) {
                try {
                    diagnostics.put("selectedModel", fallback.modelName);
                    return new StageResult(parseDecision(fallback.decisionJson), fallback.decisionJson, diagnostics);
                } catch (RuntimeException e) {
                    fallback = fallback.withError(fallback.modelName + " returned invalid decision JSON: " + e.getMessage());
                    diagnostics.put("fallback", fallback.toDiagnostics());
                }
            }
            diagnostics.put("error", "Gemini primary and fallback failed: " + primary.error + "; " + fallback.error);
            return new StageResult(SupervisorDecision.keepGoing(
                "Gemini primary and fallback failed: " + primary.error + "; " + fallback.error),
                "",
                diagnostics);
        }

        diagnostics.put("error", "Gemini request failed: " + primary.error);
        return new StageResult(SupervisorDecision.keepGoing("Gemini request failed: " + primary.error),
            "",
            diagnostics);
    }

    private String buildRequestBody(String stage, SupervisorObservation observation, String priorProposalJson) {
        Object observationJson = SimpleJson.parse(SupervisorJson.observationToJson(observation));

        Map<String, Object> userPayload = new LinkedHashMap<>();
        userPayload.put("task", taskForStage(stage));
        userPayload.put("stage", stage);
        userPayload.put("cautionLevel", supervisorConfig.getCautionLevel().name());
        userPayload.put("decisionMode", supervisorConfig.getDecisionMode().name());
        userPayload.put("supervisorPolicy", supervisorPolicy());
        userPayload.put("observation", observationJson);
        userPayload.put("decision_schema", decisionSchemaText());
        if (priorProposalJson != null && !priorProposalJson.isBlank()) {
            userPayload.put("priorProposal", SimpleJson.parse(priorProposalJson));
        }

        Map<String, Object> request = new LinkedHashMap<>();
        request.put("systemInstruction", Map.of("parts", List.of(Map.of("text", systemPrompt(stage)))));
        request.put("contents", List.of(Map.of(
            "role", "user",
            "parts", List.of(Map.of("text", SimpleJson.stringify(userPayload))))));
        request.put("generationConfig", Map.of(
            "temperature", temperatureForCaution(),
            "responseMimeType", "application/json",
            "responseSchema", responseSchema()));
        return SimpleJson.stringify(request);
    }

    private ModelResult callModel(String modelName, String requestBody) {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/models/" + modelName + ":generateContent"))
                .timeout(DEFAULT_TIMEOUT)
                .header("x-goog-api-key", apiKey)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                return ModelResult.error(modelName, modelName + " request failed with HTTP " + response.statusCode());
            }
            String text = extractText(response.body());
            String json = parseModelJson(text);
            return json != null
                ? ModelResult.decision(modelName, json, text)
                : ModelResult.error(modelName, modelName + " returned non-JSON content");
        } catch (Exception e) {
            return ModelResult.error(modelName, modelName + " request failed: " + e.getMessage());
        }
    }

    private String taskForStage(String stage) {
        if ("FINAL_DECISION".equals(stage)) {
            return "Finalize the prior proposal into exactly one executable supervisor decision JSON. You may revise it if the proposal is risky or unsupported.";
        }
        return "Analyze this compact RL observation, diagnose the most actionable issue, and return one proposed supervisor decision JSON.";
    }

    private String systemPrompt(String stage) {
        return BASE_SYSTEM_PROMPT + "\n" + cautionPrompt() + "\n" + stagePrompt(stage);
    }

    private String cautionPrompt() {
        return switch (supervisorConfig.getCautionLevel()) {
            case CONSERVATIVE -> """
                Caution mode: CONSERVATIVE.
                Prefer KEEP_GOING unless there is clear evidence.
                Use SET_EPSILON only for small exploration adjustments.
                Use REPLACE_REWARD_CONFIG sparingly with tiny numeric changes.
                """;
            case BALANCED -> """
                Caution mode: BALANCED.
                Prefer evidence-based action over passive KEEP_GOING when metrics are stalled or regressing.
                Use bounded reward and epsilon changes, and rely on branch tests for reward tuning.
                """;
            case BOLD -> """
                Caution mode: BOLD.
                Act like a real supervisor: propose substantial reward/epsilon hypotheses when the run is stalled, degenerate, or optimizing the wrong behavior.
                Large reward changes must be branch-tested and include rollbackPlan, expectedEffect, riskLevel, and changeMagnitude.
                """;
            case EXPERIMENTAL -> """
                Caution mode: EXPERIMENTAL.
                Aggressive hypotheses are allowed for research runs, but destructive live changes are not.
                Large changes must go through candidate branches, and STOP_TRAINING requires strong budget-waste evidence.
                """;
        };
    }

    private String stagePrompt(String stage) {
        if ("FINAL_DECISION".equals(stage)) {
            return """
                Stage: FINAL_DECISION.
                You are reviewing your priorProposal. Return the final executable decision only.
                Keep the action if the proposal is well-supported; downgrade to KEEP_GOING or REQUEST_PROMOTION_CHECK if evidence is weak.
                """;
        }
        return """
            Stage: ANALYZE_AND_PROPOSE.
            First reason internally about objective, recentSupervisorDecisions, branch results, reward breakdown, invalid actions, and time budget.
            Then return one proposed decision JSON using the schema. No markdown, no extra prose.
            """;
    }

    private Map<String, Object> supervisorPolicy() {
        Map<String, Object> policy = new LinkedHashMap<>();
        policy.put("cautionLevel", supervisorConfig.getCautionLevel().name());
        policy.put("decisionMode", supervisorConfig.getDecisionMode().name());
        policy.put("largeRewardChanges", "Must be branch-tested before replacing the live branch.");
        policy.put("useKeepGoing", supervisorConfig.getCautionLevel() == LlmSupervisorConfig.CautionLevel.CONSERVATIVE
            ? "Use when evidence is weak."
            : "Use only when no actionable diagnosis exists.");
        policy.put("highImpactActions", List.of(
            "REPLACE_REWARD_CONFIG",
            "large SET_EPSILON",
            "STOP_TRAINING",
            "PROMOTE_BRANCH",
            "JUMP_TO_BRANCH",
            "START_NEW_RUN"));
        return policy;
    }

    private double temperatureForCaution() {
        return switch (supervisorConfig.getCautionLevel()) {
            case CONSERVATIVE -> 0.15;
            case BALANCED -> 0.2;
            case BOLD -> 0.35;
            case EXPERIMENTAL -> 0.45;
        };
    }

    private boolean isHighImpactDecision(SupervisorDecision decision, SupervisorObservation observation) {
        if (decision == null || decision.getAction() == null) {
            return false;
        }
        return switch (decision.getAction()) {
            case REPLACE_REWARD_CONFIG, STOP_TRAINING, START_NEW_RUN, RESTART_TRAINING,
                    PROMOTE_BRANCH, JUMP_TO_BRANCH -> true;
            case SET_EPSILON -> {
                Double proposed = decision.getProposedEpsilon();
                yield proposed != null && Math.abs(proposed - observation.epsilon) > epsilonHighImpactDelta();
            }
            default -> false;
        };
    }

    private double epsilonHighImpactDelta() {
        return switch (supervisorConfig.getCautionLevel()) {
            case CONSERVATIVE -> 0.05;
            case BALANCED -> 0.12;
            case BOLD -> 0.25;
            case EXPERIMENTAL -> 0.35;
        };
    }

    @SuppressWarnings("unchecked")
    private String extractText(String responseBody) {
        Object parsed = SimpleJson.parse(responseBody);
        if (!(parsed instanceof Map<?, ?> root)) {
            return "";
        }
        Object candidatesObject = root.get("candidates");
        if (!(candidatesObject instanceof List<?> candidates) || candidates.isEmpty()) {
            return "";
        }
        Object first = candidates.get(0);
        if (!(first instanceof Map<?, ?> candidate)) {
            return "";
        }
        Object contentObject = candidate.get("content");
        if (!(contentObject instanceof Map<?, ?> content)) {
            return "";
        }
        Object partsObject = content.get("parts");
        if (!(partsObject instanceof List<?> parts)) {
            return "";
        }
        StringBuilder text = new StringBuilder();
        for (Object partObject : parts) {
            if (partObject instanceof Map<?, ?> part) {
                Object value = part.get("text");
                if (value instanceof String s) {
                    if (text.length() > 0) {
                        text.append('\n');
                    }
                    text.append(s);
                }
            }
        }
        return text.toString().trim();
    }

    private String parseModelJson(String text) {
        if (text == null || text.isBlank()) {
            return null;
        }
        try {
            SimpleJson.parse(text);
            return text;
        } catch (RuntimeException ignored) {
            Matcher matcher = Pattern.compile("\\{[\\s\\S]*}").matcher(text);
            if (matcher.find()) {
                String candidate = matcher.group();
                try {
                    SimpleJson.parse(candidate);
                    return candidate;
                } catch (RuntimeException ignoredAgain) {
                    return null;
                }
            }
            return null;
        }
    }

    private SupervisorDecision parseDecision(String decisionJson) {
        RLRewardConfig currentRewardConfig = currentRewardConfigSupplier.get();
        SupervisorDecision decision = SupervisorJson.decisionFromJson(decisionJson, currentRewardConfig);
        if (decision.getProposedCallFrequency() == null) {
            throw new IllegalArgumentException("missing required callFrequency");
        }
        return decision;
    }

    private Map<String, Object> decisionSchemaText() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("action", "one of observation.allowedActions");
        schema.put("epsilon", "optional number for SET_EPSILON");
        schema.put("rewardConfig", "optional numeric patch for RLRewardConfig fields");
        schema.put("rewardTerms", "optional array of {name, scope: STEP|FINAL, expression}");
        schema.put("modelName", "required safe unique name for START_NEW_RUN or JUMP_TO_BRANCH");
        schema.put("reportModelName", "required for REQUEST_HISTORICAL_REPORT");
        schema.put("reportStartEpoch", "optional non-negative integer");
        schema.put("reportEndEpoch", "optional non-negative integer");
        schema.put("callFrequency", "required cadence preset: VERY_SOON=base*0.25, SOON=base*0.5, MEDIUM=base*1, LONG=base*2");
        schema.put("confidence", "optional 0.0-1.0 estimate");
        schema.put("riskLevel", "optional LOW, MEDIUM, HIGH, or CRITICAL");
        schema.put("expectedEffect", "optional short expected metric or behavior improvement");
        schema.put("rollbackPlan", "optional short rollback/checkpoint plan for risky changes");
        schema.put("changeMagnitude", "optional NONE, SMALL, MEDIUM, LARGE, or EXPERIMENTAL");
        schema.put("requiresBranchTest", "optional boolean; true for reward or large epsilon changes");
        schema.put("reason", "short explanation");
        return schema;
    }

    private Map<String, Object> responseSchema() {
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("action", Map.of(
            "type", "STRING",
            "enum", List.of(
                "KEEP_GOING",
                "SET_EPSILON",
                "REPLACE_REWARD_CONFIG",
                "REQUEST_PROMOTION_CHECK",
                "START_NEW_RUN",
                "STOP_TRAINING",
                "REQUEST_HISTORICAL_REPORT",
                "PROMOTE_BRANCH",
                "JUMP_TO_BRANCH")));
        properties.put("epsilon", Map.of("type", "NUMBER"));
        properties.put("rewardConfig", Map.of("type", "OBJECT"));
        properties.put("rewardTerms", Map.of(
            "type", "ARRAY",
            "items", Map.of(
                "type", "OBJECT",
                "properties", Map.of(
                    "name", Map.of("type", "STRING"),
                    "scope", Map.of("type", "STRING", "enum", List.of("STEP", "FINAL")),
                    "expression", Map.of("type", "STRING")))));
        properties.put("use2dCnn", Map.of("type", "BOOLEAN"));
        properties.put("modelName", Map.of("type", "STRING"));
        properties.put("reportModelName", Map.of("type", "STRING"));
        properties.put("reportStartEpoch", Map.of("type", "INTEGER"));
        properties.put("reportEndEpoch", Map.of("type", "INTEGER"));
        properties.put("callFrequency", Map.of(
            "type", "STRING",
            "enum", List.of("VERY_SOON", "SOON", "MEDIUM", "LONG")));
        properties.put("confidence", Map.of("type", "NUMBER"));
        properties.put("riskLevel", Map.of(
            "type", "STRING",
            "enum", List.of("LOW", "MEDIUM", "HIGH", "CRITICAL")));
        properties.put("expectedEffect", Map.of("type", "STRING"));
        properties.put("rollbackPlan", Map.of("type", "STRING"));
        properties.put("changeMagnitude", Map.of(
            "type", "STRING",
            "enum", List.of("NONE", "SMALL", "MEDIUM", "LARGE", "EXPERIMENTAL")));
        properties.put("requiresBranchTest", Map.of("type", "BOOLEAN"));
        properties.put("reason", Map.of("type", "STRING"));
        return Map.of(
            "type", "OBJECT",
            "properties", properties,
            "required", List.of("action", "callFrequency", "reason"),
            "propertyOrdering", List.of(
                "action", "epsilon", "rewardConfig", "rewardTerms", "use2dCnn",
                "modelName", "reportModelName", "reportStartEpoch", "reportEndEpoch",
                "callFrequency", "confidence", "riskLevel", "expectedEffect", "rollbackPlan",
                "changeMagnitude", "requiresBranchTest", "reason"));
    }

    private static String firstNonBlank(String... values) {
        if (values == null) {
            return "";
        }
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value.trim();
            }
        }
        return "";
    }

    private static final class StageResult {
        private final SupervisorDecision decision;
        private final String decisionJson;
        private final Map<String, Object> diagnostics;

        private StageResult(SupervisorDecision decision, String decisionJson, Map<String, Object> diagnostics) {
            this.decision = decision != null ? decision : SupervisorDecision.keepGoing("Supervisor returned no decision.");
            this.decisionJson = decisionJson != null ? decisionJson : "";
            this.diagnostics = diagnostics != null ? diagnostics : Map.of();
        }
    }

    private static final class ModelResult {
        private final String modelName;
        private final String decisionJson;
        private final String rawText;
        private final String error;

        private ModelResult(String modelName, String decisionJson, String rawText, String error) {
            this.modelName = modelName != null ? modelName : "";
            this.decisionJson = decisionJson;
            this.rawText = rawText != null ? rawText : "";
            this.error = error;
        }

        static ModelResult decision(String modelName, String decisionJson, String rawText) {
            return new ModelResult(modelName, decisionJson, rawText, null);
        }

        static ModelResult error(String modelName, String error) {
            return new ModelResult(modelName, null, "", error != null ? error : "unknown error");
        }

        ModelResult withError(String error) {
            return new ModelResult(modelName, decisionJson, rawText, error != null ? error : "unknown error");
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
}
