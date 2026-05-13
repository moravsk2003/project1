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
public class GeminiLlmSupervisor implements LlmSupervisor {
    private static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(30);
    private static final String DEFAULT_MODEL = "gemini-2.5-flash";
    private static final String DEFAULT_FALLBACK_MODEL = "gemma-4-31b-it";
    private static final String DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

    private static final String SYSTEM_PROMPT = """
        You supervise reinforcement learning for a Rust base builder.
        Return exactly one JSON object and no markdown.
        Use only actions listed in observation.allowedActions.
        Prefer KEEP_GOING unless the observation gives clear evidence.
        Use SET_EPSILON for small exploration adjustments.
        Use REPLACE_REWARD_CONFIG sparingly and only with small numeric changes or safe arithmetic rewardTerms.
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
        """;

    private final HttpClient httpClient;
    private final String apiKey;
    private final String model;
    private final String fallbackModel;
    private final String baseUrl;
    private final Supplier<RLRewardConfig> currentRewardConfigSupplier;

    public GeminiLlmSupervisor(String apiKey, Supplier<RLRewardConfig> currentRewardConfigSupplier) {
        this(apiKey,
            firstNonBlank(System.getenv("GEMINI_MODEL"), DEFAULT_MODEL),
            firstNonBlank(System.getenv("GEMINI_FALLBACK_MODEL"), DEFAULT_FALLBACK_MODEL),
            firstNonBlank(System.getenv("GEMINI_BASE_URL"), DEFAULT_BASE_URL),
            currentRewardConfigSupplier,
            HttpClient.newBuilder().connectTimeout(DEFAULT_TIMEOUT).build());
    }

    GeminiLlmSupervisor(String apiKey,
                        String model,
                        String fallbackModel,
                        String baseUrl,
                        Supplier<RLRewardConfig> currentRewardConfigSupplier,
                        HttpClient httpClient) {
        this.apiKey = firstNonBlank(apiKey, System.getenv("GEMINI_API_KEY"), System.getenv("GOOGLE_API_KEY"));
        this.model = firstNonBlank(model, DEFAULT_MODEL);
        this.fallbackModel = firstNonBlank(fallbackModel, "");
        this.baseUrl = firstNonBlank(baseUrl, DEFAULT_BASE_URL).replaceAll("/+$", "");
        this.currentRewardConfigSupplier = currentRewardConfigSupplier != null
            ? currentRewardConfigSupplier
            : RLRewardConfig::createDefault;
        this.httpClient = httpClient != null
            ? httpClient
            : HttpClient.newBuilder().connectTimeout(DEFAULT_TIMEOUT).build();
    }

    @Override
    public SupervisorDecision review(SupervisorObservation observation) throws Exception {
        if (apiKey == null || apiKey.isBlank()) {
            return SupervisorDecision.keepGoing("No GEMINI_API_KEY/GOOGLE_API_KEY/UI API key configured.");
        }

        String requestBody = buildRequestBody(observation);
        ModelResult primary = callModel(model, requestBody);
        if (primary.decisionJson != null) {
            try {
                return parseDecision(primary.decisionJson);
            } catch (RuntimeException e) {
                primary = ModelResult.error(model + " returned invalid decision JSON: " + e.getMessage());
            }
        }

        if (fallbackModel != null && !fallbackModel.isBlank() && !fallbackModel.equals(model)) {
            ModelResult fallback = callModel(fallbackModel, requestBody);
            if (fallback.decisionJson != null) {
                try {
                    return parseDecision(fallback.decisionJson);
                } catch (RuntimeException e) {
                    fallback = ModelResult.error(fallbackModel + " returned invalid decision JSON: " + e.getMessage());
                }
            }
            return SupervisorDecision.keepGoing(
                "Gemini primary and fallback failed: " + primary.error + "; " + fallback.error);
        }

        return SupervisorDecision.keepGoing("Gemini request failed: " + primary.error);
    }

    private String buildRequestBody(SupervisorObservation observation) {
        Object observationJson = SimpleJson.parse(SupervisorJson.observationToJson(observation));

        Map<String, Object> userPayload = new LinkedHashMap<>();
        userPayload.put("task", "Review this compact RL observation and return one supervisor decision.");
        userPayload.put("observation", observationJson);
        userPayload.put("decision_schema", decisionSchemaText());

        Map<String, Object> request = new LinkedHashMap<>();
        request.put("systemInstruction", Map.of("parts", List.of(Map.of("text", SYSTEM_PROMPT))));
        request.put("contents", List.of(Map.of(
            "role", "user",
            "parts", List.of(Map.of("text", SimpleJson.stringify(userPayload))))));
        request.put("generationConfig", Map.of(
            "temperature", 0.2,
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
                return ModelResult.error(modelName + " request failed with HTTP " + response.statusCode());
            }
            String text = extractText(response.body());
            String json = parseModelJson(text);
            return json != null
                ? ModelResult.decision(json)
                : ModelResult.error(modelName + " returned non-JSON content");
        } catch (Exception e) {
            return ModelResult.error(modelName + " request failed: " + e.getMessage());
        }
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
        properties.put("reason", Map.of("type", "STRING"));
        return Map.of(
            "type", "OBJECT",
            "properties", properties,
            "required", List.of("action", "callFrequency", "reason"),
            "propertyOrdering", List.of(
                "action", "epsilon", "rewardConfig", "rewardTerms", "use2dCnn",
                "modelName", "reportModelName", "reportStartEpoch", "reportEndEpoch",
                "callFrequency", "reason"));
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

    private static final class ModelResult {
        private final String decisionJson;
        private final String error;

        private ModelResult(String decisionJson, String error) {
            this.decisionJson = decisionJson;
            this.error = error;
        }

        static ModelResult decision(String decisionJson) {
            return new ModelResult(decisionJson, null);
        }

        static ModelResult error(String error) {
            return new ModelResult(null, error != null ? error : "unknown error");
        }
    }
}
