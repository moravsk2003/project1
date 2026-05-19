package com.rustbuilder.ai.rl.supervisor.provider;


import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorObservation;
import com.rustbuilder.ai.rl.supervisor.ports.LlmSupervisor;
import com.rustbuilder.ai.rl.supervisor.ports.LlmSupervisorDiagnostics;
import com.rustbuilder.ai.rl.supervisor.serialization.SupervisorJson;
import com.rustbuilder.ai.rl.domain.RLRewardConfig;
import com.rustbuilder.ai.rl.supervisor.provider.gemini.GeminiModelClient;
import com.rustbuilder.ai.rl.supervisor.provider.gemini.GeminiPromptCatalog;
import com.rustbuilder.ai.rl.supervisor.provider.gemini.GeminiRequestBuilder;
import com.rustbuilder.ai.rl.supervisor.provider.gemini.GeminiResponseParser;
import com.rustbuilder.ai.rl.supervisor.provider.gemini.GeminiStageResult;
import com.rustbuilder.ai.rl.supervisor.provider.gemini.GeminiSupervisorStageRunner;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Facade for the Java-native Gemini supervisor provider.
 */
public class GeminiLlmSupervisor implements LlmSupervisor, LlmSupervisorDiagnostics {
    private static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(30);
    public static final String DEFAULT_MODEL = "gemma-4-31b-it";
    public static final String DEFAULT_FALLBACK_MODEL = "gemini-3.1-flash-lite";
    private static final String DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

    private final HttpClient httpClient;
    private final String apiKey;
    private final String model;
    private final String fallbackModel;
    private final String baseUrl;
    private final LlmSupervisorConfig supervisorConfig;
    private final Supplier<RLRewardConfig> currentRewardConfigSupplier;
    private final GeminiPromptCatalog promptCatalog;
    private final GeminiSupervisorStageRunner stageRunner;
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
        this.promptCatalog = new GeminiPromptCatalog();

        GeminiResponseParser responseParser = new GeminiResponseParser();
        GeminiRequestBuilder requestBuilder = new GeminiRequestBuilder(this.supervisorConfig, promptCatalog);
        GeminiModelClient modelClient = new GeminiModelClient(
            this.httpClient,
            this.apiKey,
            this.baseUrl,
            DEFAULT_TIMEOUT,
            responseParser);
        this.stageRunner = new GeminiSupervisorStageRunner(
            this.model,
            this.fallbackModel,
            modelClient,
            requestBuilder,
            responseParser,
            this.currentRewardConfigSupplier);
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
        GeminiStageResult proposal = stageRunner.requestDecision("ANALYZE_AND_PROPOSE", observation, null);
        stages.add(proposal.getDiagnostics());
        if (decisionMode == LlmSupervisorConfig.DecisionMode.SINGLE_STEP) {
            diagnostics.put("finalStage", "ANALYZE_AND_PROPOSE");
            lastDiagnostics = diagnostics;
            return proposal.getDecision();
        }

        boolean twoStage = decisionMode == LlmSupervisorConfig.DecisionMode.TWO_STAGE_ALWAYS
            || promptCatalog.isHighImpactDecision(proposal.getDecision(), observation, supervisorConfig);
        diagnostics.put("highImpact", twoStage);
        if (!twoStage) {
            diagnostics.put("finalStage", "ANALYZE_AND_PROPOSE");
            lastDiagnostics = diagnostics;
            return proposal.getDecision();
        }

        GeminiStageResult finalDecision = stageRunner.requestDecision(
            "FINAL_DECISION",
            observation,
            proposal.getDecisionJson());
        stages.add(finalDecision.getDiagnostics());
        diagnostics.put("finalStage", "FINAL_DECISION");
        lastDiagnostics = diagnostics;
        return finalDecision.getDecision();
    }

    @Override
    public Map<String, Object> getLastDiagnostics() {
        return lastDiagnostics != null ? Map.copyOf(lastDiagnostics) : Map.of();
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
}
