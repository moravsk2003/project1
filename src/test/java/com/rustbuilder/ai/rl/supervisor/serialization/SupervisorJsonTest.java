package com.rustbuilder.ai.rl.supervisor.serialization;


import com.rustbuilder.ai.rl.domain.reward.RewardFormulaSet;
import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorObservation;
import com.rustbuilder.ai.rl.supervisor.ports.LlmSupervisor;
import com.rustbuilder.ai.rl.supervisor.provider.GeminiLlmSupervisor;
import com.rustbuilder.ai.rl.supervisor.provider.LlmSupervisorFactory;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.rustbuilder.ai.rl.domain.RLRewardConfig;
import com.rustbuilder.ai.rl.domain.reward.RewardFormulaScope;
import java.lang.reflect.Field;
import java.util.Map;
import org.junit.jupiter.api.Test;

class SupervisorJsonTest {

    @Test
    void parsesEpsilonDecision() {
        SupervisorDecision decision = SupervisorJson.decisionFromJson(
            "{\"action\":\"SET_EPSILON\",\"epsilon\":0.25,\"reason\":\"settle\"}",
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.SET_EPSILON, decision.getAction());
        assertEquals(0.25, decision.getProposedEpsilon(), 0.0);
    }

    @Test
    void parsesRewardConfigAndFormulaTerms() {
        RLRewardConfig current = RLRewardConfig.createDefault();
        current.blockGrowthReward = 0.05;

        SupervisorDecision decision = SupervisorJson.decisionFromJson(
            "{\"action\":\"REPLACE_REWARD_CONFIG\","
                + "\"rewardConfig\":{\"blockGrowthReward\":0.12},"
                + "\"rewardTerms\":[{\"name\":\"socket_shape\",\"scope\":\"STEP\",\"expression\":\"socket_connections * 0.1\"}]}",
            current);

        RLRewardConfig config = decision.getProposedRewardConfig();
        assertEquals(SupervisorAction.REPLACE_REWARD_CONFIG, decision.getAction());
        assertEquals(0.12, config.blockGrowthReward, 0.0);
        assertEquals(0.3, config.rewardFormulaSet.evaluate(
            RewardFormulaScope.STEP,
            Map.of("socket_connections", 3.0)), 1e-9);
    }

    @Test
    void observationContractIncludesTrainingOnlyActionsAndFormulaVariables() {
        String json = SupervisorJson.observationToJson(observation(true));

        assertTrue(json.contains("\"allowedActions\":[\"KEEP_GOING\",\"SET_EPSILON\",\"REPLACE_REWARD_CONFIG\",\"SET_CURRICULUM_OBJECTIVE\",\"STOP_TRAINING\",\"REQUEST_PROMOTION_CHECK\",\"PROMOTE_BRANCH\",\"JUMP_TO_BRANCH\"]"));
        assertTrue(json.contains("\"rewardFormulaContract\""));
        assertTrue(json.contains("\"socket_connections\""));
        assertTrue(json.contains("\"final_score\""));
        assertTrue(json.contains("Always include reason and callFrequency"));
    }

    @Test
    void idleObservationContractAllowsStartingRunsAndReports() {
        String json = SupervisorJson.observationToJson(observation(false));

        assertTrue(json.contains("\"allowedActions\":[\"KEEP_GOING\",\"SET_CURRICULUM_OBJECTIVE\",\"START_NEW_RUN\",\"REQUEST_HISTORICAL_REPORT\",\"PROMOTE_BRANCH\",\"JUMP_TO_BRANCH\"]"));
    }

    @Test
    void parsesIdleAutopilotDecisions() {
        RLRewardConfig current = RLRewardConfig.createDefault();
        current.earlyStopPenaltyMult = -0.1;
        SupervisorDecision start = SupervisorJson.decisionFromJson(
            "{\"action\":\"START_NEW_RUN\",\"modelName\":\"auto_run_1\",\"use2dCnn\":true,\"reason\":\"idle\"}",
            current);
        SupervisorDecision report = SupervisorJson.decisionFromJson(
            "{\"action\":\"REQUEST_HISTORICAL_REPORT\",\"reportModelName\":\"model_a\",\"reportStartEpoch\":2,\"reportEndEpoch\":5}",
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.START_NEW_RUN, start.getAction());
        assertEquals("auto_run_1", start.getProposedModelName());
        assertEquals(Boolean.TRUE, start.getProposedUse2dCnn());
        assertNull(start.getProposedRewardConfig());
        assertEquals(SupervisorAction.REQUEST_HISTORICAL_REPORT, report.getAction());
        assertEquals("model_a", report.getReportModelName());
        assertEquals(2, report.getReportStartEpoch());
        assertEquals(5, report.getReportEndEpoch());
    }

    @Test
    void parsesBranchPromotionDecision() {
        SupervisorDecision decision = SupervisorJson.decisionFromJson(
            "{\"action\":\"PROMOTE_BRANCH\",\"reason\":\"candidate cleared comparison\"}",
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.PROMOTE_BRANCH, decision.getAction());
    }

    @Test
    void parsesBranchJumpDecision() {
        SupervisorDecision decision = SupervisorJson.decisionFromJson(
            "{\"action\":\"JUMP_TO_BRANCH\",\"modelName\":\"branch_a\",\"reason\":\"return to known branch\"}",
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.JUMP_TO_BRANCH, decision.getAction());
        assertEquals("branch_a", decision.getProposedModelName());
    }

    @Test
    void parsesCurriculumObjectiveDecision() {
        SupervisorDecision decision = SupervisorJson.decisionFromJson(
            "{\"action\":\"SET_CURRICULUM_OBJECTIVE\","
                + "\"objectiveId\":\"tc_one_layer\","
                + "\"objectiveDescription\":\"Place and enclose TC reliably\","
                + "\"successCriteria\":{\"tcPresentRateMin\":0.75,\"tcEnclosedRateMin\":0.35},"
                + "\"failureSignals\":{\"invalidActionRateMax\":0.45},"
                + "\"reason\":\"TC is present but not stable\"}",
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.SET_CURRICULUM_OBJECTIVE, decision.getAction());
        assertEquals("tc_one_layer", decision.getObjectiveId());
        assertEquals(0.75, decision.getObjectiveSuccessCriteria().get("tcPresentRateMin"), 0.0);
        assertEquals(0.45, decision.getObjectiveFailureSignals().get("invalidActionRateMax"), 0.0);
    }

    @Test
    void parsesSupervisorCallFrequencyFromDecision() {
        SupervisorDecision decision = SupervisorJson.decisionFromJson(
            "{\"action\":\"KEEP_GOING\",\"callFrequency\":\"SOON\",\"reason\":\"review sooner\"}",
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.KEEP_GOING, decision.getAction());
        assertEquals(LlmSupervisorConfig.CallFrequency.SOON, decision.getProposedCallFrequency());
    }

    @Test
    void parsesSupervisorAnalysisMetadata() {
        SupervisorDecision decision = SupervisorJson.decisionFromJson(
            "{\"action\":\"SET_EPSILON\",\"epsilon\":0.2,\"reason\":\"careful change\","
                + "\"confidence\":0.72,"
                + "\"riskLevel\":\"medium\","
                + "\"expectedEffect\":\"less random wandering\","
                + "\"rollbackPlan\":\"restore epsilon\","
                + "\"changeMagnitude\":\"small\","
                + "\"requiresBranchTest\":true}",
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.SET_EPSILON, decision.getAction());
        assertEquals(0.72, decision.getConfidence(), 0.0);
        assertEquals("medium", decision.getRiskLevel());
        assertEquals("less random wandering", decision.getExpectedEffect());
        assertEquals("restore epsilon", decision.getRollbackPlan());
        assertEquals("small", decision.getChangeMagnitude());
        assertTrue(decision.getRequiresBranchTest());
    }

    @Test
    void detectsNativeGeminiSupervisorCommands() {
        assertTrue(LlmSupervisorFactory.isNativeGeminiCommand("builtin:gemini"));
        assertTrue(LlmSupervisorFactory.isNativeGeminiCommand(
            "powershell -ExecutionPolicy Bypass -File scripts/llm_supervisor_gemini.ps1"));
        assertTrue(LlmSupervisorFactory.isNativeGeminiCommand(
            "python scripts/llm_supervisor_gemini.py"));
        assertFalse(LlmSupervisorFactory.isNativeGeminiCommand("python scripts/custom_provider.py"));
    }

    @Test
    void factoryUsesUiApiKeyForBuiltInGemini() throws Exception {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");
        config.setExternalCommand(LlmSupervisorFactory.BUILTIN_GEMINI_COMMAND);
        config.setApiKey("ui-key-123");

        LlmSupervisor supervisor = LlmSupervisorFactory.create(
            config,
            RLRewardConfig::createDefault,
            Map.of());

        assertTrue(supervisor instanceof GeminiLlmSupervisor);
        Field apiKeyField = GeminiLlmSupervisor.class.getDeclaredField("apiKey");
        apiKeyField.setAccessible(true);
        assertEquals("ui-key-123", apiKeyField.get(supervisor));
    }

    @Test
    void factoryKeepsOldGeminiScriptCommandsOnJavaPath() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");
        config.setExternalCommand("powershell -ExecutionPolicy Bypass -File scripts/llm_supervisor_gemini.ps1");

        LlmSupervisor supervisor = LlmSupervisorFactory.create(
            config,
            RLRewardConfig::createDefault,
            Map.of("GEMINI_API_KEY", "env-key-123"));

        assertTrue(supervisor instanceof GeminiLlmSupervisor);
    }

    @Test
    void factoryUsesBuiltInGeminiWhenEnabledCommandIsBlank() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");
        config.setExternalCommand("");

        LlmSupervisor supervisor = LlmSupervisorFactory.create(
            config,
            RLRewardConfig::createDefault,
            Map.of("GEMINI_API_KEY", "env-key-123"));

        assertTrue(LlmSupervisorFactory.isNativeGeminiCommand(""));
        assertTrue(supervisor instanceof GeminiLlmSupervisor);
    }

    @Test
    void builtInGeminiDefaultsToGemmaPrimaryAndGeminiFlashLiteFallback() throws Exception {
        GeminiLlmSupervisor supervisor = new GeminiLlmSupervisor("ui-key-123", RLRewardConfig::createDefault);

        Field modelField = GeminiLlmSupervisor.class.getDeclaredField("model");
        Field fallbackField = GeminiLlmSupervisor.class.getDeclaredField("fallbackModel");
        modelField.setAccessible(true);
        fallbackField.setAccessible(true);
        assertEquals("gemma-4-31b-it", modelField.get(supervisor));
        assertEquals("gemini-3.1-flash-lite", fallbackField.get(supervisor));
    }

    @Test
    void builtInGeminiUsesConfiguredPrimaryAndFallbackModels() throws Exception {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");
        config.setExternalCommand(LlmSupervisorFactory.BUILTIN_GEMINI_COMMAND);
        config.setPrimaryModel("gemini-custom");
        config.setFallbackModel("gemma-custom");

        LlmSupervisor supervisor = LlmSupervisorFactory.create(
            config,
            RLRewardConfig::createDefault,
            Map.of("GEMINI_API_KEY", "env-key-123"));

        Field modelField = GeminiLlmSupervisor.class.getDeclaredField("model");
        Field fallbackField = GeminiLlmSupervisor.class.getDeclaredField("fallbackModel");
        modelField.setAccessible(true);
        fallbackField.setAccessible(true);
        assertEquals("gemini-custom", modelField.get(supervisor));
        assertEquals("gemma-custom", fallbackField.get(supervisor));
    }

    @Test
    void callFrequencyAppliesPresetMultiplierToBaseEpisodes() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");
        config.setCallIntervalEpisodes(1000);

        config.setCallFrequency(LlmSupervisorConfig.CallFrequency.VERY_SOON);
        assertEquals(250, config.getEffectiveCallIntervalEpisodes());

        config.setCallFrequency(LlmSupervisorConfig.CallFrequency.SOON);
        assertEquals(500, config.getEffectiveCallIntervalEpisodes());

        config.setCallFrequency(LlmSupervisorConfig.CallFrequency.MEDIUM);
        assertEquals(1000, config.getEffectiveCallIntervalEpisodes());

        config.setCallFrequency(LlmSupervisorConfig.CallFrequency.LONG);
        assertEquals(2000, config.getEffectiveCallIntervalEpisodes());
    }

    private SupervisorObservation observation(boolean trainingRunning) {
        return new SupervisorObservation(
            "candidate",
            10,
            -1.0,
            0.1,
            0.2,
            1.0,
            -0.5,
            2.0,
            0.0,
            0,
            10,
            0.99,
            0.1,
            32,
            5,
            true,
            1,
            6,
            true,
            1,
            6,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            100,
            "MAX_STEPS",
            "step",
            "final",
            "2026-05-12T00:00:00Z",
            "2026-05-12T00:01:00Z",
            "2026-05-12T01:00:00Z",
            60_000L,
            3_540_000L,
            true,
            false,
            Map.of(),
            Map.of(),
            null,
            Map.of("trainingRunning", trainingRunning),
            Map.of("bestScoreDeltaSinceLastSupervisor", 0.5),
            null);
    }
}
