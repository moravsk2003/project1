package com.rustbuilder.ai.rl.supervisor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.rustbuilder.ai.rl.RLRewardConfig;
import com.rustbuilder.ai.rl.reward.RewardFormulaScope;
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

        assertTrue(json.contains("\"allowedActions\":[\"KEEP_GOING\",\"SET_EPSILON\",\"REPLACE_REWARD_CONFIG\",\"STOP_TRAINING\",\"REQUEST_PROMOTION_CHECK\",\"PROMOTE_BRANCH\",\"JUMP_TO_BRANCH\"]"));
        assertTrue(json.contains("\"rewardFormulaContract\""));
        assertTrue(json.contains("\"socket_connections\""));
        assertTrue(json.contains("\"final_score\""));
    }

    @Test
    void idleObservationContractAllowsStartingRunsAndReports() {
        String json = SupervisorJson.observationToJson(observation(false));

        assertTrue(json.contains("\"allowedActions\":[\"KEEP_GOING\",\"START_NEW_RUN\",\"REQUEST_HISTORICAL_REPORT\",\"PROMOTE_BRANCH\",\"JUMP_TO_BRANCH\"]"));
    }

    @Test
    void parsesIdleAutopilotDecisions() {
        SupervisorDecision start = SupervisorJson.decisionFromJson(
            "{\"action\":\"START_NEW_RUN\",\"modelName\":\"auto_run_1\",\"use2dCnn\":true,\"reason\":\"idle\"}",
            RLRewardConfig.createDefault());
        SupervisorDecision report = SupervisorJson.decisionFromJson(
            "{\"action\":\"REQUEST_HISTORICAL_REPORT\",\"reportModelName\":\"model_a\",\"reportStartEpoch\":2,\"reportEndEpoch\":5}",
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.START_NEW_RUN, start.getAction());
        assertEquals("auto_run_1", start.getProposedModelName());
        assertEquals(Boolean.TRUE, start.getProposedUse2dCnn());
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
