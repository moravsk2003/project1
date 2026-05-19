package com.rustbuilder.ai.rl.supervisor.validation;


import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorObservation;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import com.rustbuilder.ai.rl.domain.RLRewardConfig;
import java.util.Map;
import org.junit.jupiter.api.Test;

class SupervisorDecisionValidatorTest {

    private final SupervisorDecisionValidator validator = new SupervisorDecisionValidator();

    @Test
    void clampsSupervisorEpsilonProposal() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");

        SupervisorDecision decision = validator.validate(
            SupervisorDecision.setEpsilon(5.0, "explore less")
                .withCallFrequency(LlmSupervisorConfig.CallFrequency.VERY_SOON),
            config,
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.SET_EPSILON, decision.getAction());
        assertEquals(1.0, decision.getProposedEpsilon(), 0.0);
        assertEquals(LlmSupervisorConfig.CallFrequency.VERY_SOON, decision.getProposedCallFrequency());
    }

    @Test
    void blocksEpsilonChangesWhenDisabledByPolicy() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");
        config.setAllowEpsilonChanges(false);

        SupervisorDecision decision = validator.validate(
            SupervisorDecision.setEpsilon(0.2, "try exploitation"),
            config,
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.KEEP_GOING, decision.getAction());
    }

    @Test
    void disabledSupervisorAlwaysKeepsGoing() {
        SupervisorDecision decision = validator.validate(
            SupervisorDecision.stopTraining("bad run"),
            LlmSupervisorConfig.disabled(),
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.KEEP_GOING, decision.getAction());
    }

    @Test
    void sanitizesRewardConfigProposal() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");
        RLRewardConfig current = RLRewardConfig.createDefault();
        current.basePlacementReward = 0.25;

        RLRewardConfig proposed = RLRewardConfig.createDefault();
        proposed.basePlacementReward = Double.NaN;
        proposed.penaltyCollision = -500.0;

        SupervisorDecision decision = validator.validate(
            SupervisorDecision.replaceRewardConfig(proposed, "tune penalties"),
            config,
            current);

        RLRewardConfig sanitized = decision.getProposedRewardConfig();
        assertEquals(SupervisorAction.REPLACE_REWARD_CONFIG, decision.getAction());
        assertEquals(0.25, sanitized.basePlacementReward, 0.0);
        assertEquals(-0.75, sanitized.penaltyCollision, 0.0);
    }

    @Test
    void experimentalCautionAllowsWiderRewardConfigChanges() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");
        config.setCautionLevel(LlmSupervisorConfig.CautionLevel.EXPERIMENTAL);

        RLRewardConfig proposed = RLRewardConfig.createDefault();
        proposed.raidBonusMultiplier = 500.0;

        SupervisorDecision decision = validator.validate(
            SupervisorDecision.replaceRewardConfig(proposed, "large experiment"),
            config,
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.REPLACE_REWARD_CONFIG, decision.getAction());
        assertEquals(100.0, decision.getProposedRewardConfig().raidBonusMultiplier, 0.0);
    }

    @Test
    void rejectsUnsafeAutopilotModelNames() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");

        SupervisorDecision decision = validator.validate(
            SupervisorDecision.startNewRun(false, RLRewardConfig.createDefault(), "../bad", "idle"),
            config,
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.KEEP_GOING, decision.getAction());
    }

    @Test
    void startRunDropsRewardConfigChanges() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");
        RLRewardConfig proposed = RLRewardConfig.createDefault();
        proposed.raidBonusMultiplier = 500.0;

        SupervisorDecision decision = validator.validate(
            SupervisorDecision.startNewRun(true, proposed, "auto_run_2d", "try 2d"),
            config,
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.START_NEW_RUN, decision.getAction());
        assertEquals("auto_run_2d", decision.getProposedModelName());
        assertNull(decision.getProposedRewardConfig());
    }

    @Test
    void clampsEpsilonDeltaByCautionWhenObservationExists() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");

        SupervisorDecision decision = validator.validate(
            SupervisorDecision.setEpsilon(0.0, "explore much less"),
            config,
            RLRewardConfig.createDefault(),
            observation(true));

        assertEquals(SupervisorAction.SET_EPSILON, decision.getAction());
        assertEquals(0.84, decision.getProposedEpsilon(), 1e-9);
    }

    @Test
    void sanitizesBranchJumpModelName() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");

        SupervisorDecision decision = validator.validate(
            SupervisorDecision.jumpToBranch("known_branch_1", "jump"),
            config,
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.JUMP_TO_BRANCH, decision.getAction());
        assertEquals("known_branch_1", decision.getProposedModelName());
    }

    @Test
    void sanitizesCurriculumObjectiveCriteria() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");

        SupervisorDecision decision = validator.validate(
            SupervisorDecision.setCurriculumObjective(
                "tc_one_layer",
                "Place and enclose TC",
                Map.of("tcPresentRateMin", 0.75, "unknownMetric", 1.0),
                Map.of("invalidActionRateMax", 0.45),
                "advance objective"),
            config,
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.SET_CURRICULUM_OBJECTIVE, decision.getAction());
        assertEquals("tc_one_layer", decision.getObjectiveId());
        assertEquals(1, decision.getObjectiveSuccessCriteria().size());
        assertEquals(0.75, decision.getObjectiveSuccessCriteria().get("tcPresentRateMin"), 0.0);
    }

    @Test
    void rejectsIdleOnlyActionsDuringTrainingObservation() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");

        SupervisorDecision decision = validator.validate(
            SupervisorDecision.startNewRun(false, RLRewardConfig.createDefault(), "new_run", "restart"),
            config,
            RLRewardConfig.createDefault(),
            observation(true));

        assertEquals(SupervisorAction.KEEP_GOING, decision.getAction());
    }

    @Test
    void allowsStartRunForIdleObservation() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");

        SupervisorDecision decision = validator.validate(
            SupervisorDecision.startNewRun(false, RLRewardConfig.createDefault(), "new_run", "start"),
            config,
            RLRewardConfig.createDefault(),
            observation(false));

        assertEquals(SupervisorAction.START_NEW_RUN, decision.getAction());
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
            Map.of(),
            null);
    }
}
