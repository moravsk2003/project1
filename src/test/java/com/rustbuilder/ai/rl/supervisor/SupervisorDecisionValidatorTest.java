package com.rustbuilder.ai.rl.supervisor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.rustbuilder.ai.rl.RLRewardConfig;
import org.junit.jupiter.api.Test;

class SupervisorDecisionValidatorTest {

    private final SupervisorDecisionValidator validator = new SupervisorDecisionValidator();

    @Test
    void clampsSupervisorEpsilonProposal() {
        LlmSupervisorConfig config = LlmSupervisorConfig.enabledDefault("candidate");

        SupervisorDecision decision = validator.validate(
            SupervisorDecision.setEpsilon(5.0, "explore less"),
            config,
            RLRewardConfig.createDefault());

        assertEquals(SupervisorAction.SET_EPSILON, decision.getAction());
        assertEquals(1.0, decision.getProposedEpsilon(), 0.0);
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
        assertEquals(-100.0, sanitized.penaltyCollision, 0.0);
    }
}
