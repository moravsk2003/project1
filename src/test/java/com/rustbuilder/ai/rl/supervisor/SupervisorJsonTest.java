package com.rustbuilder.ai.rl.supervisor;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
}
