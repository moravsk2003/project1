package com.rustbuilder.ai.rl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.rustbuilder.ai.rl.supervisor.LlmSupervisorConfig;
import java.lang.reflect.Field;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class RLTrainingServiceTest {

    @Test
    void resetForNewTrainingRunStartsFreshExplorationAndRewardLineage() {
        RLTrainingService service = new RLTrainingService();
        try {
            RLRewardConfig rewardConfig = RLRewardConfig.createDefault();
            rewardConfig.basePlacementReward = 0.77;
            service.setRewardConfig(rewardConfig);
            service.setEpisodesTrained(250);
            service.setBestScore(42.0);
            service.setEpsilon(0.5);
            LlmSupervisorConfig supervisorConfig = LlmSupervisorConfig.enabledDefault("candidate");
            supervisorConfig.setCallFrequency(LlmSupervisorConfig.CallFrequency.LONG);
            service.setSupervisorConfig(supervisorConfig);

            service.resetForNewTrainingRun(true);

            assertEquals(0, service.getEpisodesTrained());
            assertEquals(1.0, service.getEpsilon(), 0.0);
            assertEquals(-Double.MAX_VALUE, service.getBestScore(), 0.0);
            assertEquals(0, service.getMemorySize());
            assertTrue(service.isUse2dCnn());
            assertEquals(RLRewardConfig.createDefault().basePlacementReward,
                service.getRewardConfig().basePlacementReward, 0.0);
            assertEquals(RLRewardConfig.createDefault().earlyStopPenaltyMult,
                service.getRewardConfig().earlyStopPenaltyMult, 0.0);
            assertEquals(LlmSupervisorConfig.CallFrequency.MEDIUM,
                service.getSupervisorConfig().getCallFrequency());
        } finally {
            service.getLlmOrchestrator().stop();
        }
    }

    @Test
    void duplicatePromoteBranchIsHandledAsNoOp() throws Exception {
        RLTrainingService service = new RLTrainingService();
        try {
            String candidate = "codex_model_candidate_123";
            setField(service, "latestBranchCandidateModelName", candidate);
            setField(service, "lastPromotedBranchModelName", candidate);
            setField(service, "latestBranchComparison", new LinkedHashMap<>(Map.of(
                "status", "finished",
                "savedCandidateModelName", candidate,
                "savedWinnerModelName", candidate,
                "savedWinnerBranch", "candidate",
                "promoteRecommended", true,
                "promotionApplied", false
            )));

            assertTrue(service.promoteLatestBranch());
            assertTrue(service.getSupervisorBranchStatus().contains("already promoted"));
            assertEquals("", getField(service, "latestBranchCandidateModelName"));
        } finally {
            service.getLlmOrchestrator().stop();
        }
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static Object getField(Object target, String name) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        return field.get(target);
    }
}
