package com.rustbuilder.ai.rl;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class EpisodeRunnerTest {

    @Test
    void growthStreakRewardsFirstGrowthStep() {
        RLRewardConfig config = RLRewardConfig.createDefault();

        assertEquals(config.growthStreakBonus, EpisodeRunner.calculateGrowthStreakReward(1, config), 0.0);
    }

    @Test
    void growthStreakGrowsUntilCap() {
        RLRewardConfig config = RLRewardConfig.createDefault();

        assertEquals(config.growthStreakBonus * 2, EpisodeRunner.calculateGrowthStreakReward(2, config), 0.0);
        assertEquals(config.growthStreakBonus * 15, EpisodeRunner.calculateGrowthStreakReward(15, config), 0.0);
        assertEquals(config.growthStreakBonus * 15, EpisodeRunner.calculateGrowthStreakReward(100, config), 0.0);
    }

    @Test
    void growthStreakCanBeDisabledByConfig() {
        RLRewardConfig config = RLRewardConfig.createDefault();
        config.growthStreakBonus = 0.0;

        assertEquals(0.0, EpisodeRunner.calculateGrowthStreakReward(10, config), 0.0);
    }

    @Test
    void componentDeltaRewardsMainGrowthAndPenalizesNewComponents() {
        RLRewardConfig config = RLRewardConfig.createDefault();

        EpisodeRunner.ComponentDeltaReward reward = EpisodeRunner.calculateComponentDeltaReward(
            new EpisodeRunner.ComponentStats(1, 4),
            new EpisodeRunner.ComponentStats(2, 7),
            config
        );

        assertEquals(0.06, reward.mainComponentReward, 1e-9);
        assertEquals(0.025, reward.componentIncreasePenalty, 1e-9);
        assertEquals(0.035, reward.total(), 1e-9);
    }

    @Test
    void componentDeltaDoesNotRewardShrinkingMainComponentOrFewerComponents() {
        RLRewardConfig config = RLRewardConfig.createDefault();

        EpisodeRunner.ComponentDeltaReward reward = EpisodeRunner.calculateComponentDeltaReward(
            new EpisodeRunner.ComponentStats(3, 8),
            new EpisodeRunner.ComponentStats(1, 6),
            config
        );

        assertEquals(0.0, reward.mainComponentReward, 0.0);
        assertEquals(0.0, reward.componentIncreasePenalty, 0.0);
        assertEquals(0.0, reward.total(), 0.0);
    }

    @Test
    void tcProtectionDeltaRewardIsPositiveOnlyAndBudgetCapped() {
        RLRewardConfig config = RLRewardConfig.createDefault();

        assertEquals(0.4, EpisodeRunner.calculateTcProtectionDeltaReward(1.0, 3.0, config, 1.0), 1e-9);
        assertEquals(0.15, EpisodeRunner.calculateTcProtectionDeltaReward(1.0, 3.0, config, 0.15), 1e-9);
        assertEquals(0.0, EpisodeRunner.calculateTcProtectionDeltaReward(3.0, 1.0, config, 1.0), 0.0);
    }
}
