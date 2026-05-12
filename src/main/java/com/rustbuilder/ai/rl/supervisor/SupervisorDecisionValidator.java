package com.rustbuilder.ai.rl.supervisor;

import com.rustbuilder.ai.rl.RLRewardConfig;
import com.rustbuilder.ai.rl.reward.RewardFormulaSet;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

/**
 * Guards the training runtime from unsafe supervisor output.
 */
public class SupervisorDecisionValidator {
    private static final double MIN_EPSILON = 0.01;
    private static final double MAX_EPSILON = 1.0;
    private static final double MAX_ABS_REWARD_VALUE = 100.0;

    public SupervisorDecision validate(SupervisorDecision decision,
                                       LlmSupervisorConfig config,
                                       RLRewardConfig currentRewardConfig) {
        if (decision == null) {
            return SupervisorDecision.keepGoing("Supervisor returned no decision.");
        }
        if (config == null || !config.isEnabled()) {
            return SupervisorDecision.keepGoing("Supervisor is disabled.");
        }

        switch (decision.getAction()) {
            case SET_EPSILON:
                if (!config.isAllowEpsilonChanges()) {
                    return SupervisorDecision.keepGoing("Epsilon changes are not allowed.");
                }
                Double epsilon = decision.getProposedEpsilon();
                if (epsilon == null || epsilon.isNaN() || epsilon.isInfinite()) {
                    return SupervisorDecision.keepGoing("Invalid epsilon proposal.");
                }
                return SupervisorDecision.setEpsilon(clamp(epsilon, MIN_EPSILON, MAX_EPSILON), decision.getReason());

            case REPLACE_REWARD_CONFIG:
                if (!config.isAllowRewardConfigChanges()) {
                    return SupervisorDecision.keepGoing("Reward config changes are not allowed.");
                }
                RLRewardConfig rewardConfig = decision.getProposedRewardConfig();
                if (rewardConfig == null) {
                    return SupervisorDecision.keepGoing("Invalid reward config proposal.");
                }
                return SupervisorDecision.replaceRewardConfig(
                    sanitizeRewardConfig(rewardConfig, currentRewardConfig),
                    decision.getReason());

            case STOP_TRAINING:
            case REQUEST_PROMOTION_CHECK:
            case KEEP_GOING:
            default:
                return decision;
        }
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private RLRewardConfig sanitizeRewardConfig(RLRewardConfig proposed, RLRewardConfig currentRewardConfig) {
        RLRewardConfig sanitized = currentRewardConfig != null
            ? currentRewardConfig.clone()
            : RLRewardConfig.createDefault();

        for (Field field : RLRewardConfig.class.getFields()) {
            if (field.getType() != double.class || Modifier.isStatic(field.getModifiers())) {
                continue;
            }
            try {
                double proposedValue = field.getDouble(proposed);
                if (Double.isNaN(proposedValue) || Double.isInfinite(proposedValue)) {
                    continue;
                }
                field.setDouble(sanitized, clamp(proposedValue, -MAX_ABS_REWARD_VALUE, MAX_ABS_REWARD_VALUE));
            } catch (IllegalAccessException ignored) {
                // Public config fields should be accessible; keep current value if reflection fails.
            }
        }
        sanitized.rewardFormulaSet = proposed.rewardFormulaSet != null
            ? proposed.rewardFormulaSet.clone()
            : new RewardFormulaSet();

        return sanitized;
    }
}
