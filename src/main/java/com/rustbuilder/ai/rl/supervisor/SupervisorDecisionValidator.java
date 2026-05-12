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
    private static final int MAX_MODEL_NAME_LENGTH = 80;

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

            case START_NEW_RUN:
            case RESTART_TRAINING:
                String modelName = sanitizeModelName(decision.getProposedModelName());
                if (modelName.isBlank()) {
                    return SupervisorDecision.keepGoing(decision.getAction() + " decision had no safe modelName.");
                }
                RLRewardConfig startConfig = config.isAllowRewardConfigChanges()
                    ? sanitizeRewardConfig(decision.getProposedRewardConfig(), currentRewardConfig)
                    : (currentRewardConfig != null ? currentRewardConfig.clone() : RLRewardConfig.createDefault());
                return decision.getAction() == SupervisorAction.START_NEW_RUN
                    ? SupervisorDecision.startNewRun(decision.getProposedUse2dCnn(), startConfig, modelName, decision.getReason())
                    : SupervisorDecision.restartTraining(decision.getProposedUse2dCnn(), startConfig, modelName, decision.getReason());

            case REQUEST_HISTORICAL_REPORT:
                String reportModelName = sanitizeModelName(decision.getReportModelName());
                if (reportModelName.isBlank()) {
                    return SupervisorDecision.keepGoing("REQUEST_HISTORICAL_REPORT decision had no safe reportModelName.");
                }
                return SupervisorDecision.requestHistoricalReport(
                    reportModelName,
                    sanitizeEpoch(decision.getReportStartEpoch()),
                    sanitizeEpoch(decision.getReportEndEpoch()),
                    decision.getReason());

            case PROMOTE_BRANCH:
                return SupervisorDecision.promoteBranch(decision.getReason());

            case JUMP_TO_BRANCH:
                String branchModelName = sanitizeModelName(decision.getProposedModelName());
                if (branchModelName.isBlank()) {
                    return SupervisorDecision.keepGoing("JUMP_TO_BRANCH decision had no safe modelName.");
                }
                return SupervisorDecision.jumpToBranch(branchModelName, decision.getReason());

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
        if (proposed == null) {
            return sanitized;
        }

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

    private static String sanitizeModelName(String value) {
        if (value == null) {
            return "";
        }
        String trimmed = value.trim();
        if (trimmed.isBlank() || trimmed.length() > MAX_MODEL_NAME_LENGTH) {
            return "";
        }
        return trimmed.matches("[A-Za-z0-9._-]+") ? trimmed : "";
    }

    private static Integer sanitizeEpoch(Integer epoch) {
        if (epoch == null) {
            return null;
        }
        return Math.max(0, epoch);
    }
}
