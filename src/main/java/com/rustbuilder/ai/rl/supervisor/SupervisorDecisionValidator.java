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
        return validate(decision, config, currentRewardConfig, null);
    }

    public SupervisorDecision validate(SupervisorDecision decision,
                                       LlmSupervisorConfig config,
                                       RLRewardConfig currentRewardConfig,
                                       SupervisorObservation observation) {
        if (decision == null) {
            return SupervisorDecision.keepGoing("Supervisor returned no decision.");
        }
        if (config == null || !config.isEnabled()) {
            return SupervisorDecision.keepGoing("Supervisor is disabled.");
        }
        if (observation != null && !SupervisorJson.allowedActionsFor(observation).contains(decision.getAction().name())) {
            return SupervisorDecision.keepGoing(
                "Ignoring " + decision.getAction() + " because it is not allowed for the current training state.");
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
                double sanitizedEpsilon = clamp(epsilon, MIN_EPSILON, MAX_EPSILON);
                if (observation != null) {
                    double deltaLimit = epsilonDeltaLimit(config);
                    sanitizedEpsilon = clamp(sanitizedEpsilon,
                        Math.max(MIN_EPSILON, observation.epsilon - deltaLimit),
                        Math.min(MAX_EPSILON, observation.epsilon + deltaLimit));
                }
                return preserveMetadata(
                    SupervisorDecision.setEpsilon(sanitizedEpsilon, decision.getReason()),
                    decision);

            case REPLACE_REWARD_CONFIG:
                if (!config.isAllowRewardConfigChanges()) {
                    return SupervisorDecision.keepGoing("Reward config changes are not allowed.");
                }
                RLRewardConfig rewardConfig = decision.getProposedRewardConfig();
                if (rewardConfig == null) {
                    return SupervisorDecision.keepGoing("Invalid reward config proposal.");
                }
                return preserveMetadata(SupervisorDecision.replaceRewardConfig(
                    sanitizeRewardConfig(rewardConfig, currentRewardConfig, config),
                    decision.getReason()), decision);

            case START_NEW_RUN:
            case RESTART_TRAINING:
                String modelName = sanitizeModelName(decision.getProposedModelName());
                if (modelName.isBlank()) {
                    return SupervisorDecision.keepGoing(decision.getAction() + " decision had no safe modelName.");
                }
                SupervisorDecision startDecision = decision.getAction() == SupervisorAction.START_NEW_RUN
                    ? SupervisorDecision.startNewRun(decision.getProposedUse2dCnn(), null, modelName, decision.getReason())
                    : SupervisorDecision.restartTraining(decision.getProposedUse2dCnn(), null, modelName, decision.getReason());
                return preserveMetadata(startDecision, decision);

            case REQUEST_HISTORICAL_REPORT:
                String reportModelName = sanitizeModelName(decision.getReportModelName());
                if (reportModelName.isBlank()) {
                    return SupervisorDecision.keepGoing("REQUEST_HISTORICAL_REPORT decision had no safe reportModelName.");
                }
                return preserveMetadata(SupervisorDecision.requestHistoricalReport(
                    reportModelName,
                    sanitizeEpoch(decision.getReportStartEpoch()),
                    sanitizeEpoch(decision.getReportEndEpoch()),
                    decision.getReason()), decision);

            case PROMOTE_BRANCH:
                return preserveMetadata(SupervisorDecision.promoteBranch(decision.getReason()), decision);

            case JUMP_TO_BRANCH:
                String branchModelName = sanitizeModelName(decision.getProposedModelName());
                if (branchModelName.isBlank()) {
                    return SupervisorDecision.keepGoing("JUMP_TO_BRANCH decision had no safe modelName.");
                }
                return preserveMetadata(SupervisorDecision.jumpToBranch(branchModelName, decision.getReason()), decision);

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

    private static SupervisorDecision preserveMetadata(SupervisorDecision sanitized,
                                                       SupervisorDecision source) {
        SupervisorDecision result = source.getProposedCallFrequency() != null
            ? sanitized.withCallFrequency(source.getProposedCallFrequency())
            : sanitized;
        return result.withAnalysisMetadata(
            source.getConfidence(),
            source.getRiskLevel(),
            source.getExpectedEffect(),
            source.getRollbackPlan(),
            source.getChangeMagnitude(),
            source.getRequiresBranchTest());
    }

    private RLRewardConfig sanitizeRewardConfig(RLRewardConfig proposed,
                                                RLRewardConfig currentRewardConfig,
                                                LlmSupervisorConfig config) {
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
                double currentValue = field.getDouble(sanitized);
                double deltaLimit = rewardDeltaLimit(config, currentValue);
                double deltaClamped = clamp(proposedValue,
                    currentValue - deltaLimit,
                    currentValue + deltaLimit);
                field.setDouble(sanitized, clamp(deltaClamped, -MAX_ABS_REWARD_VALUE, MAX_ABS_REWARD_VALUE));
            } catch (IllegalAccessException ignored) {
                // Public config fields should be accessible; keep current value if reflection fails.
            }
        }
        sanitized.rewardFormulaSet = proposed.rewardFormulaSet != null
            ? proposed.rewardFormulaSet.clone()
            : new RewardFormulaSet();

        return sanitized;
    }

    private static double epsilonDeltaLimit(LlmSupervisorConfig config) {
        LlmSupervisorConfig.CautionLevel level = config != null
            ? config.getCautionLevel()
            : LlmSupervisorConfig.CautionLevel.BALANCED;
        return switch (level) {
            case CONSERVATIVE -> 0.05;
            case BALANCED -> 0.15;
            case BOLD -> 0.35;
            case EXPERIMENTAL -> 1.0;
        };
    }

    private static double rewardDeltaLimit(LlmSupervisorConfig config, double currentValue) {
        LlmSupervisorConfig.CautionLevel level = config != null
            ? config.getCautionLevel()
            : LlmSupervisorConfig.CautionLevel.BALANCED;
        double abs = Math.abs(currentValue);
        return switch (level) {
            case CONSERVATIVE -> Math.max(0.10, abs * 0.20);
            case BALANCED -> Math.max(0.50, abs * 0.50);
            case BOLD -> Math.max(2.00, abs * 1.50);
            case EXPERIMENTAL -> Math.max(10.00, abs * 5.00);
        };
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
