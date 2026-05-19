package com.rustbuilder.ai.rl.supervisor.validation;


import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorObservation;
import com.rustbuilder.ai.rl.supervisor.serialization.SupervisorJson;
import com.rustbuilder.ai.rl.domain.RLRewardConfig;
import com.rustbuilder.ai.rl.domain.reward.RewardFormulaSet;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Guards the training runtime from unsafe supervisor output.
 */
public class SupervisorDecisionValidator {
    private static final double MIN_EPSILON = 0.01;
    private static final double MAX_EPSILON = 1.0;
    private static final double MAX_ABS_REWARD_VALUE = 100.0;
    private static final int MAX_MODEL_NAME_LENGTH = 80;
    private static final int MAX_OBJECTIVE_ID_LENGTH = 80;
    private static final int MAX_OBJECTIVE_DESCRIPTION_LENGTH = 280;
    private static final Set<String> ALLOWED_OBJECTIVE_CRITERIA = Set.of(
        "medianScoreMin",
        "p25ScoreMin",
        "goodBaseRateMin",
        "validPlacementRateMin",
        "invalidActionRateMax",
        "medianBlocksMin",
        "p25BlocksMin",
        "tcPresentRateMin",
        "tcEnclosedRateMin",
        "lootRoomPresentRateMin",
        "connectedMainComponentRateMin",
        "medianRaidSulfurToTcMin",
        "p25RaidSulfurToTcMin");

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

            case SET_CURRICULUM_OBJECTIVE:
                String objectiveId = sanitizeObjectiveId(decision.getObjectiveId());
                if (objectiveId.isBlank()) {
                    return SupervisorDecision.keepGoing("SET_CURRICULUM_OBJECTIVE decision had no safe objectiveId.");
                }
                Map<String, Double> successCriteria = sanitizeObjectiveCriteria(decision.getObjectiveSuccessCriteria());
                if (successCriteria.isEmpty()) {
                    return SupervisorDecision.keepGoing("SET_CURRICULUM_OBJECTIVE decision had no supported successCriteria.");
                }
                return preserveMetadata(SupervisorDecision.setCurriculumObjective(
                    objectiveId,
                    sanitizeObjectiveDescription(decision.getObjectiveDescription(), objectiveId),
                    successCriteria,
                    sanitizeObjectiveCriteria(decision.getObjectiveFailureSignals()),
                    decision.getReason()), decision);

            case START_NEW_RUN:
            case RESTART_TRAINING:
                String modelName = sanitizeModelName(decision.getProposedModelName());
                if (modelName.isBlank()) {
                    return SupervisorDecision.keepGoing(decision.getAction() + " decision had no safe modelName.");
                }
                Double proposedStartEpsilon = decision.getProposedEpsilon();
                if (observation != null && decision.getAction() == SupervisorAction.START_NEW_RUN
                        && (proposedStartEpsilon == null || proposedStartEpsilon.isNaN() || proposedStartEpsilon.isInfinite())) {
                    return SupervisorDecision.keepGoing("START_NEW_RUN decision must include a safe epsilon.");
                }
                if (observation != null
                        && decision.getAction() == SupervisorAction.START_NEW_RUN
                        && availableModelNames(observation).contains(modelName)) {
                    return SupervisorDecision.keepGoing("START_NEW_RUN modelName already exists; use LOAD_EXISTING_MODEL for existing models.");
                }
                Double sanitizedStartEpsilon = proposedStartEpsilon != null
                    ? clamp(proposedStartEpsilon, MIN_EPSILON, MAX_EPSILON)
                    : null;
                SupervisorDecision startDecision = decision.getAction() == SupervisorAction.START_NEW_RUN
                    ? SupervisorDecision.startNewRun(decision.getProposedUse2dCnn(), null, modelName, sanitizedStartEpsilon, decision.getReason())
                    : SupervisorDecision.restartTraining(decision.getProposedUse2dCnn(), null, modelName, decision.getReason());
                return preserveMetadata(startDecision, decision);

            case LOAD_EXISTING_MODEL:
                String loadModelName = sanitizeModelName(decision.getProposedModelName());
                if (loadModelName.isBlank()) {
                    return SupervisorDecision.keepGoing("LOAD_EXISTING_MODEL decision had no safe modelName.");
                }
                Double loadEpsilon = decision.getProposedEpsilon();
                if (observation != null && (loadEpsilon == null || loadEpsilon.isNaN() || loadEpsilon.isInfinite())) {
                    return SupervisorDecision.keepGoing("LOAD_EXISTING_MODEL decision must include a safe epsilon.");
                }
                if (observation != null && !isCompatibleAvailableModel(observation, loadModelName)) {
                    return SupervisorDecision.keepGoing("LOAD_EXISTING_MODEL can only target a compatible model from availableModels.");
                }
                Double sanitizedLoadEpsilon = loadEpsilon != null
                    ? clamp(loadEpsilon, MIN_EPSILON, MAX_EPSILON)
                    : null;
                return preserveMetadata(
                    SupervisorDecision.loadExistingModel(loadModelName, sanitizedLoadEpsilon, decision.getReason()),
                    decision);

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

    private static String sanitizeObjectiveId(String value) {
        if (value == null) {
            return "";
        }
        String trimmed = value.trim();
        if (trimmed.isBlank() || trimmed.length() > MAX_OBJECTIVE_ID_LENGTH) {
            return "";
        }
        return trimmed.matches("[A-Za-z0-9._-]+") ? trimmed : "";
    }

    private static String sanitizeObjectiveDescription(String value, String fallback) {
        String text = value != null ? value.trim() : "";
        if (text.isBlank()) {
            text = fallback != null ? fallback : "";
        }
        if (text.length() > MAX_OBJECTIVE_DESCRIPTION_LENGTH) {
            text = text.substring(0, MAX_OBJECTIVE_DESCRIPTION_LENGTH);
        }
        return text;
    }

    private static Map<String, Double> sanitizeObjectiveCriteria(Map<String, Double> criteria) {
        if (criteria == null || criteria.isEmpty()) {
            return Map.of();
        }
        Map<String, Double> sanitized = new LinkedHashMap<>();
        for (Map.Entry<String, Double> entry : criteria.entrySet()) {
            String key = entry.getKey();
            Double value = entry.getValue();
            if (!ALLOWED_OBJECTIVE_CRITERIA.contains(key)
                    || value == null
                    || value.isNaN()
                    || value.isInfinite()) {
                continue;
            }
            sanitized.put(key, clamp(value, 0.0, 10_000.0));
        }
        return sanitized;
    }

    private static Integer sanitizeEpoch(Integer epoch) {
        if (epoch == null) {
            return null;
        }
        return Math.max(0, epoch);
    }

    private static List<String> availableModelNames(SupervisorObservation observation) {
        if (observation == null || observation.trendMetrics == null) {
            return List.of();
        }
        Object modelsObject = observation.trendMetrics.get("availableModels");
        if (!(modelsObject instanceof List<?> models)) {
            return List.of();
        }
        java.util.ArrayList<String> names = new java.util.ArrayList<>();
        for (Object item : models) {
            if (item instanceof Map<?, ?> model) {
                Object name = model.get("name");
                if (name instanceof String text && !text.isBlank()) {
                    names.add(text);
                }
            }
        }
        return names;
    }

    private static boolean isCompatibleAvailableModel(SupervisorObservation observation, String modelName) {
        if (observation == null || observation.trendMetrics == null || modelName == null) {
            return false;
        }
        Object modelsObject = observation.trendMetrics.get("availableModels");
        if (!(modelsObject instanceof List<?> models)) {
            return false;
        }
        for (Object item : models) {
            if (!(item instanceof Map<?, ?> model)) {
                continue;
            }
            Object name = model.get("name");
            if (!(name instanceof String text) || !modelName.equals(text)) {
                continue;
            }
            Object compatible = model.get("compatible");
            return Boolean.TRUE.equals(compatible);
        }
        return false;
    }
}
