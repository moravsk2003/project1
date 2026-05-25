package com.rustbuilder.ai.rl.supervisor.serialization;


import com.rustbuilder.ai.rl.domain.log.StopReason;
import com.rustbuilder.ai.rl.domain.reward.RewardFormulaSet;
import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorAction;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorActionDirection;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorObservation;
import com.rustbuilder.ai.rl.domain.RLRewardConfig;
import com.rustbuilder.ai.rl.domain.reward.RewardFormulaScope;
import com.rustbuilder.ai.rl.domain.reward.RewardFormulaTerm;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class SupervisorJson {
    private static final List<String> TRAINING_ACTIONS = List.of(
        SupervisorAction.KEEP_GOING.name(),
        SupervisorAction.SET_EPSILON.name(),
        SupervisorAction.REPLACE_REWARD_CONFIG.name(),
        SupervisorAction.SET_CURRICULUM_OBJECTIVE.name(),
        SupervisorAction.STOP_TRAINING.name(),
        SupervisorAction.REQUEST_PROMOTION_CHECK.name(),
        SupervisorAction.PROMOTE_BRANCH.name(),
        SupervisorAction.JUMP_TO_BRANCH.name());
    private static final List<String> IDLE_ACTIONS = List.of(
        SupervisorAction.KEEP_GOING.name(),
        SupervisorAction.SET_CURRICULUM_OBJECTIVE.name(),
        SupervisorAction.START_NEW_RUN.name(),
        SupervisorAction.LOAD_EXISTING_MODEL.name(),
        SupervisorAction.REQUEST_HISTORICAL_REPORT.name(),
        SupervisorAction.PROMOTE_BRANCH.name(),
        SupervisorAction.JUMP_TO_BRANCH.name());
    private static final List<String> STEP_REWARD_FORMULA_VARIABLES = List.of(
        "inserted",
        "survived",
        "invalid",
        "block_count",
        "socket_connections",
        "stability",
        "is_foundation",
        "is_tc_action",
        "is_workbench_action",
        "is_loot_room_action",
        "error_no_support",
        "error_collision",
        "error_bad_socket");
    private static final List<String> FINAL_REWARD_FORMULA_VARIABLES = List.of(
        "final_score",
        "raw_score_reward",
        "logistics_score",
        "cost_score",
        "raid_score",
        "working_area_score",
        "safe_zone_score",
        "sulfur_to_tc",
        "component_count",
        "main_component_blocks",
        "blocks_placed",
        "tc_present",
        "tc_enclosed",
        "invalid_rate",
        "early_stop_penalty");
    private static final List<String> REWARD_FORMULA_FUNCTIONS = List.of(
        "abs(x)",
        "sqrt(x)",
        "min(a,b)",
        "max(a,b)",
        "clamp(x,min,max)");

    private SupervisorJson() {
    }

    public static String observationToJson(SupervisorObservation obs) {
        return SimpleJson.stringify(observationToMap(obs));
    }

    public static Map<String, Object> observationToMap(SupervisorObservation obs) {
        return observationToMap(obs, null);
    }

    public static Map<String, Object> observationToMap(SupervisorObservation obs,
                                                       List<String> allowedActionsOverride) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("branchId", obs.branchId);
        map.put("totalEpisodesTrained", obs.totalEpisodesTrained);
        map.put("bestScore", obs.bestScore);
        map.put("avgEvalScore", obs.avgEvalScore);
        map.put("currentEpisodeEvalScore", obs.currentEpisodeEvalScore);
        map.put("currentEpisodeStepReward", obs.currentEpisodeStepReward);
        map.put("currentEpisodeFinalReward", obs.currentEpisodeFinalReward);
        map.put("bestTotalReward", obs.bestTotalReward);
        map.put("invalidActionRate", obs.invalidActionRate);
        map.put("lastEpisodeInvalidActions", obs.lastEpisodeInvalidActions);
        map.put("lastEpisodeTotalActions", obs.lastEpisodeTotalActions);
        map.put("epsilon", obs.epsilon);
        map.put("lastTrainLoss", obs.lastTrainLoss);
        map.put("memorySize", obs.memorySize);
        map.put("bestBaseBlocks", obs.bestBaseBlocks);
        map.put("bestBaseHasTC", obs.bestBaseHasTC);
        map.put("bestBaseDoors", obs.bestBaseDoors);
        map.put("episodeBlocksPlaced", obs.episodeBlocksPlaced);
        map.put("episodeHasTC", obs.episodeHasTC);
        map.put("componentCount", obs.componentCount);
        map.put("mainComponentBlocks", obs.mainComponentBlocks);
        map.put("evalLogisticsScore", obs.evalLogisticsScore);
        map.put("evalCostScore", obs.evalCostScore);
        map.put("evalRaidScore", obs.evalRaidScore);
        map.put("evalWorkingAreaScore", obs.evalWorkingAreaScore);
        map.put("evalSafeZoneScore", obs.evalSafeZoneScore);
        map.put("raidSulfurToTC", obs.raidSulfurToTC);
        map.put("stopReason", obs.stopReason);
        map.put("stepRewardBreakdown", obs.stepRewardBreakdown);
        map.put("finalRewardBreakdown", obs.finalRewardBreakdown);
        map.put("trainingStartTime", obs.trainingStartTimeIso);
        map.put("currentTime", obs.currentTimeIso);
        map.put("trainingDeadline", obs.trainingDeadlineIso);
        map.put("trainingElapsedSeconds", obs.trainingElapsedMs / 1000.0);
        map.put("trainingRemainingSeconds", obs.trainingRemainingMs >= 0 ? obs.trainingRemainingMs / 1000.0 : null);
        map.put("trainingTimeLimitEnabled", obs.trainingTimeLimitEnabled);
        map.put("trainingTimeLimitReached", obs.trainingTimeLimitReached);
        map.put("invalidActionReasons", obs.invalidActionReasons);
        map.put("actionTypeCounts", obs.actionTypeCounts);
        map.put("currentRewardConfig", obs.currentRewardConfig);
        map.put("trainingContext", obs.trainingContext);
        map.put("trendMetrics", obs.trendMetrics);
        map.put("rewardFormulaContract", rewardFormulaContract());
        List<String> allowedActions = allowedActionsOverride != null
            ? List.copyOf(allowedActionsOverride)
            : allowedActions(obs);
        map.put("allowedActions", allowedActions);
        map.put("actionDirections", actionDirectionMaps(allowedActions));
        
        if (obs.historicalReport != null && !obs.historicalReport.isBlank()) {
            map.put("historicalReport", obs.historicalReport);
        }
        map.put("responseContract",
            "Built-in LLM mode uses two calls: first choose one actionDirections.direction that is available now, then return one concrete action from allowedActions. The final action must be one of allowedActions. Always include reason and callFrequency. callFrequency must be VERY_SOON, SOON, MEDIUM, or LONG and selects the next review cadence. For SET_EPSILON include epsilon. For REPLACE_REWARD_CONFIG include only changed rewardConfig fields and/or rewardTerms. For SET_CURRICULUM_OBJECTIVE include objectiveId, objectiveDescription, and numeric successCriteria/failureSignals. For START_NEW_RUN include a new modelName and epsilon; rewardConfig/rewardTerms are ignored because new runs reset reward state to defaults. For LOAD_EXISTING_MODEL include a compatible modelName from trendMetrics.availableModels and epsilon. For JUMP_TO_BRANCH include modelName. For important changes include confidence, riskLevel, expectedEffect, rollbackPlan, changeMagnitude, and requiresBranchTest.");
        return map;
    }

    public static List<String> allowedActionsFor(SupervisorObservation obs) {
        Object running = obs.trainingContext != null ? obs.trainingContext.get("trainingRunning") : null;
        return Boolean.FALSE.equals(running) ? IDLE_ACTIONS : TRAINING_ACTIONS;
    }

    private static List<String> allowedActions(SupervisorObservation obs) {
        return allowedActionsFor(obs);
    }

    public static List<Map<String, Object>> actionDirectionMaps(List<String> allowedActions) {
        List<Map<String, Object>> result = new ArrayList<>();
        for (SupervisorActionDirection direction : SupervisorActionDirection.allowedDirectionsFor(allowedActions)) {
            result.add(direction.toMap(allowedActions));
        }
        return result;
    }

    private static Map<String, Object> rewardFormulaContract() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("stepVariables", STEP_REWARD_FORMULA_VARIABLES);
        map.put("finalVariables", FINAL_REWARD_FORMULA_VARIABLES);
        map.put("functions", REWARD_FORMULA_FUNCTIONS);
        map.put("unknownVariablesEvaluateToZero", true);
        return map;
    }

    @SuppressWarnings("unchecked")
    public static SupervisorDecision decisionFromJson(String json, RLRewardConfig currentRewardConfig) {
        Object parsed = SimpleJson.parse(json);
        if (!(parsed instanceof Map<?, ?> parsedMap)) {
            return SupervisorDecision.keepGoing("Supervisor did not return a JSON object.");
        }

        Map<String, Object> map = (Map<String, Object>) parsedMap;
        SupervisorAction action = parseAction(stringValue(map.get("action")));
        String reason = stringValue(map.get("reason"));
        LlmSupervisorConfig.CallFrequency callFrequency = parseCallFrequency(stringValue(map.get("callFrequency")));

        if (action == SupervisorAction.SET_EPSILON) {
            Double epsilon = doubleValue(map.get("epsilon"));
            SupervisorDecision decision = epsilon != null
                ? SupervisorDecision.setEpsilon(epsilon, reason)
                : SupervisorDecision.keepGoing("SET_EPSILON decision had no epsilon.");
            return withMetadata(withCallFrequency(decision, callFrequency), map);
        }

        if (action == SupervisorAction.REPLACE_REWARD_CONFIG) {
            RLRewardConfig rewardConfig = currentRewardConfig != null
                ? currentRewardConfig.clone()
                : RLRewardConfig.createDefault();
            applyRewardConfigPatch(rewardConfig, map.get("rewardConfig"));
            applyRewardTerms(rewardConfig, map.get("rewardTerms"));
            return withMetadata(withCallFrequency(SupervisorDecision.replaceRewardConfig(rewardConfig, reason), callFrequency), map);
        }

        if (action == SupervisorAction.SET_CURRICULUM_OBJECTIVE) {
            SupervisorDecision decision = SupervisorDecision.setCurriculumObjective(
                stringValue(map.get("objectiveId")),
                stringValue(map.get("objectiveDescription")),
                doubleMapValue(map.get("successCriteria")),
                doubleMapValue(map.get("failureSignals")),
                reason);
            return withMetadata(withCallFrequency(decision, callFrequency), map);
        }

        if (action == SupervisorAction.STOP_TRAINING) {
            return withMetadata(withCallFrequency(SupervisorDecision.stopTraining(reason), callFrequency), map);
        }
        if (action == SupervisorAction.REQUEST_PROMOTION_CHECK) {
            return withMetadata(withCallFrequency(SupervisorDecision.requestPromotionCheck(reason), callFrequency), map);
        }
        
        if (action == SupervisorAction.START_NEW_RUN || action == SupervisorAction.RESTART_TRAINING) {
            Boolean use2dCnn = booleanValue(map.get("use2dCnn"));
            String modelName = stringValue(map.get("modelName"));
            Double epsilon = doubleValue(map.get("epsilon"));
            
            if (action == SupervisorAction.START_NEW_RUN) {
                return withMetadata(withCallFrequency(SupervisorDecision.startNewRun(use2dCnn, null, modelName, epsilon, reason), callFrequency), map);
            } else {
                return withMetadata(withCallFrequency(SupervisorDecision.restartTraining(use2dCnn, null, modelName, reason), callFrequency), map);
            }
        }

        if (action == SupervisorAction.LOAD_EXISTING_MODEL) {
            String modelName = stringValue(map.get("modelName"));
            Double epsilon = doubleValue(map.get("epsilon"));
            return withMetadata(withCallFrequency(SupervisorDecision.loadExistingModel(modelName, epsilon, reason), callFrequency), map);
        }
        
        if (action == SupervisorAction.REQUEST_HISTORICAL_REPORT) {
            String reportModelName = stringValue(map.get("reportModelName"));
            Integer startEpoch = integerValue(map.get("reportStartEpoch"));
            Integer endEpoch = integerValue(map.get("reportEndEpoch"));
            return withMetadata(withCallFrequency(SupervisorDecision.requestHistoricalReport(reportModelName, startEpoch, endEpoch, reason), callFrequency), map);
        }

        if (action == SupervisorAction.PROMOTE_BRANCH) {
            return withMetadata(withCallFrequency(SupervisorDecision.promoteBranch(reason), callFrequency), map);
        }

        if (action == SupervisorAction.JUMP_TO_BRANCH) {
            return withMetadata(withCallFrequency(SupervisorDecision.jumpToBranch(stringValue(map.get("modelName")), reason), callFrequency), map);
        }
        
        return withMetadata(withCallFrequency(SupervisorDecision.keepGoing(reason), callFrequency), map);
    }

    private static SupervisorDecision withCallFrequency(SupervisorDecision decision,
                                                        LlmSupervisorConfig.CallFrequency callFrequency) {
        return callFrequency != null ? decision.withCallFrequency(callFrequency) : decision;
    }

    private static SupervisorDecision withMetadata(SupervisorDecision decision, Map<String, Object> map) {
        if (decision == null || map == null) {
            return decision;
        }
        return decision.withAnalysisMetadata(
            doubleValue(map.get("confidence")),
            stringValue(map.get("riskLevel")),
            stringValue(map.get("expectedEffect")),
            stringValue(map.get("rollbackPlan")),
            stringValue(map.get("changeMagnitude")),
            booleanValue(map.get("requiresBranchTest")));
    }

    private static Integer integerValue(Object val) {
        if (val instanceof Number n) {
            return n.intValue();
        }
        if (val instanceof String s) {
            try {
                return Integer.parseInt(s.trim());
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    private static SupervisorAction parseAction(String action) {
        if (action == null || action.isBlank()) {
            return SupervisorAction.KEEP_GOING;
        }
        try {
            return SupervisorAction.valueOf(action.trim().toUpperCase());
        } catch (IllegalArgumentException e) {
            return SupervisorAction.KEEP_GOING;
        }
    }

    private static LlmSupervisorConfig.CallFrequency parseCallFrequency(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        String normalized = value.trim().toUpperCase().replace('-', '_').replace(' ', '_');
        try {
            return LlmSupervisorConfig.CallFrequency.valueOf(normalized);
        } catch (IllegalArgumentException ignored) {
            return null;
        }
    }

    @SuppressWarnings("unchecked")
    private static void applyRewardConfigPatch(RLRewardConfig target, Object patchObject) {
        if (!(patchObject instanceof Map<?, ?> patch)) {
            return;
        }
        Map<String, Object> values = (Map<String, Object>) patch;
        for (Field field : RLRewardConfig.class.getFields()) {
            if (field.getType() != double.class || !values.containsKey(field.getName())) {
                continue;
            }
            Double value = doubleValue(values.get(field.getName()));
            if (value == null) {
                continue;
            }
            try {
                field.setDouble(target, value);
            } catch (IllegalAccessException ignored) {
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static void applyRewardTerms(RLRewardConfig target, Object rewardTermsObject) {
        if (!(rewardTermsObject instanceof List<?> terms)) {
            return;
        }
        for (Object item : terms) {
            if (!(item instanceof Map<?, ?> termMap)) {
                continue;
            }
            Map<String, Object> term = (Map<String, Object>) termMap;
            String name = stringValue(term.get("name"));
            String scopeRaw = stringValue(term.get("scope"));
            String expression = stringValue(term.get("expression"));
            RewardFormulaScope scope = "FINAL".equalsIgnoreCase(scopeRaw)
                ? RewardFormulaScope.FINAL
                : RewardFormulaScope.STEP;
            target.rewardFormulaSet.addTerm(new RewardFormulaTerm(name, scope, expression));
        }
    }

    private static String stringValue(Object value) {
        return value instanceof String ? (String) value : "";
    }

    private static Double doubleValue(Object value) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value instanceof String text) {
            try {
                return Double.parseDouble(text);
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Double> doubleMapValue(Object value) {
        if (!(value instanceof Map<?, ?> raw)) {
            return Map.of();
        }
        Map<String, Object> values = (Map<String, Object>) raw;
        Map<String, Double> result = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : values.entrySet()) {
            Double number = doubleValue(entry.getValue());
            if (entry.getKey() != null && number != null && !number.isNaN() && !number.isInfinite()) {
                result.put(entry.getKey(), number);
            }
        }
        return result;
    }

    private static Boolean booleanValue(Object value) {
        if (value instanceof Boolean) {
            return (Boolean) value;
        }
        if (value instanceof String text) {
            return Boolean.parseBoolean(text);
        }
        return null;
    }
}
