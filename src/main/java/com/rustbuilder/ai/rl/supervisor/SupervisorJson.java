package com.rustbuilder.ai.rl.supervisor;

import com.rustbuilder.ai.rl.RLRewardConfig;
import com.rustbuilder.ai.rl.reward.RewardFormulaScope;
import com.rustbuilder.ai.rl.reward.RewardFormulaTerm;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class SupervisorJson {
    private SupervisorJson() {
    }

    public static String observationToJson(SupervisorObservation obs) {
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
        map.put("allowedActions", List.of(
            SupervisorAction.KEEP_GOING.name(),
            SupervisorAction.SET_EPSILON.name(),
            SupervisorAction.REPLACE_REWARD_CONFIG.name(),
            SupervisorAction.STOP_TRAINING.name(),
            SupervisorAction.REQUEST_PROMOTION_CHECK.name(),
            SupervisorAction.START_NEW_RUN.name(),
            SupervisorAction.RESTART_TRAINING.name(),
            SupervisorAction.REQUEST_HISTORICAL_REPORT.name()));
        
        if (obs.historicalReport != null && !obs.historicalReport.isBlank()) {
            map.put("historicalReport", obs.historicalReport);
        }
        map.put("responseContract",
            "Return one JSON object with action, optional epsilon, optional rewardConfig, optional rewardTerms, and reason.");
        return SimpleJson.stringify(map);
    }

    @SuppressWarnings("unchecked")
    static SupervisorDecision decisionFromJson(String json, RLRewardConfig currentRewardConfig) {
        Object parsed = SimpleJson.parse(json);
        if (!(parsed instanceof Map<?, ?> parsedMap)) {
            return SupervisorDecision.keepGoing("Supervisor did not return a JSON object.");
        }

        Map<String, Object> map = (Map<String, Object>) parsedMap;
        SupervisorAction action = parseAction(stringValue(map.get("action")));
        String reason = stringValue(map.get("reason"));

        if (action == SupervisorAction.SET_EPSILON) {
            Double epsilon = doubleValue(map.get("epsilon"));
            return epsilon != null
                ? SupervisorDecision.setEpsilon(epsilon, reason)
                : SupervisorDecision.keepGoing("SET_EPSILON decision had no epsilon.");
        }

        if (action == SupervisorAction.REPLACE_REWARD_CONFIG) {
            RLRewardConfig rewardConfig = currentRewardConfig != null
                ? currentRewardConfig.clone()
                : RLRewardConfig.createDefault();
            applyRewardConfigPatch(rewardConfig, map.get("rewardConfig"));
            applyRewardTerms(rewardConfig, map.get("rewardTerms"));
            return SupervisorDecision.replaceRewardConfig(rewardConfig, reason);
        }

        if (action == SupervisorAction.STOP_TRAINING) {
            return SupervisorDecision.stopTraining(reason);
        }
        if (action == SupervisorAction.REQUEST_PROMOTION_CHECK) {
            return SupervisorDecision.requestPromotionCheck(reason);
        }
        
        if (action == SupervisorAction.START_NEW_RUN || action == SupervisorAction.RESTART_TRAINING) {
            RLRewardConfig rewardConfig = currentRewardConfig != null
                ? currentRewardConfig.clone()
                : RLRewardConfig.createDefault();
            if (map.containsKey("rewardConfig") || map.containsKey("rewardTerms")) {
                applyRewardConfigPatch(rewardConfig, map.get("rewardConfig"));
                applyRewardTerms(rewardConfig, map.get("rewardTerms"));
            }
            Boolean use2dCnn = booleanValue(map.get("use2dCnn"));
            String modelName = stringValue(map.get("modelName"));
            
            if (action == SupervisorAction.START_NEW_RUN) {
                return SupervisorDecision.startNewRun(use2dCnn, rewardConfig, modelName, reason);
            } else {
                return SupervisorDecision.restartTraining(use2dCnn, rewardConfig, modelName, reason);
            }
        }
        
        if (action == SupervisorAction.REQUEST_HISTORICAL_REPORT) {
            String reportModelName = stringValue(map.get("reportModelName"));
            Integer startEpoch = integerValue(map.get("reportStartEpoch"));
            Integer endEpoch = integerValue(map.get("reportEndEpoch"));
            return SupervisorDecision.requestHistoricalReport(reportModelName, startEpoch, endEpoch, reason);
        }
        
        return SupervisorDecision.keepGoing(reason);
    }

    private static Integer integerValue(Object val) {
        if (val instanceof Number n) {
            return n.intValue();
        }
        if (val instanceof String s) {
            try { return Integer.parseInt(s.trim()); } catch (Exception ignored) {}
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
