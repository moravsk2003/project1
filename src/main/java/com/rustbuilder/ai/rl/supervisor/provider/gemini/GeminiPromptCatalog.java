package com.rustbuilder.ai.rl.supervisor.provider.gemini;


import com.rustbuilder.ai.rl.domain.log.StopReason;
import com.rustbuilder.ai.rl.domain.RLRewardConfig;
import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorObservation;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Gemini-specific prompt policy and JSON schema hints.
 */
public final class GeminiPromptCatalog {
    private static final String BASE_SYSTEM_PROMPT = """
        You supervise reinforcement learning for a Rust base builder.
        Return exactly one JSON object and no markdown.
        Use only actions listed in observation.allowedActions.
        Use STOP_TRAINING only when the run is clearly wasting the remaining budget.
        Use START_NEW_RUN only when observation.trainingContext.trainingRunning is false and provide modelName.
        Do not include rewardConfig or rewardTerms with START_NEW_RUN; new runs reset reward state to defaults.
        During active training, stopReason is the last episode stop reason, not proof that the whole run stopped.
        Never use START_NEW_RUN to continue an active run; use KEEP_GOING unless a listed training action is justified.
        Always set callFrequency to VERY_SOON, SOON, MEDIUM, or LONG to choose the next review cadence.
        Do not omit callFrequency, even when the action is KEEP_GOING.
        Use faster cadence while unstable or after branch changes; use longer cadence when learning is stable.
        You may use SET_CURRICULUM_OBJECTIVE to choose measurable intermediate training goals. Base it on stability windows, not one lucky best base.
        The current objective lives in observation.trainingContext.curriculumObjective; progress is in observation.trendMetrics.curriculumObjectiveProgress.
        Use REQUEST_HISTORICAL_REPORT only when idle and you need prior epoch trends before starting a run.
        Use PROMOTE_BRANCH only when observation.trendMetrics.latestBranchPromotionOpen is true. Never promote a branch whose latestBranchComparison.promotionApplied is true.
        Use JUMP_TO_BRANCH only with a modelName from observation.trendMetrics.knownBranchModelNames.
        Treat observation.trendMetrics.appliedChangeJournal as the source of truth for live changes; prior decisions may be proposals or branch tests that were not promoted.
        Zero eval metrics are often expected before a protected TC/base exists; check observation.trendMetrics.currentEpisodeEvaluationDiagnostics before calling the evaluator broken.
        If observation.trendMetrics.supervisorHealth.degraded is true, avoid stale high-impact actions unless current evidence is explicit.
        Never invent fields outside rewardConfig or rewardTerms.
        Formula terms may use only variables and functions from observation.rewardFormulaContract.
        Use trainingRemainingSeconds and trainingDeadline to avoid disruptive changes near the end of a run.
        Include confidence, riskLevel, expectedEffect, rollbackPlan, changeMagnitude, and requiresBranchTest when useful.
        """;

    public String taskForStage(String stage) {
        if ("FINAL_DECISION".equals(stage)) {
            return "Finalize the prior proposal into exactly one executable supervisor decision JSON. You may revise it if the proposal is risky or unsupported.";
        }
        return "Analyze this compact RL observation, diagnose the most actionable issue, and return one proposed supervisor decision JSON.";
    }

    public String systemPrompt(String stage, LlmSupervisorConfig config) {
        return BASE_SYSTEM_PROMPT + "\n" + cautionPrompt(cautionLevel(config)) + "\n" + stagePrompt(stage);
    }

    public Map<String, Object> supervisorPolicy(LlmSupervisorConfig config) {
        LlmSupervisorConfig.CautionLevel cautionLevel = cautionLevel(config);
        Map<String, Object> policy = new LinkedHashMap<>();
        policy.put("cautionLevel", cautionLevel.name());
        policy.put("decisionMode", decisionMode(config).name());
        policy.put("largeRewardChanges", "Must be branch-tested before replacing the live branch.");
        policy.put("useKeepGoing", cautionLevel == LlmSupervisorConfig.CautionLevel.CONSERVATIVE
            ? "Use when evidence is weak."
            : "Use only when no actionable diagnosis exists.");
        policy.put("highImpactActions", List.of(
            "REPLACE_REWARD_CONFIG",
            "large SET_EPSILON",
            "STOP_TRAINING",
            "SET_CURRICULUM_OBJECTIVE",
            "PROMOTE_BRANCH",
            "JUMP_TO_BRANCH",
            "START_NEW_RUN"));
        return policy;
    }

    public double temperatureFor(LlmSupervisorConfig config) {
        return switch (cautionLevel(config)) {
            case CONSERVATIVE -> 0.15;
            case BALANCED -> 0.2;
            case BOLD -> 0.35;
            case EXPERIMENTAL -> 0.45;
        };
    }

    public boolean isHighImpactDecision(SupervisorDecision decision,
                                        SupervisorObservation observation,
                                        LlmSupervisorConfig config) {
        if (decision == null || decision.getAction() == null) {
            return false;
        }
        return switch (decision.getAction()) {
            case REPLACE_REWARD_CONFIG, STOP_TRAINING, START_NEW_RUN, RESTART_TRAINING,
                    SET_CURRICULUM_OBJECTIVE, PROMOTE_BRANCH, JUMP_TO_BRANCH -> true;
            case SET_EPSILON -> {
                Double proposed = decision.getProposedEpsilon();
                yield observation != null
                    && proposed != null
                    && Math.abs(proposed - observation.epsilon) > epsilonHighImpactDelta(config);
            }
            default -> false;
        };
    }

    public Map<String, Object> decisionSchemaText() {
        Map<String, Object> schema = new LinkedHashMap<>();
        schema.put("action", "one of observation.allowedActions");
        schema.put("epsilon", "optional number for SET_EPSILON");
        schema.put("rewardConfig", "optional numeric patch for RLRewardConfig fields; only for REPLACE_REWARD_CONFIG");
        schema.put("rewardTerms", "optional array of {name, scope: STEP|FINAL, expression}; only for REPLACE_REWARD_CONFIG");
        schema.put("objectiveId", "required safe id for SET_CURRICULUM_OBJECTIVE");
        schema.put("objectiveDescription", "required measurable intermediate goal for SET_CURRICULUM_OBJECTIVE");
        schema.put("successCriteria", "numeric thresholds for SET_CURRICULUM_OBJECTIVE, e.g. invalidActionRateMax, medianBlocksMin, tcPresentRateMin");
        schema.put("failureSignals", "optional numeric regression thresholds for SET_CURRICULUM_OBJECTIVE");
        schema.put("modelName", "required safe unique name for START_NEW_RUN or JUMP_TO_BRANCH");
        schema.put("reportModelName", "required for REQUEST_HISTORICAL_REPORT");
        schema.put("reportStartEpoch", "optional non-negative integer");
        schema.put("reportEndEpoch", "optional non-negative integer");
        schema.put("callFrequency", "required cadence preset: VERY_SOON=base*0.25, SOON=base*0.5, MEDIUM=base*1, LONG=base*2");
        schema.put("confidence", "optional 0.0-1.0 estimate");
        schema.put("riskLevel", "optional LOW, MEDIUM, HIGH, or CRITICAL");
        schema.put("expectedEffect", "optional short expected metric or behavior improvement");
        schema.put("rollbackPlan", "optional short rollback/checkpoint plan for risky changes");
        schema.put("changeMagnitude", "optional NONE, SMALL, MEDIUM, LARGE, or EXPERIMENTAL");
        schema.put("requiresBranchTest", "optional boolean; true for reward or large epsilon changes");
        schema.put("reason", "short explanation");
        return schema;
    }

    public Map<String, Object> responseSchema() {
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("action", Map.of(
            "type", "STRING",
            "enum", List.of(
                "KEEP_GOING",
                "SET_EPSILON",
                "REPLACE_REWARD_CONFIG",
                "SET_CURRICULUM_OBJECTIVE",
                "REQUEST_PROMOTION_CHECK",
                "START_NEW_RUN",
                "STOP_TRAINING",
                "REQUEST_HISTORICAL_REPORT",
                "PROMOTE_BRANCH",
                "JUMP_TO_BRANCH")));
        properties.put("epsilon", Map.of("type", "NUMBER"));
        properties.put("rewardConfig", Map.of("type", "OBJECT"));
        properties.put("rewardTerms", Map.of(
            "type", "ARRAY",
            "items", Map.of(
                "type", "OBJECT",
                "properties", Map.of(
                    "name", Map.of("type", "STRING"),
                    "scope", Map.of("type", "STRING", "enum", List.of("STEP", "FINAL")),
                    "expression", Map.of("type", "STRING")))));
        properties.put("objectiveId", Map.of("type", "STRING"));
        properties.put("objectiveDescription", Map.of("type", "STRING"));
        properties.put("successCriteria", Map.of("type", "OBJECT"));
        properties.put("failureSignals", Map.of("type", "OBJECT"));
        properties.put("use2dCnn", Map.of("type", "BOOLEAN"));
        properties.put("modelName", Map.of("type", "STRING"));
        properties.put("reportModelName", Map.of("type", "STRING"));
        properties.put("reportStartEpoch", Map.of("type", "INTEGER"));
        properties.put("reportEndEpoch", Map.of("type", "INTEGER"));
        properties.put("callFrequency", Map.of(
            "type", "STRING",
            "enum", List.of("VERY_SOON", "SOON", "MEDIUM", "LONG")));
        properties.put("confidence", Map.of("type", "NUMBER"));
        properties.put("riskLevel", Map.of(
            "type", "STRING",
            "enum", List.of("LOW", "MEDIUM", "HIGH", "CRITICAL")));
        properties.put("expectedEffect", Map.of("type", "STRING"));
        properties.put("rollbackPlan", Map.of("type", "STRING"));
        properties.put("changeMagnitude", Map.of(
            "type", "STRING",
            "enum", List.of("NONE", "SMALL", "MEDIUM", "LARGE", "EXPERIMENTAL")));
        properties.put("requiresBranchTest", Map.of("type", "BOOLEAN"));
        properties.put("reason", Map.of("type", "STRING"));
        return Map.of(
            "type", "OBJECT",
            "properties", properties,
            "required", List.of("action", "callFrequency", "reason"),
            "propertyOrdering", List.of(
                "action", "epsilon", "rewardConfig", "rewardTerms",
                "objectiveId", "objectiveDescription", "successCriteria", "failureSignals",
                "use2dCnn", "modelName", "reportModelName", "reportStartEpoch", "reportEndEpoch",
                "callFrequency", "confidence", "riskLevel", "expectedEffect", "rollbackPlan",
                "changeMagnitude", "requiresBranchTest", "reason"));
    }

    private String cautionPrompt(LlmSupervisorConfig.CautionLevel cautionLevel) {
        return switch (cautionLevel) {
            case CONSERVATIVE -> """
                Caution mode: CONSERVATIVE.
                Prefer KEEP_GOING unless there is clear evidence.
                Use SET_EPSILON only for small exploration adjustments.
                Use REPLACE_REWARD_CONFIG sparingly with tiny numeric changes.
                """;
            case BALANCED -> """
                Caution mode: BALANCED.
                Prefer evidence-based action over passive KEEP_GOING when metrics are stalled or regressing.
                Use bounded reward and epsilon changes, and rely on branch tests for reward tuning.
                """;
            case BOLD -> """
                Caution mode: BOLD.
                Act like a real supervisor: propose substantial reward/epsilon hypotheses when the run is stalled, degenerate, or optimizing the wrong behavior.
                Large reward changes must be branch-tested and include rollbackPlan, expectedEffect, riskLevel, and changeMagnitude.
                """;
            case EXPERIMENTAL -> """
                Caution mode: EXPERIMENTAL.
                Aggressive hypotheses are allowed for research runs, but destructive live changes are not.
                Large changes must go through candidate branches, and STOP_TRAINING requires strong budget-waste evidence.
                """;
        };
    }

    private String stagePrompt(String stage) {
        if ("FINAL_DECISION".equals(stage)) {
            return """
                Stage: FINAL_DECISION.
                You are reviewing your priorProposal. Return the final executable decision only.
                Keep the action if the proposal is well-supported; downgrade to KEEP_GOING or REQUEST_PROMOTION_CHECK if evidence is weak.
                """;
        }
        return """
            Stage: ANALYZE_AND_PROPOSE.
            First reason internally about objective, recentSupervisorDecisions, branch results, reward breakdown, invalid actions, and time budget.
            Then return one proposed decision JSON using the schema. No markdown, no extra prose.
            """;
    }

    private double epsilonHighImpactDelta(LlmSupervisorConfig config) {
        return switch (cautionLevel(config)) {
            case CONSERVATIVE -> 0.05;
            case BALANCED -> 0.12;
            case BOLD -> 0.25;
            case EXPERIMENTAL -> 0.35;
        };
    }

    private LlmSupervisorConfig.CautionLevel cautionLevel(LlmSupervisorConfig config) {
        return config != null
            ? config.getCautionLevel()
            : LlmSupervisorConfig.CautionLevel.BALANCED;
    }

    private LlmSupervisorConfig.DecisionMode decisionMode(LlmSupervisorConfig config) {
        return config != null
            ? config.getDecisionMode()
            : LlmSupervisorConfig.DecisionMode.SINGLE_STEP;
    }
}
