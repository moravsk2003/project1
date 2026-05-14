package com.rustbuilder.ai.rl.supervisor;

import com.rustbuilder.ai.rl.RLRewardConfig;

/**
 * A sanitized command produced by an LLM supervisor implementation.
 */
public final class SupervisorDecision {
    private final SupervisorAction action;
    private final Double proposedEpsilon;
    private final RLRewardConfig proposedRewardConfig;
    private final Boolean proposedUse2dCnn;
    private final String proposedModelName;
    private final String reportModelName;
    private final Integer reportStartEpoch;
    private final Integer reportEndEpoch;
    private final String reason;
    private final LlmSupervisorConfig.CallFrequency proposedCallFrequency;
    private final Double confidence;
    private final String riskLevel;
    private final String expectedEffect;
    private final String rollbackPlan;
    private final String changeMagnitude;
    private final Boolean requiresBranchTest;

    private SupervisorDecision(SupervisorAction action,
                               Double proposedEpsilon,
                               RLRewardConfig proposedRewardConfig,
                               Boolean proposedUse2dCnn,
                               String proposedModelName,
                               String reportModelName,
                               Integer reportStartEpoch,
                               Integer reportEndEpoch,
                               String reason) {
        this(action,
            proposedEpsilon,
            proposedRewardConfig,
            proposedUse2dCnn,
            proposedModelName,
            reportModelName,
            reportStartEpoch,
            reportEndEpoch,
            reason,
            null,
            null,
            null,
            null,
            null,
            null,
            null);
    }

    private SupervisorDecision(SupervisorAction action,
                               Double proposedEpsilon,
                               RLRewardConfig proposedRewardConfig,
                               Boolean proposedUse2dCnn,
                               String proposedModelName,
                               String reportModelName,
                               Integer reportStartEpoch,
                               Integer reportEndEpoch,
                               String reason,
                               LlmSupervisorConfig.CallFrequency proposedCallFrequency,
                               Double confidence,
                               String riskLevel,
                               String expectedEffect,
                               String rollbackPlan,
                               String changeMagnitude,
                               Boolean requiresBranchTest) {
        this.action = action != null ? action : SupervisorAction.KEEP_GOING;
        this.proposedEpsilon = proposedEpsilon;
        this.proposedRewardConfig = proposedRewardConfig != null ? proposedRewardConfig.clone() : null;
        this.proposedUse2dCnn = proposedUse2dCnn;
        this.proposedModelName = proposedModelName;
        this.reportModelName = reportModelName;
        this.reportStartEpoch = reportStartEpoch;
        this.reportEndEpoch = reportEndEpoch;
        this.reason = reason != null ? reason : "";
        this.proposedCallFrequency = proposedCallFrequency;
        this.confidence = confidence;
        this.riskLevel = riskLevel != null ? riskLevel : "";
        this.expectedEffect = expectedEffect != null ? expectedEffect : "";
        this.rollbackPlan = rollbackPlan != null ? rollbackPlan : "";
        this.changeMagnitude = changeMagnitude != null ? changeMagnitude : "";
        this.requiresBranchTest = requiresBranchTest;
    }

    public static SupervisorDecision keepGoing(String reason) {
        return new SupervisorDecision(SupervisorAction.KEEP_GOING, null, null, null, null, null, null, null, reason);
    }

    public static SupervisorDecision setEpsilon(double epsilon, String reason) {
        return new SupervisorDecision(SupervisorAction.SET_EPSILON, epsilon, null, null, null, null, null, null, reason);
    }

    public static SupervisorDecision replaceRewardConfig(RLRewardConfig rewardConfig, String reason) {
        return new SupervisorDecision(SupervisorAction.REPLACE_REWARD_CONFIG, null, rewardConfig, null, null, null, null, null, reason);
    }

    public static SupervisorDecision stopTraining(String reason) {
        return new SupervisorDecision(SupervisorAction.STOP_TRAINING, null, null, null, null, null, null, null, reason);
    }

    public static SupervisorDecision requestPromotionCheck(String reason) {
        return new SupervisorDecision(SupervisorAction.REQUEST_PROMOTION_CHECK, null, null, null, null, null, null, null, reason);
    }
    
    public static SupervisorDecision startNewRun(Boolean use2dCnn, RLRewardConfig config, String modelName, String reason) {
        return new SupervisorDecision(SupervisorAction.START_NEW_RUN, null, config, use2dCnn, modelName, null, null, null, reason);
    }

    public static SupervisorDecision restartTraining(Boolean use2dCnn, RLRewardConfig config, String modelName, String reason) {
        return new SupervisorDecision(SupervisorAction.RESTART_TRAINING, null, config, use2dCnn, modelName, null, null, null, reason);
    }

    public static SupervisorDecision requestHistoricalReport(String reportModelName, Integer reportStartEpoch, Integer reportEndEpoch, String reason) {
        return new SupervisorDecision(SupervisorAction.REQUEST_HISTORICAL_REPORT, null, null, null, null, reportModelName, reportStartEpoch, reportEndEpoch, reason);
    }

    public static SupervisorDecision promoteBranch(String reason) {
        return new SupervisorDecision(SupervisorAction.PROMOTE_BRANCH, null, null, null, null, null, null, null, reason);
    }

    public static SupervisorDecision jumpToBranch(String modelName, String reason) {
        return new SupervisorDecision(SupervisorAction.JUMP_TO_BRANCH, null, null, null, modelName, null, null, null, reason);
    }

    public SupervisorDecision withCallFrequency(LlmSupervisorConfig.CallFrequency callFrequency) {
        return new SupervisorDecision(action,
            proposedEpsilon,
            proposedRewardConfig,
            proposedUse2dCnn,
            proposedModelName,
            reportModelName,
            reportStartEpoch,
            reportEndEpoch,
            reason,
            callFrequency,
            confidence,
            riskLevel,
            expectedEffect,
            rollbackPlan,
            changeMagnitude,
            requiresBranchTest);
    }

    public SupervisorDecision withAnalysisMetadata(Double confidence,
                                                   String riskLevel,
                                                   String expectedEffect,
                                                   String rollbackPlan,
                                                   String changeMagnitude,
                                                   Boolean requiresBranchTest) {
        return new SupervisorDecision(action,
            proposedEpsilon,
            proposedRewardConfig,
            proposedUse2dCnn,
            proposedModelName,
            reportModelName,
            reportStartEpoch,
            reportEndEpoch,
            reason,
            proposedCallFrequency,
            confidence,
            riskLevel,
            expectedEffect,
            rollbackPlan,
            changeMagnitude,
            requiresBranchTest);
    }

    public SupervisorAction getAction() {
        return action;
    }

    public Double getProposedEpsilon() {
        return proposedEpsilon;
    }

    public RLRewardConfig getProposedRewardConfig() {
        return proposedRewardConfig != null ? proposedRewardConfig.clone() : null;
    }
    
    public Boolean getProposedUse2dCnn() {
        return proposedUse2dCnn;
    }
    
    public String getProposedModelName() {
        return proposedModelName;
    }
    
    public String getReason() {
        return reason;
    }

    public String getReportModelName() {
        return reportModelName;
    }

    public Integer getReportStartEpoch() {
        return reportStartEpoch;
    }

    public Integer getReportEndEpoch() {
        return reportEndEpoch;
    }

    public LlmSupervisorConfig.CallFrequency getProposedCallFrequency() {
        return proposedCallFrequency;
    }

    public Double getConfidence() {
        return confidence;
    }

    public String getRiskLevel() {
        return riskLevel;
    }

    public String getExpectedEffect() {
        return expectedEffect;
    }

    public String getRollbackPlan() {
        return rollbackPlan;
    }

    public String getChangeMagnitude() {
        return changeMagnitude;
    }

    public Boolean getRequiresBranchTest() {
        return requiresBranchTest;
    }
}
