package com.rustbuilder.ai.rl.supervisor.config;

import java.io.Serializable;

/**
 * Runtime limits for the LLM supervisor hook.
 */
public class LlmSupervisorConfig implements Serializable, Cloneable {
    private static final long serialVersionUID = 1L;

    public static final int DEFAULT_CALL_INTERVAL_EPISODES = 1000;
    private static final int MIN_CALL_INTERVAL_EPISODES = 1;

    public enum CallFrequency {
        VERY_SOON("Very soon", 0.25),
        SOON("Soon", 0.5),
        MEDIUM("Medium", 1.0),
        LONG("Long", 2.0);

        private final String label;
        private final double multiplier;

        CallFrequency(String label, double multiplier) {
            this.label = label;
            this.multiplier = multiplier;
        }

        public double getMultiplier() {
            return multiplier;
        }

        @Override
        public String toString() {
            return label + " (x" + formatMultiplier(multiplier) + ")";
        }

        private static String formatMultiplier(double value) {
            return value == Math.rint(value)
                ? String.valueOf((int) value)
                : String.valueOf(value);
        }
    }

    public enum CautionLevel {
        CONSERVATIVE("Safe", "Small live changes; prefer waiting unless the evidence is clear."),
        BALANCED("Balanced", "Default: bounded changes with branch tests for high-impact moves."),
        BOLD("Bold", "Allows larger hypotheses when training is stalled or optimizing the wrong thing."),
        EXPERIMENTAL("Research", "Aggressive research mode; validator still clamps unsafe live changes.");

        private final String label;
        private final String description;

        CautionLevel(String label, String description) {
            this.label = label;
            this.description = description;
        }

        public String getDescription() {
            return description;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    public enum DecisionMode {
        SINGLE_STEP("Single step"),
        TWO_STAGE_HIGH_IMPACT("Two-stage for big changes"),
        TWO_STAGE_ALWAYS("Two-stage always");

        private final String label;

        DecisionMode(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    private boolean enabled;
    private int callIntervalEpisodes = DEFAULT_CALL_INTERVAL_EPISODES;
    private CallFrequency callFrequency = CallFrequency.MEDIUM;
    private CautionLevel cautionLevel = CautionLevel.BALANCED;
    private DecisionMode decisionMode = DecisionMode.TWO_STAGE_HIGH_IMPACT;
    private String branchId = "candidate";
    private boolean allowEpsilonChanges = true;
    private boolean allowRewardConfigChanges = true;
    private String externalCommand = "";
    private String primaryModel = "";
    private String fallbackModel = "";
    private transient String apiKey = "";
    private LlmSupervisorApplyMode applyMode = LlmSupervisorApplyMode.AUTO_APPLY;
    private long autopilotTrainingDurationMs = 0L;

    public static LlmSupervisorConfig disabled() {
        LlmSupervisorConfig config = new LlmSupervisorConfig();
        config.setEnabled(false);
        return config;
    }

    public static LlmSupervisorConfig enabledDefault(String branchId) {
        LlmSupervisorConfig config = new LlmSupervisorConfig();
        config.setEnabled(true);
        config.setBranchId(branchId);
        return config;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public int getCallIntervalEpisodes() {
        return callIntervalEpisodes;
    }

    public void setCallIntervalEpisodes(int callIntervalEpisodes) {
        this.callIntervalEpisodes = Math.max(MIN_CALL_INTERVAL_EPISODES, callIntervalEpisodes);
    }

    public CallFrequency getCallFrequency() {
        return callFrequency != null ? callFrequency : CallFrequency.MEDIUM;
    }

    public void setCallFrequency(CallFrequency callFrequency) {
        this.callFrequency = callFrequency != null ? callFrequency : CallFrequency.MEDIUM;
    }

    public CautionLevel getCautionLevel() {
        return cautionLevel != null ? cautionLevel : CautionLevel.BALANCED;
    }

    public void setCautionLevel(CautionLevel cautionLevel) {
        this.cautionLevel = cautionLevel != null ? cautionLevel : CautionLevel.BALANCED;
    }

    public DecisionMode getDecisionMode() {
        return decisionMode != null ? decisionMode : DecisionMode.TWO_STAGE_HIGH_IMPACT;
    }

    public void setDecisionMode(DecisionMode decisionMode) {
        this.decisionMode = decisionMode != null ? decisionMode : DecisionMode.TWO_STAGE_HIGH_IMPACT;
    }

    public int getEffectiveCallIntervalEpisodes() {
        return Math.max(MIN_CALL_INTERVAL_EPISODES,
            (int) Math.round(getCallIntervalEpisodes() * getCallFrequency().getMultiplier()));
    }

    public String getBranchId() {
        return branchId;
    }

    public void setBranchId(String branchId) {
        this.branchId = (branchId == null || branchId.isBlank()) ? "candidate" : branchId.trim();
    }

    public boolean isAllowEpsilonChanges() {
        return allowEpsilonChanges;
    }

    public void setAllowEpsilonChanges(boolean allowEpsilonChanges) {
        this.allowEpsilonChanges = allowEpsilonChanges;
    }

    public boolean isAllowRewardConfigChanges() {
        return allowRewardConfigChanges;
    }

    public void setAllowRewardConfigChanges(boolean allowRewardConfigChanges) {
        this.allowRewardConfigChanges = allowRewardConfigChanges;
    }

    public String getExternalCommand() {
        return externalCommand;
    }

    public void setExternalCommand(String externalCommand) {
        this.externalCommand = externalCommand != null ? externalCommand.trim() : "";
    }

    public String getPrimaryModel() {
        return primaryModel != null ? primaryModel : "";
    }

    public void setPrimaryModel(String primaryModel) {
        this.primaryModel = primaryModel != null ? primaryModel.trim() : "";
    }

    public String getFallbackModel() {
        return fallbackModel != null ? fallbackModel : "";
    }

    public void setFallbackModel(String fallbackModel) {
        this.fallbackModel = fallbackModel != null ? fallbackModel.trim() : "";
    }

    public String getApiKey() {
        return apiKey != null ? apiKey : "";
    }

    public void setApiKey(String apiKey) {
        this.apiKey = apiKey != null ? apiKey.trim() : "";
    }

    public LlmSupervisorApplyMode getApplyMode() {
        return applyMode != null ? applyMode : LlmSupervisorApplyMode.AUTO_APPLY;
    }

    public void setApplyMode(LlmSupervisorApplyMode applyMode) {
        this.applyMode = applyMode != null ? applyMode : LlmSupervisorApplyMode.AUTO_APPLY;
    }

    public long getAutopilotTrainingDurationMs() {
        return Math.max(0L, autopilotTrainingDurationMs);
    }

    public void setAutopilotTrainingDurationMs(long autopilotTrainingDurationMs) {
        this.autopilotTrainingDurationMs = Math.max(0L, autopilotTrainingDurationMs);
    }

    @Override
    public LlmSupervisorConfig clone() {
        try {
            return (LlmSupervisorConfig) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError(e);
        }
    }
}
