package com.rustbuilder.ai.rl.supervisor;

import java.io.Serializable;

/**
 * Runtime limits for the LLM supervisor hook.
 */
public class LlmSupervisorConfig implements Serializable, Cloneable {
    private static final long serialVersionUID = 1L;

    public static final int DEFAULT_CALL_INTERVAL_EPISODES = 1000;
    private static final int MIN_CALL_INTERVAL_EPISODES = 1;

    private boolean enabled;
    private int callIntervalEpisodes = DEFAULT_CALL_INTERVAL_EPISODES;
    private String branchId = "candidate";
    private boolean allowEpsilonChanges = true;
    private boolean allowRewardConfigChanges = true;
    private String externalCommand = "";
    private transient String apiKey = "";
    private LlmSupervisorApplyMode applyMode = LlmSupervisorApplyMode.AUTO_APPLY;

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

    @Override
    public LlmSupervisorConfig clone() {
        try {
            return (LlmSupervisorConfig) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError(e);
        }
    }
}
