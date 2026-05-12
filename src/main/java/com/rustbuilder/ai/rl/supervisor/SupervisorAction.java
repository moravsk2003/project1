package com.rustbuilder.ai.rl.supervisor;

public enum SupervisorAction {
    KEEP_GOING,
    SET_EPSILON,
    REPLACE_REWARD_CONFIG,
    STOP_TRAINING,
    REQUEST_PROMOTION_CHECK,
    START_NEW_RUN,
    RESTART_TRAINING,
    REQUEST_HISTORICAL_REPORT,
    PROMOTE_BRANCH,
    JUMP_TO_BRANCH
}
