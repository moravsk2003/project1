package com.rustbuilder.ai.rl;

/**
 * Enumeration of possible reasons for a placement to fail or be invalid.
 */
public enum PlacementError {
    NONE,
    COLLISION,
    NO_SUPPORT,
    BAD_SOCKET_IS_FIRST,
    BAD_SOCKET_NO_TARGET,
    BAD_SOCKET_WRONG_TARGET_TYPE,
    BAD_SOCKET_NO_SOCKET_ALIGNMENT,
    BAD_SOCKET_CENTERDIST_REJECT,
    GLOBAL_LIMIT,
    FLOOR_CONSTRAINT,
    UNKNOWN
}
