package com.rustbuilder.core.placement;

/**
 * Domain-level reasons why a requested placement cannot be applied.
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
    BAD_SOCKET,
    OUT_OF_BOUNDS,
    FLOOR_CONSTRAINT,
    UNKNOWN
}
