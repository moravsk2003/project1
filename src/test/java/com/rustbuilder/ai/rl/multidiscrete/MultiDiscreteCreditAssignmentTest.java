package com.rustbuilder.ai.rl.multidiscrete;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

import com.rustbuilder.core.placement.PlacementError;

class MultiDiscreteCreditAssignmentTest {

    @Test
    void noSupportBlamesOnlyPlacementHeads() {
        assertArrayEquals(
            new double[] {0.0, 1.0, 1.0, 1.0, 1.0},
            MultiDiscreteCreditAssignment.forPlacement(false, PlacementError.NO_SUPPORT),
            0.0
        );
    }

    @Test
    void firstBadSocketBlamesOnlyType() {
        assertArrayEquals(
            new double[] {1.0, 0.0, 0.0, 0.0, 0.0},
            MultiDiscreteCreditAssignment.forPlacement(false, PlacementError.BAD_SOCKET_IS_FIRST),
            0.0
        );
    }

    @Test
    void successfulPlacementCreditsEveryHead() {
        assertArrayEquals(
            new double[] {1.0, 1.0, 1.0, 1.0, 1.0},
            MultiDiscreteCreditAssignment.forPlacement(true, PlacementError.NONE),
            0.0
        );
    }
}
