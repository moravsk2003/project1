package com.rustbuilder.ai.rl.supervisor.observability;


import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import org.junit.jupiter.api.Test;

class LlmSupervisorRateLimiterTest {

    @Test
    void allowsRequestsWithinConfiguredLimits() {
        LlmSupervisorRateLimiter limiter = new LlmSupervisorRateLimiter(
            10, 1_500, 1_000_000, fixedClock());

        assertEquals(Duration.ZERO, limiter.reserveDelay(1_000));
        assertEquals(Duration.ZERO, limiter.reserveDelay(1_000));
    }

    @Test
    void pausesWhenRequestsPerMinuteLimitIsReached() {
        LlmSupervisorRateLimiter limiter = new LlmSupervisorRateLimiter(
            2, 1_500, 1_000_000, fixedClock());

        assertEquals(Duration.ZERO, limiter.reserveDelay(1_000));
        assertEquals(Duration.ZERO, limiter.reserveDelay(1_000));

        Duration delay = limiter.reserveDelay(1_000);
        assertTrue(delay.compareTo(Duration.ZERO) > 0);
        assertTrue(delay.compareTo(Duration.ofMinutes(1)) <= 0);
    }

    @Test
    void pausesWhenTokensPerMinuteLimitIsReached() {
        LlmSupervisorRateLimiter limiter = new LlmSupervisorRateLimiter(
            10, 1_500, 2_000, fixedClock());

        assertEquals(Duration.ZERO, limiter.reserveDelay(1_500));

        Duration delay = limiter.reserveDelay(1_000);
        assertTrue(delay.compareTo(Duration.ZERO) > 0);
    }

    private static Clock fixedClock() {
        return Clock.fixed(Instant.parse("2026-05-11T12:00:00Z"), ZoneOffset.UTC);
    }
}
