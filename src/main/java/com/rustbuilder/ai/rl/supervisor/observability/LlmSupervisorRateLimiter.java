package com.rustbuilder.ai.rl.supervisor.observability;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;

/**
 * Conservative local limiter for external LLM supervisor calls.
 */
public class LlmSupervisorRateLimiter {
    public static final int DEFAULT_REQUESTS_PER_MINUTE = 10;
    public static final int DEFAULT_REQUESTS_PER_DAY = 1_500;
    public static final int DEFAULT_TOKENS_PER_MINUTE = 1_000_000;

    private final int requestsPerMinute;
    private final int requestsPerDay;
    private final int tokensPerMinute;
    private final Clock clock;

    private Instant minuteWindowStart;
    private int minuteRequests;
    private int minuteTokens;
    private LocalDate dayWindow;
    private int dayRequests;

    public LlmSupervisorRateLimiter() {
        this(DEFAULT_REQUESTS_PER_MINUTE, DEFAULT_REQUESTS_PER_DAY, DEFAULT_TOKENS_PER_MINUTE, Clock.systemDefaultZone());
    }

    LlmSupervisorRateLimiter(int requestsPerMinute, int requestsPerDay, int tokensPerMinute, Clock clock) {
        this.requestsPerMinute = Math.max(1, requestsPerMinute);
        this.requestsPerDay = Math.max(1, requestsPerDay);
        this.tokensPerMinute = Math.max(1, tokensPerMinute);
        this.clock = clock != null ? clock : Clock.systemDefaultZone();
        this.minuteWindowStart = Instant.now(this.clock);
        this.dayWindow = LocalDate.now(this.clock);
    }

    public synchronized Duration reserveDelay(int estimatedTokens) {
        Instant now = Instant.now(clock);
        LocalDate today = LocalDate.now(clock);
        resetWindowsIfNeeded(now, today);

        int tokens = Math.max(1, estimatedTokens);
        Duration delay = Duration.ZERO;

        if (minuteRequests + 1 > requestsPerMinute || minuteTokens + tokens > tokensPerMinute) {
            delay = max(delay, Duration.between(now, minuteWindowStart.plus(Duration.ofMinutes(1))));
        }
        if (dayRequests + 1 > requestsPerDay) {
            delay = max(delay, Duration.between(now, nextDayStart(today, clock.getZone())));
        }

        if (delay.isZero() || delay.isNegative()) {
            minuteRequests++;
            minuteTokens = Math.min(tokensPerMinute, minuteTokens + tokens);
            dayRequests++;
            return Duration.ZERO;
        }
        return delay;
    }

    public synchronized void reserveNowAfterDelay(int estimatedTokens) {
        Instant now = Instant.now(clock);
        LocalDate today = LocalDate.now(clock);
        resetWindowsIfNeeded(now, today);
        minuteRequests++;
        minuteTokens = Math.min(tokensPerMinute, minuteTokens + Math.max(1, estimatedTokens));
        dayRequests++;
    }

    public int getRequestsPerMinute() {
        return requestsPerMinute;
    }

    public int getRequestsPerDay() {
        return requestsPerDay;
    }

    public int getTokensPerMinute() {
        return tokensPerMinute;
    }

    private void resetWindowsIfNeeded(Instant now, LocalDate today) {
        if (!today.equals(dayWindow)) {
            dayWindow = today;
            dayRequests = 0;
        }
        if (!now.isBefore(minuteWindowStart.plus(Duration.ofMinutes(1)))) {
            minuteWindowStart = now;
            minuteRequests = 0;
            minuteTokens = 0;
        }
    }

    private static Instant nextDayStart(LocalDate today, ZoneId zoneId) {
        return today.plusDays(1).atStartOfDay(zoneId).toInstant();
    }

    private static Duration max(Duration a, Duration b) {
        return a.compareTo(b) >= 0 ? a : b;
    }
}
