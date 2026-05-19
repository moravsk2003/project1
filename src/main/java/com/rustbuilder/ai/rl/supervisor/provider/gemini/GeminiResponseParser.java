package com.rustbuilder.ai.rl.supervisor.provider.gemini;

import com.rustbuilder.ai.rl.domain.RLRewardConfig;
import com.rustbuilder.ai.rl.supervisor.serialization.SimpleJson;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorActionDirection;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDecision;
import com.rustbuilder.ai.rl.supervisor.domain.SupervisorDirectionDecision;
import com.rustbuilder.ai.rl.supervisor.serialization.SupervisorJson;

import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Extracts and validates supervisor decision JSON from Gemini responses.
 */
public final class GeminiResponseParser {
    private static final Pattern JSON_OBJECT_PATTERN = Pattern.compile("\\{[\\s\\S]*}");

    public String extractText(String responseBody) {
        Object parsed = SimpleJson.parse(responseBody);
        if (!(parsed instanceof Map<?, ?> root)) {
            return "";
        }
        Object candidatesObject = root.get("candidates");
        if (!(candidatesObject instanceof List<?> candidates) || candidates.isEmpty()) {
            return "";
        }
        Object first = candidates.get(0);
        if (!(first instanceof Map<?, ?> candidate)) {
            return "";
        }
        Object contentObject = candidate.get("content");
        if (!(contentObject instanceof Map<?, ?> content)) {
            return "";
        }
        Object partsObject = content.get("parts");
        if (!(partsObject instanceof List<?> parts)) {
            return "";
        }
        StringBuilder text = new StringBuilder();
        for (Object partObject : parts) {
            if (partObject instanceof Map<?, ?> part) {
                Object value = part.get("text");
                if (value instanceof String s) {
                    if (text.length() > 0) {
                        text.append('\n');
                    }
                    text.append(s);
                }
            }
        }
        return text.toString().trim();
    }

    public String parseModelJson(String text) {
        if (text == null || text.isBlank()) {
            return null;
        }
        try {
            SimpleJson.parse(text);
            return text;
        } catch (RuntimeException ignored) {
            Matcher matcher = JSON_OBJECT_PATTERN.matcher(text);
            if (matcher.find()) {
                String candidate = matcher.group();
                try {
                    SimpleJson.parse(candidate);
                    return candidate;
                } catch (RuntimeException ignoredAgain) {
                    return null;
                }
            }
            return null;
        }
    }

    public SupervisorDecision parseDecision(String decisionJson,
                                            Supplier<RLRewardConfig> currentRewardConfigSupplier) {
        RLRewardConfig currentRewardConfig = currentRewardConfigSupplier != null
            ? currentRewardConfigSupplier.get()
            : RLRewardConfig.createDefault();
        SupervisorDecision decision = SupervisorJson.decisionFromJson(decisionJson, currentRewardConfig);
        if (decision.getProposedCallFrequency() == null) {
            throw new IllegalArgumentException("missing required callFrequency");
        }
        return decision;
    }

    public SupervisorDirectionDecision parseDirection(String directionJson,
                                                      List<SupervisorActionDirection> allowedDirections) {
        Object parsed = SimpleJson.parse(directionJson);
        if (!(parsed instanceof Map<?, ?> map)) {
            throw new IllegalArgumentException("direction response was not a JSON object");
        }
        SupervisorActionDirection direction = SupervisorActionDirection.fromId(stringValue(map.get("direction")));
        if (direction == null) {
            throw new IllegalArgumentException("missing or invalid direction");
        }
        if (allowedDirections != null && !allowedDirections.contains(direction)) {
            throw new IllegalArgumentException("direction is not available now: " + direction.name());
        }
        return new SupervisorDirectionDecision(
            direction,
            stringValue(map.get("reason")),
            doubleValue(map.get("confidence")));
    }

    private static String stringValue(Object value) {
        return value instanceof String ? (String) value : "";
    }

    private static Double doubleValue(Object value) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value instanceof String text) {
            try {
                return Double.parseDouble(text);
            } catch (NumberFormatException ignored) {
                return null;
            }
        }
        return null;
    }
}
