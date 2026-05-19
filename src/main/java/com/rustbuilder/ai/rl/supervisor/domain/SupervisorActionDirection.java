package com.rustbuilder.ai.rl.supervisor.domain;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Coarse supervisor intent groups used by the first LLM call.
 */
public enum SupervisorActionDirection {
    CONTINUE("Continue observing", SupervisorAction.KEEP_GOING),
    EXPLORATION("Tune exploration", SupervisorAction.SET_EPSILON),
    REWARD_TUNING("Tune rewards", SupervisorAction.REPLACE_REWARD_CONFIG),
    CURRICULUM("Set curriculum objective", SupervisorAction.SET_CURRICULUM_OBJECTIVE),
    TRAINING_CONTROL("Control active training",
        SupervisorAction.STOP_TRAINING,
        SupervisorAction.REQUEST_PROMOTION_CHECK),
    STARTUP("Start or resume training",
        SupervisorAction.START_NEW_RUN,
        SupervisorAction.LOAD_EXISTING_MODEL),
    HISTORICAL_REVIEW("Read historical logs", SupervisorAction.REQUEST_HISTORICAL_REPORT),
    BRANCH_MANAGEMENT("Manage branch models",
        SupervisorAction.PROMOTE_BRANCH,
        SupervisorAction.JUMP_TO_BRANCH);

    private final String label;
    private final List<SupervisorAction> actions;

    SupervisorActionDirection(String label, SupervisorAction... actions) {
        this.label = label;
        this.actions = List.of(actions);
    }

    public String getLabel() {
        return label;
    }

    public List<String> allowedActionNames(List<String> currentlyAllowedActions) {
        Set<String> allowed = currentlyAllowedActions != null
            ? Set.copyOf(currentlyAllowedActions)
            : Set.of();
        List<String> names = new ArrayList<>();
        for (SupervisorAction action : actions) {
            if (allowed.contains(action.name())) {
                names.add(action.name());
            }
        }
        return names;
    }

    public boolean allows(SupervisorAction action, List<String> currentlyAllowedActions) {
        if (action == null) {
            return false;
        }
        return allowedActionNames(currentlyAllowedActions).contains(action.name());
    }

    public Map<String, Object> toMap(List<String> currentlyAllowedActions) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("direction", name());
        map.put("label", label);
        map.put("actions", allowedActionNames(currentlyAllowedActions));
        return map;
    }

    public static List<SupervisorActionDirection> allowedDirectionsFor(List<String> currentlyAllowedActions) {
        List<SupervisorActionDirection> result = new ArrayList<>();
        for (SupervisorActionDirection direction : values()) {
            if (!direction.allowedActionNames(currentlyAllowedActions).isEmpty()) {
                result.add(direction);
            }
        }
        return result;
    }

    public static SupervisorActionDirection fromId(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        String normalized = value.trim().toUpperCase(Locale.ROOT).replace('-', '_').replace(' ', '_');
        try {
            return SupervisorActionDirection.valueOf(normalized);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }
}
