package com.rustbuilder.ai.rl.reward;

import java.io.Serializable;
import java.util.Map;

public class RewardFormulaTerm implements Serializable, Cloneable {
    private static final long serialVersionUID = 1L;
    private static final int MAX_NAME_LENGTH = 64;
    private static final int MAX_EXPRESSION_LENGTH = 256;

    private String name;
    private RewardFormulaScope scope;
    private String expression;
    private boolean enabled = true;

    public RewardFormulaTerm(String name, RewardFormulaScope scope, String expression) {
        this.name = truncate((name == null || name.isBlank()) ? "unnamed_formula" : name.trim(), MAX_NAME_LENGTH);
        this.scope = scope != null ? scope : RewardFormulaScope.STEP;
        this.expression = truncate(expression != null ? expression.trim() : "0", MAX_EXPRESSION_LENGTH);
    }

    public String getName() {
        return name;
    }

    public RewardFormulaScope getScope() {
        return scope;
    }

    public String getExpression() {
        return expression;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public double evaluate(Map<String, Double> variables) {
        if (!enabled) {
            return 0.0;
        }
        return RewardFormulaEvaluator.evaluate(expression, variables);
    }

    @Override
    public RewardFormulaTerm clone() {
        try {
            return (RewardFormulaTerm) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new AssertionError(e);
        }
    }

    private static String truncate(String value, int maxLength) {
        return value.length() <= maxLength ? value : value.substring(0, maxLength);
    }
}
