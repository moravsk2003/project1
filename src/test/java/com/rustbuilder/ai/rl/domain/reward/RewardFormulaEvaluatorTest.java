package com.rustbuilder.ai.rl.domain.reward;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Map;
import org.junit.jupiter.api.Test;

class RewardFormulaEvaluatorTest {

    @Test
    void evaluatesArithmeticVariablesAndClamp() {
        double value = RewardFormulaEvaluator.evaluate(
            "clamp(socket_connections * 0.2 + stability * 0.01, -1, 1)",
            Map.of("socket_connections", 3.0, "stability", 50.0));

        assertEquals(1.0, value, 0.0);
    }

    @Test
    void invalidExpressionFallsBackToZero() {
        double value = RewardFormulaEvaluator.evaluate("java.lang.System.exit(0)", Map.of());

        assertEquals(0.0, value, 0.0);
    }

    @Test
    void formulaSetEvaluatesOnlyMatchingScope() {
        RewardFormulaSet set = new RewardFormulaSet();
        set.addTerm(new RewardFormulaTerm("step_term", RewardFormulaScope.STEP, "growth * 0.5"));
        set.addTerm(new RewardFormulaTerm("final_term", RewardFormulaScope.FINAL, "final_score * 2"));

        assertEquals(1.5, set.evaluate(RewardFormulaScope.STEP, Map.of("growth", 3.0)), 0.0);
        assertEquals(0.8, set.evaluate(RewardFormulaScope.FINAL, Map.of("final_score", 0.4)), 0.0);
    }
}
