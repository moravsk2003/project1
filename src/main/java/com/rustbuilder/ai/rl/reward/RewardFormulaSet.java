package com.rustbuilder.ai.rl.reward;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class RewardFormulaSet implements Serializable, Cloneable {
    private static final long serialVersionUID = 1L;
    private static final int MAX_TERMS = 32;

    private List<RewardFormulaTerm> terms = new ArrayList<>();

    public void addTerm(RewardFormulaTerm term) {
        if (term == null || terms.size() >= MAX_TERMS) {
            return;
        }
        terms.add(term.clone());
    }

    public List<RewardFormulaTerm> getTerms() {
        List<RewardFormulaTerm> copy = new ArrayList<>(terms.size());
        for (RewardFormulaTerm term : terms) {
            copy.add(term.clone());
        }
        return Collections.unmodifiableList(copy);
    }

    public double evaluate(RewardFormulaScope scope, Map<String, Double> variables) {
        double total = 0.0;
        for (RewardFormulaTerm term : terms) {
            if (term.getScope() == scope) {
                total += term.evaluate(variables);
            }
        }
        return total;
    }

    public boolean isEmpty() {
        return terms.isEmpty();
    }

    @Override
    public RewardFormulaSet clone() {
        try {
            RewardFormulaSet clone = (RewardFormulaSet) super.clone();
            clone.terms = new ArrayList<>(terms.size());
            for (RewardFormulaTerm term : terms) {
                clone.terms.add(term.clone());
            }
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError(e);
        }
    }
}
