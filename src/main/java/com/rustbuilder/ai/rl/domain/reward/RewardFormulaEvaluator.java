package com.rustbuilder.ai.rl.domain.reward;

import java.util.Map;

/**
 * Tiny arithmetic-only expression evaluator for LLM-proposed reward formulas.
 */
public final class RewardFormulaEvaluator {
    private RewardFormulaEvaluator() {
    }

    public static double evaluate(String expression, Map<String, Double> variables) {
        if (expression == null || expression.isBlank()) {
            return 0.0;
        }
        try {
            double value = new Parser(expression, variables).parse();
            if (Double.isNaN(value) || Double.isInfinite(value)) {
                return 0.0;
            }
            return value;
        } catch (RuntimeException e) {
            return 0.0;
        }
    }

    private static final class Parser {
        private final String input;
        private final Map<String, Double> variables;
        private int pos = 0;

        Parser(String input, Map<String, Double> variables) {
            this.input = input;
            this.variables = variables;
        }

        double parse() {
            double value = parseExpression();
            skipWhitespace();
            if (pos != input.length()) {
                throw new IllegalArgumentException("Unexpected token at " + pos);
            }
            return value;
        }

        private double parseExpression() {
            double value = parseTerm();
            while (true) {
                skipWhitespace();
                if (match('+')) {
                    value += parseTerm();
                } else if (match('-')) {
                    value -= parseTerm();
                } else {
                    return value;
                }
            }
        }

        private double parseTerm() {
            double value = parseFactor();
            while (true) {
                skipWhitespace();
                if (match('*')) {
                    value *= parseFactor();
                } else if (match('/')) {
                    double divisor = parseFactor();
                    value = Math.abs(divisor) < 1e-9 ? 0.0 : value / divisor;
                } else {
                    return value;
                }
            }
        }

        private double parseFactor() {
            skipWhitespace();
            if (match('+')) {
                return parseFactor();
            }
            if (match('-')) {
                return -parseFactor();
            }
            if (match('(')) {
                double value = parseExpression();
                expect(')');
                return value;
            }
            if (isIdentifierStart(peek())) {
                return parseIdentifierOrFunction();
            }
            return parseNumber();
        }

        private double parseIdentifierOrFunction() {
            String name = parseIdentifier();
            skipWhitespace();
            if (!match('(')) {
                return variables != null ? variables.getOrDefault(name, 0.0) : 0.0;
            }

            double first = parseExpression();
            skipWhitespace();
            if ("abs".equals(name)) {
                expect(')');
                return Math.abs(first);
            }
            if ("sqrt".equals(name)) {
                expect(')');
                return Math.sqrt(Math.max(0.0, first));
            }

            expect(',');
            double second = parseExpression();
            skipWhitespace();
            if ("min".equals(name)) {
                expect(')');
                return Math.min(first, second);
            }
            if ("max".equals(name)) {
                expect(')');
                return Math.max(first, second);
            }

            expect(',');
            double third = parseExpression();
            skipWhitespace();
            if ("clamp".equals(name)) {
                expect(')');
                return Math.max(second, Math.min(third, first));
            }

            throw new IllegalArgumentException("Unknown function: " + name);
        }

        private String parseIdentifier() {
            int start = pos;
            while (pos < input.length()) {
                char ch = input.charAt(pos);
                if (!Character.isLetterOrDigit(ch) && ch != '_') {
                    break;
                }
                pos++;
            }
            return input.substring(start, pos);
        }

        private double parseNumber() {
            int start = pos;
            boolean hasDigit = false;
            while (pos < input.length()) {
                char ch = input.charAt(pos);
                if (Character.isDigit(ch)) {
                    hasDigit = true;
                    pos++;
                } else if (ch == '.') {
                    pos++;
                } else {
                    break;
                }
            }
            if (!hasDigit) {
                throw new IllegalArgumentException("Number expected at " + start);
            }
            return Double.parseDouble(input.substring(start, pos));
        }

        private void expect(char expected) {
            skipWhitespace();
            if (!match(expected)) {
                throw new IllegalArgumentException("Expected " + expected + " at " + pos);
            }
        }

        private boolean match(char ch) {
            if (peek() == ch) {
                pos++;
                return true;
            }
            return false;
        }

        private char peek() {
            return pos < input.length() ? input.charAt(pos) : '\0';
        }

        private void skipWhitespace() {
            while (pos < input.length() && Character.isWhitespace(input.charAt(pos))) {
                pos++;
            }
        }

        private boolean isIdentifierStart(char ch) {
            return Character.isLetter(ch) || ch == '_';
        }
    }
}
