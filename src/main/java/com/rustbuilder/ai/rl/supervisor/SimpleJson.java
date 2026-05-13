package com.rustbuilder.ai.rl.supervisor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class SimpleJson {
    private SimpleJson() {
    }

    public static Object parse(String json) {
        return new Parser(json).parse();
    }

    public static String stringify(Object value) {
        if (value == null) {
            return "null";
        }
        if (value instanceof String) {
            return "\"" + escape((String) value) + "\"";
        }
        if (value instanceof Number) {
            double d = ((Number) value).doubleValue();
            if (Double.isNaN(d) || Double.isInfinite(d)) {
                return "0";
            }
            return String.format(Locale.US, "%s", value);
        }
        if (value instanceof Boolean) {
            return ((Boolean) value) ? "true" : "false";
        }
        if (value instanceof Map<?, ?> map) {
            StringBuilder sb = new StringBuilder();
            sb.append("{");
            boolean first = true;
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (!first) sb.append(",");
                first = false;
                sb.append(stringify(String.valueOf(entry.getKey())));
                sb.append(":");
                sb.append(stringify(entry.getValue()));
            }
            sb.append("}");
            return sb.toString();
        }
        if (value instanceof Iterable<?> iterable) {
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            boolean first = true;
            for (Object item : iterable) {
                if (!first) sb.append(",");
                first = false;
                sb.append(stringify(item));
            }
            sb.append("]");
            return sb.toString();
        }
        return stringify(String.valueOf(value));
    }

    private static String escape(String value) {
        return value
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\r", "\\r")
            .replace("\n", "\\n")
            .replace("\t", "\\t");
    }

    private static final class Parser {
        private final String input;
        private int pos;

        Parser(String input) {
            this.input = input != null ? input : "";
        }

        Object parse() {
            Object value = parseValue();
            skipWhitespace();
            if (pos != input.length()) {
                throw new IllegalArgumentException("Unexpected JSON token at " + pos);
            }
            return value;
        }

        private Object parseValue() {
            skipWhitespace();
            char ch = peek();
            if (ch == '{') return parseObject();
            if (ch == '[') return parseArray();
            if (ch == '"') return parseString();
            if (ch == 't') return parseLiteral("true", Boolean.TRUE);
            if (ch == 'f') return parseLiteral("false", Boolean.FALSE);
            if (ch == 'n') return parseLiteral("null", null);
            return parseNumber();
        }

        private Map<String, Object> parseObject() {
            expect('{');
            Map<String, Object> object = new LinkedHashMap<>();
            skipWhitespace();
            if (match('}')) return object;
            while (true) {
                String key = parseString();
                skipWhitespace();
                expect(':');
                object.put(key, parseValue());
                skipWhitespace();
                if (match('}')) return object;
                expect(',');
            }
        }

        private List<Object> parseArray() {
            expect('[');
            List<Object> list = new ArrayList<>();
            skipWhitespace();
            if (match(']')) return list;
            while (true) {
                list.add(parseValue());
                skipWhitespace();
                if (match(']')) return list;
                expect(',');
            }
        }

        private String parseString() {
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (pos < input.length()) {
                char ch = input.charAt(pos++);
                if (ch == '"') {
                    return sb.toString();
                }
                if (ch != '\\') {
                    sb.append(ch);
                    continue;
                }
                if (pos >= input.length()) {
                    throw new IllegalArgumentException("Unterminated JSON escape");
                }
                char esc = input.charAt(pos++);
                switch (esc) {
                    case '"' -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case '/' -> sb.append('/');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'u' -> sb.append(parseUnicode());
                    default -> throw new IllegalArgumentException("Unknown JSON escape: " + esc);
                }
            }
            throw new IllegalArgumentException("Unterminated JSON string");
        }

        private char parseUnicode() {
            if (pos + 4 > input.length()) {
                throw new IllegalArgumentException("Invalid unicode escape");
            }
            String hex = input.substring(pos, pos + 4);
            pos += 4;
            return (char) Integer.parseInt(hex, 16);
        }

        private Object parseLiteral(String literal, Object value) {
            if (!input.startsWith(literal, pos)) {
                throw new IllegalArgumentException("Expected " + literal + " at " + pos);
            }
            pos += literal.length();
            return value;
        }

        private Number parseNumber() {
            int start = pos;
            if (peek() == '-') pos++;
            while (Character.isDigit(peek())) pos++;
            if (peek() == '.') {
                pos++;
                while (Character.isDigit(peek())) pos++;
            }
            if (peek() == 'e' || peek() == 'E') {
                pos++;
                if (peek() == '+' || peek() == '-') pos++;
                while (Character.isDigit(peek())) pos++;
            }
            if (start == pos) {
                throw new IllegalArgumentException("Expected JSON value at " + pos);
            }
            return Double.parseDouble(input.substring(start, pos));
        }

        private void expect(char ch) {
            skipWhitespace();
            if (!match(ch)) {
                throw new IllegalArgumentException("Expected " + ch + " at " + pos);
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
            while (Character.isWhitespace(peek())) {
                pos++;
            }
        }
    }
}
