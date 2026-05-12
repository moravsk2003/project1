package com.rustbuilder.ai.rl.supervisor;

import com.rustbuilder.ai.rl.RLRewardConfig;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

/**
 * LLM adapter that delegates review to an external command.
 *
 * <p>The command receives one observation JSON document on stdin and must print
 * one decision JSON document on stdout.
 */
public class ExternalCommandLlmSupervisor implements LlmSupervisor {
    private static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(90);

    private final List<String> command;
    private final Duration timeout;
    private final Supplier<RLRewardConfig> currentRewardConfigSupplier;
    private final Map<String, String> environmentOverrides;

    public ExternalCommandLlmSupervisor(String commandLine, RLRewardConfig currentRewardConfig) {
        this(splitCommandLine(commandLine), DEFAULT_TIMEOUT, () -> currentRewardConfig, Map.of());
    }

    public ExternalCommandLlmSupervisor(String commandLine, Supplier<RLRewardConfig> currentRewardConfigSupplier) {
        this(splitCommandLine(commandLine), DEFAULT_TIMEOUT, currentRewardConfigSupplier, Map.of());
    }

    public ExternalCommandLlmSupervisor(String commandLine,
                                        Supplier<RLRewardConfig> currentRewardConfigSupplier,
                                        Map<String, String> environmentOverrides) {
        this(splitCommandLine(commandLine), DEFAULT_TIMEOUT, currentRewardConfigSupplier, environmentOverrides);
    }

    public ExternalCommandLlmSupervisor(List<String> command, Duration timeout, RLRewardConfig currentRewardConfig) {
        this(command, timeout, () -> currentRewardConfig, Map.of());
    }

    public ExternalCommandLlmSupervisor(List<String> command, Duration timeout, Supplier<RLRewardConfig> currentRewardConfigSupplier) {
        this(command, timeout, currentRewardConfigSupplier, Map.of());
    }

    public ExternalCommandLlmSupervisor(List<String> command,
                                        Duration timeout,
                                        Supplier<RLRewardConfig> currentRewardConfigSupplier,
                                        Map<String, String> environmentOverrides) {
        if (command == null || command.isEmpty()) {
            throw new IllegalArgumentException("LLM supervisor command must not be empty");
        }
        this.command = List.copyOf(command);
        this.timeout = timeout != null ? timeout : DEFAULT_TIMEOUT;
        this.currentRewardConfigSupplier = currentRewardConfigSupplier != null
            ? currentRewardConfigSupplier
            : RLRewardConfig::createDefault;
        this.environmentOverrides = environmentOverrides != null
            ? Map.copyOf(environmentOverrides)
            : Map.of();
    }

    @Override
    public SupervisorDecision review(SupervisorObservation observation) throws Exception {
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        for (Map.Entry<String, String> entry : environmentOverrides.entrySet()) {
            if (entry.getKey() != null && entry.getValue() != null && !entry.getValue().isBlank()) {
                processBuilder.environment().put(entry.getKey(), entry.getValue());
            }
        }
        Process process = processBuilder.start();
        byte[] payload = SupervisorJson.observationToJson(observation).getBytes(StandardCharsets.UTF_8);
        process.getOutputStream().write(payload);
        process.getOutputStream().close();

        boolean finished = process.waitFor(timeout.toMillis(), TimeUnit.MILLISECONDS);
        if (!finished) {
            process.destroyForcibly();
            return SupervisorDecision.keepGoing("External LLM supervisor timed out.");
        }

        String stdout = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
        String stderr = new String(process.getErrorStream().readAllBytes(), StandardCharsets.UTF_8).trim();
        if (process.exitValue() != 0) {
            return SupervisorDecision.keepGoing("External LLM supervisor failed: " + stderr);
        }
        if (stdout.isBlank()) {
            return SupervisorDecision.keepGoing("External LLM supervisor returned no decision.");
        }

        RLRewardConfig currentRewardConfig = currentRewardConfigSupplier.get();
        return SupervisorJson.decisionFromJson(stdout, currentRewardConfig);
    }

    private static List<String> splitCommandLine(String commandLine) {
        if (commandLine == null || commandLine.isBlank()) {
            throw new IllegalArgumentException("LLM supervisor command must not be empty");
        }

        List<String> parts = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < commandLine.length(); i++) {
            char ch = commandLine.charAt(i);
            if (ch == '"') {
                inQuotes = !inQuotes;
                continue;
            }
            if (Character.isWhitespace(ch) && !inQuotes) {
                if (current.length() > 0) {
                    parts.add(current.toString());
                    current.setLength(0);
                }
                continue;
            }
            current.append(ch);
        }
        if (current.length() > 0) {
            parts.add(current.toString());
        }
        if (parts.isEmpty()) {
            throw new IllegalArgumentException("LLM supervisor command must not be empty");
        }
        return parts;
    }
}
