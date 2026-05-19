package com.rustbuilder.ai.rl.supervisor.provider;


import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.supervisor.ports.LlmSupervisor;
import com.rustbuilder.ai.rl.domain.RLRewardConfig;

import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Creates the supervisor provider while keeping the built-in Gemini path Java-native.
 */
public final class LlmSupervisorFactory {
    public static final String BUILTIN_GEMINI_COMMAND = "builtin:gemini";

    private LlmSupervisorFactory() {
    }

    public static LlmSupervisor create(LlmSupervisorConfig config,
                                       Supplier<RLRewardConfig> currentRewardConfigSupplier,
                                       Map<String, String> environmentOverrides) {
        if (config == null || !config.isEnabled()) {
            return new NoOpLlmSupervisor();
        }

        String command = config.getExternalCommand();
        if (command == null || command.isBlank()) {
            command = BUILTIN_GEMINI_COMMAND;
        }

        if (isNativeGeminiCommand(command)) {
            return new GeminiLlmSupervisor(resolveApiKey(config, environmentOverrides), config, currentRewardConfigSupplier);
        }

        return new ExternalCommandLlmSupervisor(command, currentRewardConfigSupplier, environmentOverrides);
    }

    public static boolean isNativeGeminiCommand(String command) {
        if (command == null || command.isBlank()) {
            return true;
        }
        String normalized = command.trim().toLowerCase(Locale.ROOT).replace('\\', '/');
        return BUILTIN_GEMINI_COMMAND.equals(normalized)
            || "gemini".equals(normalized)
            || "java:gemini".equals(normalized)
            || normalized.contains("scripts/llm_supervisor_gemini.ps1")
            || normalized.contains("scripts/llm_supervisor_gemini.py");
    }

    private static String resolveApiKey(LlmSupervisorConfig config, Map<String, String> environmentOverrides) {
        String configKey = config != null ? config.getApiKey() : "";
        if (configKey != null && !configKey.isBlank()) {
            return configKey.trim();
        }
        if (environmentOverrides == null || environmentOverrides.isEmpty()) {
            return "";
        }
        String geminiKey = environmentOverrides.get("GEMINI_API_KEY");
        if (geminiKey != null && !geminiKey.isBlank()) {
            return geminiKey.trim();
        }
        String googleKey = environmentOverrides.get("GOOGLE_API_KEY");
        if (googleKey != null && !googleKey.isBlank()) {
            return googleKey.trim();
        }
        String genericKey = environmentOverrides.get("LLM_API_KEY");
        return genericKey != null ? genericKey.trim() : "";
    }
}
