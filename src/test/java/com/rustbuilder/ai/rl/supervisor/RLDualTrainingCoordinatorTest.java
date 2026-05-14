package com.rustbuilder.ai.rl.supervisor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import com.rustbuilder.ai.rl.RLTrainingConfig;

import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

class RLDualTrainingCoordinatorTest {

    @Test
    void baselineConfigPreservesBranchOutputDirectoryWhenSupervisorIsDisabled() {
        Path branchOutputDirectory = Paths.get("models_rl", "codex_model", "branches", "codex_model_baseline_123");
        RLTrainingConfig config = new RLTrainingConfig(
            "codex_model_baseline_123",
            25,
            100,
            1.0,
            0.8,
            1.2,
            1.0,
            0.5,
            1,
            LlmSupervisorConfig.enabledDefault("baseline"),
            30_000L,
            true,
            branchOutputDirectory);

        RLTrainingConfig withoutSupervisor = new RLDualTrainingCoordinator().withoutSupervisor(config);

        assertFalse(withoutSupervisor.getSupervisorConfig().isEnabled());
        assertEquals("baseline", withoutSupervisor.getSupervisorConfig().getBranchId());
        assertEquals(branchOutputDirectory, withoutSupervisor.getOutputDirectory());
    }
}
