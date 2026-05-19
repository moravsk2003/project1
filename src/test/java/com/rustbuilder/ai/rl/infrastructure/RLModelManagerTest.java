package com.rustbuilder.ai.rl.infrastructure;


import com.rustbuilder.ai.rl.application.RLTrainingLogger;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class RLModelManagerTest {
    private static final String MODEL_NAME = "codex_legacy_flat_model_test";
    private static final String LINEAGE_OWNER = "codex_lineage_owner_test";
    private static final String LINEAGE_BRANCH = LINEAGE_OWNER + "_candidate_123";

    @AfterEach
    void cleanup() throws Exception {
        deleteIfExists(RLModelManager.getModelsDir().resolve(MODEL_NAME));
        deleteIfExists(RLModelManager.getModelsDir().resolve(LINEAGE_OWNER));
        deleteIfExists(RLModelManager.getModelsDir().resolve(LINEAGE_BRANCH));
        Files.deleteIfExists(RLModelManager.getModelsDir().resolve(MODEL_NAME + ".rmeta"));
        Files.deleteIfExists(RLModelManager.getModelsDir().resolve(MODEL_NAME + ".rnet"));
    }

    @Test
    void normalizesLegacyFlatFilesIntoMainAndLlmDirectories() throws Exception {
        Path modelDir = RLModelManager.getModelsDir().resolve(MODEL_NAME);
        Files.createDirectories(modelDir);
        Path legacyMeta = modelDir.resolve(MODEL_NAME + ".rmeta");
        Path legacyTrainingLog = modelDir.resolve(MODEL_NAME + "_multi_discrete_training.csv");
        Path legacySupervisorLog = modelDir.resolve(MODEL_NAME + "_supervisor_decisions.jsonl");
        Path legacySupervisorDebugLog = modelDir.resolve(MODEL_NAME + "_supervisor_debug.jsonl");
        Path legacyBranchExperimentLog = modelDir.resolve(MODEL_NAME + "_branch_experiments.csv");
        Path legacyTopLevelNet = RLModelManager.getModelsDir().resolve(MODEL_NAME + ".rnet");
        Files.writeString(legacyMeta, "metadata");
        Files.writeString(legacyTrainingLog, "training");
        Files.writeString(legacySupervisorLog, "supervisor");
        Files.writeString(legacySupervisorDebugLog, "debug");
        Files.writeString(legacyBranchExperimentLog, "branch");
        Files.writeString(legacyTopLevelNet, "network");

        assertTrue(RLModelManager.listModels().contains(MODEL_NAME));

        Path mainDir = RLModelManager.getModelMainDirectory(MODEL_NAME);
        Path llmDir = RLModelManager.getModelLlmDirectory(MODEL_NAME);

        assertTrue(Files.exists(mainDir.resolve(MODEL_NAME + ".rmeta")));
        assertTrue(Files.exists(mainDir.resolve(MODEL_NAME + ".rnet")));
        assertTrue(Files.exists(mainDir.resolve(MODEL_NAME + "_multi_discrete_training.csv")));
        assertTrue(Files.exists(llmDir.resolve(MODEL_NAME + "_supervisor_decisions.jsonl")));
        assertTrue(Files.exists(llmDir.resolve(MODEL_NAME + "_supervisor_debug.jsonl")));
        assertTrue(Files.exists(llmDir.resolve(MODEL_NAME + "_branch_experiments.csv")));
        assertFalse(Files.exists(legacyMeta));
        assertFalse(Files.exists(legacyTrainingLog));
        assertFalse(Files.exists(legacySupervisorLog));
        assertFalse(Files.exists(legacySupervisorDebugLog));
        assertFalse(Files.exists(legacyBranchExperimentLog));
        assertFalse(Files.exists(legacyTopLevelNet));
    }

    @Test
    void branchLookupDoesNotCreateTopLevelBranchDirectory() throws Exception {
        Path branchDir = RLModelManager.getBranchDirectory(LINEAGE_OWNER, LINEAGE_BRANCH);
        Path branchMetadata = branchDir.resolve(LINEAGE_BRANCH + ".rmeta");
        Path topLevelBranchDir = RLModelManager.getModelsDir().resolve(LINEAGE_BRANCH);
        Files.writeString(branchMetadata, "metadata");

        assertFalse(Files.exists(topLevelBranchDir));
        assertFalse(RLModelManager.listModels().contains(LINEAGE_BRANCH));
        assertTrue(RLModelManager.listBranchModels().contains(LINEAGE_BRANCH));
        assertFalse(Files.exists(topLevelBranchDir));

        Path found = RLModelManager.findExistingModelFile(LINEAGE_BRANCH, LINEAGE_BRANCH + ".rmeta");

        assertEquals(branchMetadata.normalize(), found.normalize());
        assertFalse(Files.exists(topLevelBranchDir));
    }

    @Test
    void missingStorageLookupDoesNotCreateTopLevelBranchDirectory() throws Exception {
        String missingBranch = LINEAGE_OWNER + "_candidate_999";
        Path topLevelBranchDir = RLModelManager.getModelsDir().resolve(missingBranch);

        Path found = RLModelManager.findModelStorageDirectory(missingBranch);

        assertEquals(topLevelBranchDir.resolve("main").normalize(), found.normalize());
        assertFalse(Files.exists(topLevelBranchDir));
    }

    @Test
    void rootOutputDirectoryRedirectsToModelMainDirectory() {
        Path normalized = RLModelManager.normalizeModelOutputDirectory(MODEL_NAME, RLModelManager.getModelsDir());

        assertEquals(RLModelManager.getModelsDir()
            .resolve(MODEL_NAME)
            .resolve("main")
            .normalize(), normalized.normalize());
    }

    @Test
    void generatedBranchDefaultOutputUsesOwnerBranchesDirectory() {
        Path normalized = RLModelManager.normalizeModelOutputDirectory(LINEAGE_BRANCH, null);

        assertEquals(RLModelManager.getModelsDir()
            .resolve(LINEAGE_OWNER)
            .resolve("branches")
            .resolve(LINEAGE_BRANCH)
            .normalize(), normalized.normalize());
        assertFalse(Files.exists(RLModelManager.getModelsDir().resolve(LINEAGE_BRANCH)));
    }

    @Test
    void loggerRedirectsRootOutputDirectoryToModelMainDirectory() {
        RLTrainingLogger logger = new RLTrainingLogger();

        logger.setLogFile(MODEL_NAME, true, RLModelManager.getModelsDir());

        assertFalse(Files.exists(RLModelManager.getModelsDir()
            .resolve(MODEL_NAME + "_multi_discrete_training.csv")));
        assertTrue(Files.exists(RLModelManager.getModelsDir()
            .resolve(MODEL_NAME)
            .resolve("main")
            .resolve(MODEL_NAME + "_multi_discrete_training.csv")));
    }

    @Test
    void listModelsCleansEmptyGeneratedTopLevelBranchDirectory() throws Exception {
        String generatedBranch = LINEAGE_OWNER + "_candidate_777";
        Path topLevelBranchDir = RLModelManager.getModelsDir().resolve(generatedBranch);
        Files.createDirectories(topLevelBranchDir.resolve("main"));

        RLModelManager.listModels();

        assertFalse(Files.exists(topLevelBranchDir));
        assertFalse(Files.exists(RLModelManager.getModelsDir()
            .resolve(LINEAGE_OWNER)
            .resolve("branches")
            .resolve(generatedBranch)));
    }

    @Test
    void listBranchModelsMigratesGeneratedTopLevelBranchFilesUnderOwner() throws Exception {
        Path topLevelBranchDir = RLModelManager.getModelsDir().resolve(LINEAGE_BRANCH);
        Files.createDirectories(topLevelBranchDir.resolve("main"));
        Files.writeString(topLevelBranchDir.resolve("main").resolve(LINEAGE_BRANCH + ".rmeta"), "metadata");
        Files.writeString(topLevelBranchDir.resolve("main").resolve(LINEAGE_BRANCH + "_episodes.csv"), "episodes");

        assertTrue(RLModelManager.listBranchModels().contains(LINEAGE_BRANCH));

        Path branchDir = RLModelManager.getModelsDir()
            .resolve(LINEAGE_OWNER)
            .resolve("branches")
            .resolve(LINEAGE_BRANCH);
        assertTrue(Files.exists(branchDir.resolve(LINEAGE_BRANCH + ".rmeta")));
        assertTrue(Files.exists(branchDir.resolve(LINEAGE_BRANCH + "_episodes.csv")));
        assertFalse(Files.exists(topLevelBranchDir));
    }

    @Test
    void prunesOldGeneratedBranchDirectories() throws Exception {
        Path oldBranch = generatedBranchDirectory("001");
        Path keptBranch = generatedBranchDirectory("002");
        Path newestBranch = generatedBranchDirectory("003");

        RLModelManager.pruneGeneratedBranchDirectories(LINEAGE_OWNER, 2);

        assertFalse(Files.exists(oldBranch));
        assertTrue(Files.exists(keptBranch));
        assertTrue(Files.exists(newestBranch));
    }

    @Test
    void promotedBranchLineageRewritesMainLogsWithContinuousEpisodeCounters() throws Exception {
        Path ownerMain = RLModelManager.getModelMainDirectory(LINEAGE_OWNER);
        Path branchDir = RLModelManager.getModelsDir()
            .resolve(LINEAGE_OWNER)
            .resolve("branches")
            .resolve(LINEAGE_BRANCH);
        Path inheritedMain = branchDir.resolve("inherited_main");
        Files.createDirectories(inheritedMain);
        Files.writeString(branchDir.resolve(LINEAGE_BRANCH + ".rmeta"), "metadata");

        String episodeHeader = "timestamp,run_id,epoch,episode,total_episode_count,value";
        Files.write(ownerMain.resolve(LINEAGE_OWNER + "_episodes.csv"), java.util.List.of(
            episodeHeader,
            "t0,main,1,1,1,old",
            "t1,main,1,2,2,old"));
        Files.write(inheritedMain.resolve(LINEAGE_OWNER + "_episodes.csv"), java.util.List.of(
            episodeHeader,
            "t0,main,1,1,1,old",
            "t1,main,1,2,2,old"));
        Files.write(branchDir.resolve(LINEAGE_BRANCH + "_episodes.csv"), java.util.List.of(
            episodeHeader,
            "t2,branch,1,1,1,new",
            "t3,branch,1,2,2,new"));

        String epochHeader = "timestamp,epoch,episodes,value";
        Files.write(ownerMain.resolve(LINEAGE_OWNER + "_multi_discrete_training.csv"), java.util.List.of(
            epochHeader,
            "t1,1,2,old"));
        Files.write(inheritedMain.resolve(LINEAGE_OWNER + "_multi_discrete_training.csv"), java.util.List.of(
            epochHeader,
            "t1,1,2,old"));
        Files.write(branchDir.resolve(LINEAGE_BRANCH + "_multi_discrete_training.csv"), java.util.List.of(
            epochHeader,
            "t2,1,2,new"));

        int totalEpisodes = RLModelManager.promoteBranchLineageToMain(LINEAGE_OWNER, LINEAGE_BRANCH);

        java.util.List<String> episodes = Files.readAllLines(ownerMain.resolve(LINEAGE_OWNER + "_episodes.csv"));
        assertEquals("3", episodes.get(3).split(",", -1)[4]);
        assertEquals("4", episodes.get(4).split(",", -1)[4]);
        assertTrue(Files.readString(ownerMain.resolve(LINEAGE_OWNER + "_multi_discrete_training.csv"))
            .contains("t2,2,2,new"));
        assertEquals(4, totalEpisodes);
    }

    private void deleteIfExists(Path path) throws IOException {
        if (!Files.exists(path)) {
            return;
        }
        try (Stream<Path> paths = Files.walk(path)) {
            for (Path item : paths.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(item);
            }
        }
    }

    private Path generatedBranchDirectory(String suffix) throws IOException {
        String branchName = LINEAGE_OWNER + "_candidate_" + suffix;
        Path branchDir = RLModelManager.getBranchDirectory(LINEAGE_OWNER, branchName);
        Files.writeString(branchDir.resolve(branchName + ".rmeta"), "metadata");
        return branchDir;
    }
}
