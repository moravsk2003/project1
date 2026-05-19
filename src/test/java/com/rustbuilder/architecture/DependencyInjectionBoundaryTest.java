package com.rustbuilder.architecture;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class DependencyInjectionBoundaryTest {

    @Test
    void gameControllerExposesConstructorInjectionForRuntimeServices() throws IOException {
        String source = read("src/main/java/com/rustbuilder/controller/GameController.java");

        assertTrue(source.contains("SnapResolver snappingService"));
        assertTrue(source.contains("HouseEvaluationService houseEvaluator"));
        assertFalse(source.contains("private final HouseEvaluator houseEvaluator = new HouseEvaluator()"));
    }

    @Test
    void rlTrainingServiceDelegatesLearningComponentCreationToFactories() throws IOException {
        String source = read("src/main/java/com/rustbuilder/ai/rl/application/RLTrainingService.java");

        assertTrue(source.contains("StateEncoderFactory stateEncoderFactory"));
        assertTrue(source.contains("TrainingAgentFactory trainingAgentFactory"));
        assertTrue(source.contains("EpisodeRunnerFactory episodeRunnerFactory"));
        assertFalse(source.contains("new HybridV3StateEncoder("));
        assertFalse(source.contains("new BucketedVoxelV2StateEncoder("));
        assertFalse(source.contains("new VoxelV1StateEncoder("));
        assertFalse(source.contains("new MultiDiscreteDQNAgent("));
        assertFalse(source.contains("new MultiDiscreteExperienceReplay("));
        assertFalse(source.contains("new EpisodeEvaluator("));
        assertFalse(source.contains("new EpisodeRunner("));
    }

    @Test
    void geneticAlgorithmServiceAcceptsInjectedEvaluatorAndGridFactory() throws IOException {
        String source = read("src/main/java/com/rustbuilder/ai/ea/application/GeneticAlgorithmService.java");

        assertTrue(source.contains("HouseEvaluationService evaluator"));
        assertTrue(source.contains("GridModelFactory gridModelFactory"));
        assertFalse(source.contains("new HouseEvaluator("));
    }

    @Test
    void generatorDialogUsesApplicationComponentForGeneticAlgorithmService() throws IOException {
        String source = read("src/main/java/com/rustbuilder/ui/GeneratorDialog.java");

        assertTrue(source.contains("AppComponent appComponent"));
        assertTrue(source.contains("createGeneticAlgorithmService()"));
        assertFalse(source.contains("new GeneticAlgorithmService("));
    }

    @Test
    void appComponentIsBackedByGuiceInjector() throws IOException {
        String pom = read("pom.xml");
        String module = read("src/main/java/com/rustbuilder/di/AppModule.java");
        String component = read("src/main/java/com/rustbuilder/di/AppComponent.java");

        assertTrue(pom.contains("<artifactId>guice</artifactId>"));
        assertTrue(module.contains("extends AbstractModule"));
        assertTrue(component.contains("Guice.createInjector(new AppModule())"));
    }

    private String read(String path) throws IOException {
        return Files.readString(Path.of(path), StandardCharsets.UTF_8);
    }
}
