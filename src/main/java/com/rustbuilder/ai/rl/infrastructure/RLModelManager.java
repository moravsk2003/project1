package com.rustbuilder.ai.rl.infrastructure;


import com.rustbuilder.ai.rl.application.RLTrainingService;
import com.rustbuilder.ai.rl.domain.RLRewardConfig;
import com.rustbuilder.ai.rl.environment.spec.ActionSpaceSpec;
import com.rustbuilder.ai.rl.environment.spec.EncodingRuntimeConfig;
import com.rustbuilder.ai.rl.environment.spec.GridSpec;
import com.rustbuilder.ai.rl.environment.spec.StateEncodingSpec;
import com.rustbuilder.ai.rl.environment.spec.TileIndexingMode;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.regex.Pattern;

import com.rustbuilder.ai.rl.supervisor.config.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.policy.multidiscrete.MultiDiscreteActionSpace;




/**
 * Manages RL model persistence — save/load neural networks and parameters.
 */
public class RLModelManager {

    private static final String MODELS_DIR = "models_rl";
    private static final String MAIN_DIR = "main";
    private static final String LLM_DIR = "llm";
    private static final String BRANCHES_DIR = "branches";
    private static final int DEFAULT_MAX_GENERATED_BRANCH_DIRECTORIES =
        Integer.getInteger("rustbuilder.rl.maxBranchDirs", 12);
    private static final Pattern GENERATED_BRANCH_MODEL_PATTERN =
        Pattern.compile(".*_(baseline|candidate|before_jump)_\\d+$");

    /**
     * Serializable snapshot of training state.
     */
    public static class RLModel implements Serializable {
        private static final long serialVersionUID = 1L;

        public final String name;
        public final int episodesTrained;
        public final double bestScore;
        public final double epsilon;
        public final double logisticsWeight;
        public final double costWeight;
        public final double raidWeight;
        public final double workingAreaWeight;
        public final double safeZoneWeight;
        public RLRewardConfig rewardConfig; 
        public LlmSupervisorConfig supervisorConfig;
        
        public String bestBaseRewardJson;
        public String bestBaseEvalJson;

        // Compatibility metadata
        public String stateEncoderName = "Voxel";
        public String stateEncoderVersion = "v1";
        public int voxelChannels = 11;
        public int gridWidth = 8;
        public int gridHeight = 8;
        public int gridFloors = 8;
        public int actionTypeCount = 11;
        public int actionFloorCount = 8;
        public int actionTileCount = 64;
        public int actionRotationCount = MultiDiscreteActionSpace.ROTATION_COUNT;
        public int actionAimCount = MultiDiscreteActionSpace.AIM_SECTOR_COUNT;
        public String tileIndexingMode = "LEGACY_64";
        public boolean hasGlobalVector = false;
        public int globalFeatureCount = 0;
        public boolean hasObjectTable = false;
        public boolean hasGraphState = false;
        public boolean use2dCnn = false;
        public double epsilonDecayRate = 0.9995;
        public double minEpsilon = 0.05;

        public RLModel(String name, int episodesTrained,
                       double bestScore, double epsilon,
                       double logisticsWeight, double costWeight, double raidWeight, double workingAreaWeight,
                       double safeZoneWeight,
                       RLRewardConfig rewardConfig) {
            this.name = name;
            this.episodesTrained = episodesTrained;
            this.bestScore = bestScore;
            this.epsilon = epsilon;
            this.logisticsWeight = logisticsWeight;
            this.costWeight = costWeight;
            this.raidWeight = raidWeight;
            this.workingAreaWeight = workingAreaWeight;
            this.safeZoneWeight = safeZoneWeight;
            this.rewardConfig = rewardConfig;
        }

        @Override
        public String toString() {
            return String.format("%s (Ep %d, Best: %.3f)", name, episodesTrained, bestScore);
        }
    }

    public static Path getModelsDir() {
        Path dir = Paths.get(MODELS_DIR);
        try {
            Files.createDirectories(dir);
        } catch (IOException e) {
        }
        return dir;
    }

    public static Path getModelDirectory(String modelName) {
        if (isGeneratedBranchModelName(safeModelDirectoryName(modelName))) {
            normalizeLegacyModelDirectory(modelName);
            return ensureDirectory(defaultOutputDirectoryPath(modelName));
        }
        Path dir = getModelDirectoryPath(modelName);
        try {
            Files.createDirectories(dir);
        } catch (IOException e) {
        }
        normalizeLegacyModelDirectory(modelName);
        return dir;
    }

    public static Path getModelMainDirectory(String modelName) {
        if (isGeneratedBranchModelName(safeModelDirectoryName(modelName))) {
            normalizeLegacyModelDirectory(modelName);
            return ensureDirectory(defaultOutputDirectoryPath(modelName));
        }
        Path dir = ensureDirectory(getModelDirectoryPath(modelName).resolve(MAIN_DIR));
        normalizeLegacyModelDirectory(modelName);
        return dir;
    }

    public static Path getModelLlmDirectory(String modelName) {
        if (isGeneratedBranchModelName(safeModelDirectoryName(modelName))) {
            normalizeLegacyModelDirectory(modelName);
            return ensureDirectory(defaultOutputDirectoryPath(modelName).resolve(LLM_DIR));
        }
        Path dir = ensureDirectory(getModelDirectoryPath(modelName).resolve(LLM_DIR));
        normalizeLegacyModelDirectory(modelName);
        return dir;
    }

    public static Path getModelBranchesDirectory(String modelName) {
        return ensureDirectory(getModelDirectoryPath(modelName).resolve(BRANCHES_DIR));
    }

    public static Path getBranchDirectory(String ownerModelName, String branchModelName) {
        return ensureDirectory(getBranchDirectoryPath(ownerModelName, branchModelName));
    }

    public static Path getBranchDirectoryPath(String ownerModelName, String branchModelName) {
        return getModelDirectoryPath(ownerModelName)
            .resolve(BRANCHES_DIR)
            .resolve(safeModelDirectoryName(branchModelName));
    }

    public static Path normalizeModelOutputDirectory(String modelName, Path outputDirectory) {
        Path defaultDir = defaultOutputDirectoryPath(modelName);
        if (outputDirectory == null) {
            return ensureDirectory(defaultDir);
        }

        Path modelsDir = getModelsDir().toAbsolutePath().normalize();
        Path output = outputDirectory.toAbsolutePath().normalize();
        Path topLevelModelDir = getModelDirectoryPath(modelName).toAbsolutePath().normalize();
        Path topLevelModelMainDir = topLevelModelDir.resolve(MAIN_DIR).normalize();
        if (output.equals(modelsDir) || output.equals(topLevelModelDir) || output.equals(topLevelModelMainDir)) {
            return ensureDirectory(defaultDir);
        }
        return ensureDirectory(outputDirectory);
    }

    private static Path ensureDirectory(Path dir) {
        try {
            Files.createDirectories(dir);
        } catch (IOException e) {
        }
        return dir;
    }

    public static Path resolveModelFile(String modelName, String fileName) {
        return normalizeModelOutputDirectory(modelName, null).resolve(fileName);
    }

    public static Path findExistingModelFile(String modelName, String fileName) {
        normalizeLegacyModelDirectory(modelName);
        Path root = getModelDirectoryPath(modelName);
        Path defaultOutputDir = defaultOutputDirectoryPath(modelName);
        Path[] candidates = new Path[] {
            defaultOutputDir.resolve(fileName),
            root.resolve(MAIN_DIR).resolve(fileName),
            root.resolve(LLM_DIR).resolve(fileName),
            root.resolve(fileName),
            getModelsDir().resolve(fileName)
        };
        for (Path candidate : candidates) {
            if (Files.exists(candidate)) {
                return candidate;
            }
        }
        try (Stream<Path> files = Files.walk(getModelsDir(), 5)) {
            return files
                .filter(Files::isRegularFile)
                .filter(p -> p.getFileName().toString().equals(fileName))
                .findFirst()
                .orElse(defaultOutputDir.resolve(fileName));
        } catch (IOException e) {
            return defaultOutputDir.resolve(fileName);
        }
    }

    private static Path getModelDirectoryPath(String modelName) {
        return getModelsDir().resolve(safeModelDirectoryName(modelName));
    }

    private static Path defaultOutputDirectoryPath(String modelName) {
        String safeModelName = safeModelDirectoryName(modelName);
        String ownerName = generatedBranchOwnerName(safeModelName);
        if (!ownerName.isBlank()) {
            return getBranchDirectoryPath(ownerName, safeModelName);
        }
        return getModelDirectoryPath(safeModelName).resolve(MAIN_DIR);
    }

    private static String generatedBranchOwnerName(String modelName) {
        String name = modelName != null ? modelName.trim() : "";
        if (!isGeneratedBranchModelName(name)) {
            return "";
        }
        return name.replaceFirst("_(baseline|candidate|before_jump)_\\d+$", "").trim();
    }

    private static void normalizeLegacyModelDirectory(String modelName) {
        String safeModelName = safeModelDirectoryName(modelName);
        if (isGeneratedBranchModelName(safeModelName)) {
            normalizeGeneratedBranchDirectory(safeModelName);
            return;
        }
        Path modelsDir = getModelsDir();
        Path modelDir = modelsDir.resolve(safeModelName);
        Path mainDir = modelDir.resolve(MAIN_DIR);
        Path llmDir = modelDir.resolve(LLM_DIR);
        Path legacyTopLevelMeta = modelsDir.resolve(safeModelName + ".rmeta");
        Path legacyTopLevelNet = modelsDir.resolve(safeModelName + ".rnet");
        if (!Files.exists(modelDir)
                && !Files.isRegularFile(legacyTopLevelMeta)
                && !Files.isRegularFile(legacyTopLevelNet)) {
            return;
        }
        try {
            Files.createDirectories(modelDir);
            moveLegacyRootFile(legacyTopLevelMeta, mainDir);
            moveLegacyRootFile(legacyTopLevelNet, mainDir);
            try (Stream<Path> files = Files.list(modelDir)) {
                files
                    .filter(Files::isRegularFile)
                    .forEach(path -> moveLegacyRootFile(path, legacyDestinationDir(path, mainDir, llmDir)));
            }
        } catch (IOException e) {
        }
    }

    private static void normalizeGeneratedBranchDirectory(String safeBranchModelName) {
        String ownerName = generatedBranchOwnerName(safeBranchModelName);
        if (ownerName.isBlank()) {
            return;
        }

        Path modelsDir = getModelsDir();
        Path legacyBranchDir = modelsDir.resolve(safeBranchModelName);
        Path branchDir = getBranchDirectoryPath(ownerName, safeBranchModelName);
        Path legacyTopLevelMeta = modelsDir.resolve(safeBranchModelName + ".rmeta");
        Path legacyTopLevelNet = modelsDir.resolve(safeBranchModelName + ".rnet");
        boolean hasTopLevelMeta = Files.isRegularFile(legacyTopLevelMeta);
        boolean hasTopLevelNet = Files.isRegularFile(legacyTopLevelNet);
        if (!Files.exists(legacyBranchDir) && !hasTopLevelMeta && !hasTopLevelNet) {
            return;
        }

        try {
            List<Path> files = List.of();
            if (Files.isDirectory(legacyBranchDir)) {
                try (Stream<Path> paths = Files.walk(legacyBranchDir)) {
                    files = paths
                        .filter(Files::isRegularFile)
                        .collect(Collectors.toList());
                }
            }
            if (files.isEmpty() && !hasTopLevelMeta && !hasTopLevelNet) {
                if (Files.isDirectory(legacyBranchDir) && isDirectoryTreeEmpty(legacyBranchDir)) {
                    deleteDirectoryIfExists(legacyBranchDir);
                }
                return;
            }

            Files.createDirectories(branchDir);
            moveLegacyRootFile(legacyTopLevelMeta, branchDir);
            moveLegacyRootFile(legacyTopLevelNet, branchDir);
            if (Files.isDirectory(legacyBranchDir)) {
                for (Path file : files) {
                    moveGeneratedBranchFile(legacyBranchDir, file, branchDir);
                }
                if (isDirectoryTreeEmpty(legacyBranchDir)) {
                    deleteDirectoryIfExists(legacyBranchDir);
                }
            }
        } catch (IOException e) {
        }
    }

    private static void moveGeneratedBranchFile(Path legacyBranchDir, Path source, Path branchDir) {
        if (legacyBranchDir == null || source == null || branchDir == null || !Files.isRegularFile(source)) {
            return;
        }
        Path relative = legacyBranchDir.relativize(source);
        Path destinationRelative = stripLeadingPathName(relative, MAIN_DIR);
        Path target = branchDir.resolve(destinationRelative);
        try {
            Files.createDirectories(target.getParent());
            if (!Files.exists(target)) {
                Files.move(source, target);
            }
        } catch (IOException e) {
        }
    }

    private static Path stripLeadingPathName(Path path, String leadingName) {
        if (path == null || path.getNameCount() == 0 || leadingName == null) {
            return path;
        }
        if (!leadingName.equals(path.getName(0).toString()) || path.getNameCount() == 1) {
            return path;
        }
        return path.subpath(1, path.getNameCount());
    }

    private static Path legacyDestinationDir(Path path, Path mainDir, Path llmDir) {
        String name = path.getFileName().toString();
        return name.contains("_supervisor_decisions")
            || name.contains("_supervisor_debug")
            || name.contains("_branch_experiments")
            ? llmDir
            : mainDir;
    }

    private static void moveLegacyRootFile(Path source, Path destinationDir) {
        if (source == null || destinationDir == null || !Files.isRegularFile(source)) {
            return;
        }
        try {
            Files.createDirectories(destinationDir);
            Path target = destinationDir.resolve(source.getFileName());
            if (!Files.exists(target)) {
                Files.move(source, target);
            }
        } catch (IOException e) {
        }
    }

    private static String safeModelDirectoryName(String modelName) {
        String name = modelName == null ? "" : modelName.trim();
        if (name.isEmpty()) {
            return "unnamed_model";
        }
        return name.replaceAll("[\\\\/:*?\"<>|]+", "_");
    }

    public static void saveModel(RLModel model, RLTrainingService rlService) throws IOException {
        saveModel(model, rlService, null);
    }

    public static void saveModel(RLModel model, RLTrainingService rlService, Path outputDirectory) throws IOException {
        Path dir = normalizeModelOutputDirectory(model.name, outputDirectory);
        Path file = dir.resolve(model.name + ".rmeta");
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(Files.newOutputStream(file)))) {
            oos.writeObject(model);
        }
        Path netFile = dir.resolve(model.name + ".rnet");
        rlService.getMultiDiscreteAgent().save(netFile.toString());
    }

    public static RLModel loadModel(String name, RLTrainingService rlService) throws IOException, ClassNotFoundException {
        RLModel model = loadMetadata(name);
        loadNetworkWeights(name, rlService);
        return model;
    }

    public static RLModel loadMetadata(String name) throws IOException, ClassNotFoundException {
        Path file = findExistingModelFile(name, name + ".rmeta");
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(Files.newInputStream(file)))) {
            return (RLModel) ois.readObject();
        }
    }

    public static void loadNetworkWeights(String name, RLTrainingService rlService) throws IOException {
        Path netFile = findExistingModelFile(name, name + ".rnet");
        if (Files.exists(netFile)) {
            rlService.getMultiDiscreteAgent().load(netFile.toString());
        }
    }

    public static Path findModelStorageDirectory(String modelName) {
        Path metadata = findExistingModelFile(modelName, modelName + ".rmeta");
        if (Files.exists(metadata) && metadata.getParent() != null) {
            return metadata.getParent();
        }
        return getModelDirectoryPath(modelName).resolve(MAIN_DIR);
    }

    public static int promoteBranchLineageToMain(String ownerModelName, String branchModelName) throws IOException {
        String ownerName = safeFileModelName(ownerModelName);
        String branchName = safeFileModelName(branchModelName);
        if (ownerName.isBlank() || branchName.isBlank()) {
            return 0;
        }

        Path mainDir = getModelMainDirectory(ownerName);
        Path branchDir = findModelStorageDirectory(branchName);
        int totalEpisodes = 0;
        totalEpisodes = Math.max(totalEpisodes, mergeLineageCsv(mainDir, branchDir, ownerName, branchName,
            "_episodes.csv", 4, 2, -1));
        totalEpisodes = Math.max(totalEpisodes, mergeLineageCsv(mainDir, branchDir, ownerName, branchName,
            "_performance_tmp.csv", 4, 2, -1));
        mergeLineageCsv(mainDir, branchDir, ownerName, branchName,
            "_multi_discrete_training.csv", -1, 1, -1);
        mergeLineageCsv(mainDir, branchDir, ownerName, branchName,
            "_invalid_actions.csv", -1, 2, 3);
        return totalEpisodes;
    }

    private static int mergeLineageCsv(Path mainDir, Path branchDir, String ownerName, String branchName,
                                       String suffix, int totalEpisodeColumn, int epochColumn,
                                       int episodeColumn) throws IOException {
        Path inheritedPath = branchDir.resolve("inherited_main").resolve(ownerName + suffix);
        Path currentMainPath = mainDir.resolve(ownerName + suffix);
        Path branchPath = branchDir.resolve(branchName + suffix);
        Path targetPath = currentMainPath;

        List<String> inheritedLines = readCsvLines(Files.exists(inheritedPath) ? inheritedPath : currentMainPath);
        List<String> branchLines = readCsvLines(branchPath);
        if (inheritedLines.isEmpty() && branchLines.isEmpty()) {
            return 0;
        }

        String header = !inheritedLines.isEmpty() ? inheritedLines.get(0) : branchLines.get(0);
        List<String> inheritedRows = dataRows(inheritedLines);
        List<String> branchRows = dataRows(branchLines);
        int totalOffset = maxColumnValue(inheritedRows, totalEpisodeColumn);
        int epochOffset = maxColumnValue(inheritedRows, epochColumn);
        int episodeOffset = maxColumnValue(inheritedRows, episodeColumn);

        List<String> merged = new ArrayList<>();
        merged.add(header);
        merged.addAll(inheritedRows);
        for (String row : branchRows) {
            merged.add(adjustCsvCounters(row, totalEpisodeColumn, totalOffset,
                epochColumn, epochOffset, episodeColumn, episodeOffset));
        }

        Files.createDirectories(mainDir);
        Files.write(targetPath, merged, StandardCharsets.UTF_8);
        return Math.max(
            maxColumnValue(dataRows(merged), totalEpisodeColumn),
            maxColumnValue(dataRows(merged), episodeColumn));
    }

    private static List<String> readCsvLines(Path path) throws IOException {
        if (path == null || !Files.exists(path)) {
            return List.of();
        }
        return Files.readAllLines(path, StandardCharsets.UTF_8);
    }

    private static List<String> dataRows(List<String> lines) {
        if (lines == null || lines.size() <= 1) {
            return List.of();
        }
        return lines.subList(1, lines.size());
    }

    private static String adjustCsvCounters(String row, int totalEpisodeColumn, int totalOffset,
                                            int epochColumn, int epochOffset,
                                            int episodeColumn, int episodeOffset) {
        String[] fields = row.split(",", -1);
        addToColumn(fields, totalEpisodeColumn, totalOffset);
        addToColumn(fields, epochColumn, epochOffset);
        addToColumn(fields, episodeColumn, episodeOffset);
        return String.join(",", fields);
    }

    private static void addToColumn(String[] fields, int column, int offset) {
        if (column < 0 || offset <= 0 || fields == null || column >= fields.length) {
            return;
        }
        try {
            int value = Integer.parseInt(fields[column].trim());
            fields[column] = String.valueOf(value + offset);
        } catch (NumberFormatException e) {
        }
    }

    private static int maxColumnValue(List<String> rows, int column) {
        if (column < 0 || rows == null) {
            return 0;
        }
        int max = 0;
        for (String row : rows) {
            String[] fields = row.split(",", -1);
            if (column >= fields.length) {
                continue;
            }
            try {
                max = Math.max(max, Integer.parseInt(fields[column].trim()));
            } catch (NumberFormatException e) {
            }
        }
        return max;
    }

    private static String safeFileModelName(String modelName) {
        return modelName != null ? modelName.trim() : "";
    }

    public static List<String> listModels() {
        Path dir = getModelsDir();
        cleanupEmptyGeneratedTopLevelBranchDirectories();
        Set<String> names = new LinkedHashSet<>();
        try (Stream<Path> files = Files.walk(dir, 5)) {
            files
                .filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".rmeta"))
                .filter(p -> !isBranchPath(p))
                .map(RLModelManager::metadataModelName)
                .filter(name -> !isGeneratedBranchModelName(name))
                .sorted()
                .forEach(names::add);
        } catch (IOException e) {
        }

        for (String name : List.copyOf(names)) {
            normalizeLegacyModelDirectory(name);
        }
        return names.stream().sorted().collect(Collectors.toList());
    }

    public static List<String> listBranchModels() {
        Path dir = getModelsDir();
        cleanupEmptyGeneratedTopLevelBranchDirectories();
        Set<String> names = new LinkedHashSet<>();
        try (Stream<Path> files = Files.walk(dir, 5)) {
            files
                .filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".rmeta"))
                .filter(p -> isBranchPath(p) || isGeneratedBranchModelName(metadataModelName(p)))
                .map(RLModelManager::metadataModelName)
                .filter(name -> !name.isBlank())
                .sorted()
                .forEach(names::add);
        } catch (IOException e) {
        }
        return names.stream().sorted().collect(Collectors.toList());
    }

    public static void pruneGeneratedBranchDirectories(String ownerModelName) {
        pruneGeneratedBranchDirectories(ownerModelName, DEFAULT_MAX_GENERATED_BRANCH_DIRECTORIES);
    }

    public static void pruneGeneratedBranchDirectories(String ownerModelName, int maxDirectories) {
        Path branchesDir = getModelDirectoryPath(ownerModelName).resolve(BRANCHES_DIR);
        if (!Files.isDirectory(branchesDir)) {
            return;
        }
        int keep = Math.max(0, maxDirectories);
        try (Stream<Path> directories = Files.list(branchesDir)) {
            List<Path> generated = directories
                .filter(Files::isDirectory)
                .filter(path -> isGeneratedBranchModelName(path.getFileName().toString()))
                .sorted((a, b) -> Long.compare(generatedBranchSortKey(b), generatedBranchSortKey(a)))
                .collect(Collectors.toList());
            for (int i = keep; i < generated.size(); i++) {
                deleteDirectoryIfExists(generated.get(i));
            }
        } catch (IOException e) {
        }
    }

    private static String metadataModelName(Path path) {
        if (path == null || path.getFileName() == null) {
            return "";
        }
        String fileName = path.getFileName().toString();
        if (!fileName.endsWith(".rmeta")) {
            return "";
        }
        return fileName.substring(0, fileName.length() - ".rmeta".length());
    }

    private static boolean isGeneratedBranchModelName(String modelName) {
        return modelName != null && GENERATED_BRANCH_MODEL_PATTERN.matcher(modelName).matches();
    }

    private static boolean isBranchPath(Path path) {
        if (path == null) {
            return false;
        }
        for (Path part : path) {
            if (BRANCHES_DIR.equals(part.toString())) {
                return true;
            }
        }
        return false;
    }

    private static long generatedBranchSortKey(Path path) {
        if (path != null && path.getFileName() != null) {
            String name = path.getFileName().toString();
            int lastUnderscore = name.lastIndexOf('_');
            if (lastUnderscore >= 0 && lastUnderscore + 1 < name.length()) {
                try {
                    return Long.parseLong(name.substring(lastUnderscore + 1));
                } catch (NumberFormatException e) {
                }
            }
        }
        try {
            return Files.getLastModifiedTime(path).toMillis();
        } catch (IOException e) {
            return 0L;
        }
    }

    private static void cleanupEmptyGeneratedTopLevelBranchDirectories() {
        Path modelsDir = getModelsDir();
        try (Stream<Path> children = Files.list(modelsDir)) {
            children
                .filter(path -> isGeneratedBranchStoragePath(modelsDir, path))
                .forEach(path -> normalizeGeneratedBranchDirectory(generatedBranchStorageName(path)));
        } catch (IOException e) {
        }
    }

    private static boolean isGeneratedBranchStoragePath(Path modelsDir, Path path) {
        String name = generatedBranchStorageName(path);
        if (name.isBlank() || !isGeneratedBranchModelName(name)) {
            return false;
        }
        if (Files.isDirectory(path)) {
            return path.getParent() != null && path.getParent().equals(modelsDir);
        }
        return Files.isRegularFile(path)
            && (path.getFileName().toString().endsWith(".rmeta")
                || path.getFileName().toString().endsWith(".rnet"));
    }

    private static String generatedBranchStorageName(Path path) {
        if (path == null || path.getFileName() == null) {
            return "";
        }
        String fileName = path.getFileName().toString();
        if (fileName.endsWith(".rmeta") || fileName.endsWith(".rnet")) {
            int dot = fileName.lastIndexOf('.');
            return dot > 0 ? fileName.substring(0, dot) : fileName;
        }
        return fileName;
    }

    private static boolean isDirectoryTreeEmpty(Path dir) throws IOException {
        if (!Files.isDirectory(dir)) {
            return false;
        }
        try (Stream<Path> paths = Files.walk(dir)) {
            return paths.allMatch(path -> path.equals(dir) || Files.isDirectory(path));
        }
    }

    public static boolean deleteModel(String name) {
        Path meta = getModelsDir().resolve(name + ".rmeta");
        Path net = getModelsDir().resolve(name + ".rnet");
        Path modelDir = getModelDirectoryPath(name);
        try {
            boolean d1 = Files.deleteIfExists(meta);
            boolean d2 = Files.deleteIfExists(net);
            boolean d3 = deleteDirectoryIfExists(modelDir);
            return d1 || d2 || d3;
        } catch (IOException e) {
            return false;
        }
    }

    private static boolean deleteDirectoryIfExists(Path dir) throws IOException {
        if (!Files.exists(dir)) {
            return false;
        }

        try (Stream<Path> paths = Files.walk(dir)) {
            List<Path> ordered = paths
                .sorted(Comparator.reverseOrder())
                .collect(Collectors.toList());
            for (Path path : ordered) {
                Files.deleteIfExists(path);
            }
        }
        return true;
    }

    public static RLModel createSnapshot(String name, RLTrainingService rlService,
                                        double logW, double costW, double raidW, double workingAreaW, double safeZoneW) {
        RLRewardConfig cfg = rlService.getRewardConfig();
        if (cfg == null) {
            cfg = RLRewardConfig.createDefault();
        }

        RLModel model = new RLModel(name, rlService.getEpisodesTrained(),
                           rlService.getBestScore(), rlService.getEpsilon(),
                           logW, costW, raidW, workingAreaW, safeZoneW,
                           cfg.clone());
                           
        com.rustbuilder.ai.rl.environment.spec.EncodingRuntimeConfig config = rlService.getRuntimeConfig();
        model.stateEncoderName = config.stateEncodingSpec.encoderName;
        model.stateEncoderVersion = config.stateEncodingSpec.encoderVersion;
        model.voxelChannels = config.stateEncodingSpec.voxelChannels;
        model.gridWidth = config.gridSpec.width;
        model.gridHeight = config.gridSpec.height;
        model.gridFloors = config.gridSpec.floors;
        model.actionTypeCount = config.actionSpaceSpec.typeCount;
        model.actionFloorCount = config.actionSpaceSpec.floorCount;
        model.actionTileCount = config.actionSpaceSpec.tileCount;
        model.actionRotationCount = config.actionSpaceSpec.rotationCount;
        model.actionAimCount = config.actionSpaceSpec.aimCount;
        model.tileIndexingMode = config.actionSpaceSpec.tileIndexingMode.name();
        model.hasGlobalVector = config.stateEncodingSpec.hasGlobalVector;
        model.globalFeatureCount = config.stateEncodingSpec.globalFeatureCount;
        model.hasObjectTable = config.stateEncodingSpec.hasObjectTable;
        model.hasGraphState = config.stateEncodingSpec.hasGraphState;
        model.supervisorConfig = rlService.getSupervisorConfig();
        model.use2dCnn = rlService.isUse2dCnn();
        model.epsilonDecayRate = rlService.getEpsilonDecay();
        model.minEpsilon = rlService.getMinEpsilon();
        
        model.bestBaseEvalJson = com.rustbuilder.model.GridSerializer.toJson(rlService.getBestGridModelSnapshot());
        model.bestBaseRewardJson = com.rustbuilder.model.GridSerializer.toJson(rlService.getBestRewardGridModelSnapshot());
        
        return model;
    }

    public static void restoreFromModel(RLTrainingService rlService, RLModel model) {
        com.rustbuilder.ai.rl.environment.spec.EncodingRuntimeConfig currentConfig = rlService.getRuntimeConfig();
        
        // Strict model compatibility check
        if (!model.stateEncoderName.equals(currentConfig.stateEncodingSpec.encoderName) ||
            !model.stateEncoderVersion.equals(currentConfig.stateEncodingSpec.encoderVersion) ||
            model.voxelChannels != currentConfig.stateEncodingSpec.voxelChannels ||
            model.gridWidth != currentConfig.gridSpec.width ||
            model.gridHeight != currentConfig.gridSpec.height ||
            model.gridFloors != currentConfig.gridSpec.floors ||
            model.actionTypeCount != currentConfig.actionSpaceSpec.typeCount ||
            model.actionFloorCount != currentConfig.actionSpaceSpec.floorCount ||
            model.actionTileCount != currentConfig.actionSpaceSpec.tileCount ||
            model.actionRotationCount != currentConfig.actionSpaceSpec.rotationCount ||
            model.actionAimCount != currentConfig.actionSpaceSpec.aimCount ||
            !model.tileIndexingMode.equals(currentConfig.actionSpaceSpec.tileIndexingMode.name()) ||
            model.hasGlobalVector != currentConfig.stateEncodingSpec.hasGlobalVector ||
            model.globalFeatureCount != currentConfig.stateEncodingSpec.globalFeatureCount ||
            model.hasObjectTable != currentConfig.stateEncodingSpec.hasObjectTable ||
            model.hasGraphState != currentConfig.stateEncodingSpec.hasGraphState) {
            throw new IllegalArgumentException("Model compatibility check failed! " +
                    "Model uses encoder version " + model.stateEncoderVersion + 
                    " (" + model.voxelChannels + " channels), " +
                    "but current runtime is " + currentConfig.stateEncodingSpec.encoderVersion + 
                    " (" + currentConfig.stateEncodingSpec.voxelChannels + " channels).");
        }

        rlService.resetRuntimeState();
        rlService.setEpisodesTrained(model.episodesTrained);
        rlService.setEpsilon(model.epsilon);
        rlService.setEpsilonDecay(model.epsilonDecayRate > 0.0 ? model.epsilonDecayRate : 0.9995);
        rlService.setMinEpsilon(model.minEpsilon > 0.0 ? model.minEpsilon : 0.05);
        rlService.setBestScore(model.bestScore);
        rlService.setUse2dCnn(model.use2dCnn);
        if (model.rewardConfig != null) {
            rlService.setRewardConfig(model.rewardConfig.clone());
        }
        if (model.supervisorConfig != null) {
            rlService.setSupervisorConfig(model.supervisorConfig.clone());
        }
        
        if (model.bestBaseEvalJson != null && !model.bestBaseEvalJson.isEmpty()) {
            rlService.setBestGridModel(com.rustbuilder.model.GridSerializer.fromJson(model.bestBaseEvalJson));
        }
        if (model.bestBaseRewardJson != null && !model.bestBaseRewardJson.isEmpty()) {
            rlService.setBestRewardGridModel(com.rustbuilder.model.GridSerializer.fromJson(model.bestBaseRewardJson));
        }
    }
}
