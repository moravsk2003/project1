package com.rustbuilder.ui;

import java.util.List;

import com.rustbuilder.ai.core.AIModelManager;
import com.rustbuilder.ai.core.AIModelManager.AIModel;
import com.rustbuilder.ai.ea.BaseGenome;
import com.rustbuilder.ai.ea.GeneticAlgorithmService;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.ui.hints.HintKey;
import com.rustbuilder.ui.hints.HintUtils;

import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Separator;
import javafx.scene.control.Slider;
import javafx.scene.control.Spinner;
import javafx.scene.control.TextField;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;

/**
 * Dialog for AI base generation.
 * Includes model management, priority sliders, and training controls.
 */
public class GeneratorDialog {

    private final GridModel gridModel;
    private final GameCanvas gameCanvas;
    private final Stage dialogStage;

    private GeneticAlgorithmService gaService;
    private String currentModelName;

    // UI Components
    private ComboBox<String> modelComboBox;
    private TextField newModelField;
    private Slider logisticsSlider;
    private Slider costSlider;
    private Slider raidSlider;
    private Slider workingAreaSlider;
    private Slider safeZoneSlider;
    private Label logLabel;
    private Label costLabel;
    private Label raidLabel;
    private Label workingAreaLabel;
    private Label safeZoneLabel;
    private Spinner<Integer> genSpinner;
    private Slider epochSlider;
    private Label epochLabel;
    private Slider mutationSlider;
    private Label mutationLabel;
    private ProgressBar progressBar;
    private Label statusLabel;
    private Button trainButton;
    private Button generateButton;
    private Button deleteButton;

    // Advanced settings
    private Spinner<Integer> popSizeSpinner;
    private Spinner<Integer> tournamentSpinner;
    private Spinner<Integer> eliteSpinner;
    private Spinner<Integer> freshBloodSpinner;
    private Spinner<Integer> stagnationSpinner;
    private Slider maxMutationSlider;
    private Label maxMutationLabel;
    private CheckBox autoGenCheckbox;

    // Stats
    private final Label raidStatsLabel     = new Label("Raid Cost: —");
    private final Label costStatsLabel     = new Label("Build Cost: —");
    private final Label logisticsStatsLabel = new Label("Logistics: —");
    private final Label workingAreaStatsLabel = new Label("Working Area: —");
    private final Label safeZoneStatsLabel = new Label("Safe Zone: —");

    // ── Shared styles ──────────────────────────────────────────────────────────
    private static final String CARD_STYLE =
        "-fx-background-color: #333; " +
        "-fx-background-radius: 6; " +
        "-fx-border-color: #4a4a4a; " +
        "-fx-border-radius: 6; " +
        "-fx-padding: 10 12 10 12;";

    private static final String SECTION_TITLE =
        "-fx-font-size: 13px; -fx-font-weight: bold; -fx-text-fill: #ddd;";

    private static final String BODY_LABEL =
        "-fx-text-fill: #bbb; -fx-font-size: 11px;";

    private static final String VALUE_LABEL =
        "-fx-text-fill: #e8e8e8; -fx-font-size: 11px; " +
        "-fx-font-family: 'Consolas', monospace;";

    public GeneratorDialog(GridModel gridModel, GameCanvas gameCanvas, Stage owner) {
        this.gridModel = gridModel;
        this.gameCanvas = gameCanvas;
        this.gaService = new GeneticAlgorithmService();

        dialogStage = new Stage();
        dialogStage.initModality(Modality.WINDOW_MODAL);
        dialogStage.initOwner(owner);
        dialogStage.setTitle("🤖 AI Base Generator");
        dialogStage.setResizable(true);

        VBox content = new VBox(10);
        content.setPadding(new Insets(14));
        content.setStyle("-fx-background-color: #2b2b2b;");

        content.getChildren().addAll(
            createModelSection(),
            new Separator(),
            createPrioritySection(),
            new Separator(),
            createTrainingSection(),
            new Separator(),
            createAdvancedSection(),
            new Separator(),
            createStatsSection(),
            new Separator(),
            createActionSection()
        );

        ScrollPane scroll = new ScrollPane(content);
        scroll.setFitToWidth(true);
        scroll.setStyle("-fx-background: #2b2b2b; -fx-background-color: #2b2b2b;");
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);

        Scene scene = new Scene(scroll, 460, 740);
        dialogStage.setScene(scene);
    }

    // ── MODEL ────────────────────────────────────────────────────────────────

    private VBox createModelSection() {
        VBox box = card();
        Label title = sectionTitle("📁  Model");

        HBox selectRow = new HBox(8);
        selectRow.setAlignment(Pos.CENTER_LEFT);
        modelComboBox = new ComboBox<>();
        modelComboBox.setPrefWidth(210);
        modelComboBox.setPromptText("Select model…");
        HintUtils.attachHint(modelComboBox, "Вибір моделі", "Оберіть раніше збережену модель зі списку.");
        refreshModelList();

        Button loadBtn = styledBtn("Load", "#3498db");
        loadBtn.setOnAction(e -> loadSelectedModel());

        deleteButton = styledBtn("🗑", "#e74c3c");
        deleteButton.setOnAction(e -> deleteSelectedModel());

        selectRow.getChildren().addAll(modelComboBox, loadBtn, deleteButton);

        HBox newRow = new HBox(8);
        newRow.setAlignment(Pos.CENTER_LEFT);
        newModelField = new TextField();
        newModelField.setPromptText("New model name…");
        HintUtils.attachHint(newModelField, "Назва нової моделі", "Введіть назву для створення нової або збереження поточної моделі.");
        HBox.setHgrow(newModelField, Priority.ALWAYS);

        Button createBtn = styledBtn("+ Create", "#2ecc71");
        createBtn.setOnAction(e -> createNewModel());
        newRow.getChildren().addAll(newModelField, createBtn);

        box.getChildren().addAll(title, selectRow, newRow);
        return box;
    }

    // ── PRIORITIES ──────────────────────────────────────────────────────────

    private VBox createPrioritySection() {
        VBox box = card();
        Label title = sectionTitle("⚖  Priorities");

        logLabel        = valueLabel("Logistics:        22%");
        HintUtils.attachHint(logLabel, HintKey.LOGISTICS);
        logisticsSlider = buildSlider(0, 1, 0.22, 0.25);
        logisticsSlider.valueProperty().addListener((o, ov, nv) -> updateSliderLabels());

        costLabel  = valueLabel("Resources:        18%");
        HintUtils.attachHint(costLabel, HintKey.RESOURCES);
        costSlider = buildSlider(0, 1, 0.18, 0.20);
        costSlider.valueProperty().addListener((o, ov, nv) -> updateSliderLabels());

        raidLabel  = valueLabel("Raid Resistance:  28%");
        HintUtils.attachHint(raidLabel, HintKey.RAID_RESISTANCE);
        raidSlider = buildSlider(0, 1, 0.28, 0.30);
        raidSlider.valueProperty().addListener((o, ov, nv) -> updateSliderLabels());

        workingAreaLabel = valueLabel("Working Area:     22%");
        HintUtils.attachHint(workingAreaLabel, HintKey.WORKING_AREA);
        workingAreaSlider = buildSlider(0, 1, 0.22, 0.25);
        workingAreaSlider.valueProperty().addListener((o, ov, nv) -> updateSliderLabels());

        safeZoneLabel = valueLabel("Safe Zone:        10%");
        HintUtils.attachHint(safeZoneLabel, HintKey.SAFE_ZONE);
        safeZoneSlider = buildSlider(0, 1, 0.10, 0.10);
        safeZoneSlider.valueProperty().addListener((o, ov, nv) -> updateSliderLabels());

        box.getChildren().addAll(title,
            logLabel, logisticsSlider,
            costLabel, costSlider,
            raidLabel, raidSlider,
            workingAreaLabel, workingAreaSlider,
            safeZoneLabel, safeZoneSlider);
        return box;
    }

    private void updateSliderLabels() {
        double sum = logisticsSlider.getValue() + costSlider.getValue() + raidSlider.getValue() + workingAreaSlider.getValue() + safeZoneSlider.getValue();
        if (sum < 0.001) sum = 1.0;
        logLabel .setText(String.format("Logistics:        %3.0f%%", logisticsSlider.getValue() / sum * 100));
        costLabel.setText(String.format("Resources:        %3.0f%%", costSlider.getValue()       / sum * 100));
        raidLabel.setText(String.format("Raid Resistance:  %3.0f%%", raidSlider.getValue()       / sum * 100));
        workingAreaLabel.setText(String.format("Working Area:     %3.0f%%", workingAreaSlider.getValue() / sum * 100));
        safeZoneLabel.setText(String.format("Safe Zone:        %3.0f%%", safeZoneSlider.getValue() / sum * 100));
    }

    private double[] getNormalizedWeights() {
        double sum = logisticsSlider.getValue() + costSlider.getValue() + raidSlider.getValue() + workingAreaSlider.getValue() + safeZoneSlider.getValue();
        if (sum < 0.001) return new double[]{0.22, 0.18, 0.28, 0.22, 0.10};
        return new double[]{
            logisticsSlider.getValue() / sum,
            costSlider.getValue() / sum,
            raidSlider.getValue() / sum,
            workingAreaSlider.getValue() / sum,
            safeZoneSlider.getValue() / sum
        };
    }

    // ── TRAINING ─────────────────────────────────────────────────────────────

    private VBox createTrainingSection() {
        VBox box = card();
        Label title = sectionTitle("🧬  Training");

        Label genLabel = bodyLabel("Generations:");
        HintUtils.attachHint(genLabel, HintKey.GENERATIONS);
        genSpinner = new Spinner<>(1, 2000, 20, 5);
        genSpinner.setPrefWidth(80);
        genSpinner.setEditable(true);

        trainButton = styledBtn("▶ Train", "#e67e22");
        trainButton.setStyle(trainButton.getStyle() + " -fx-font-weight: bold;");
        trainButton.setOnAction(e -> startTraining());

        HBox trainRow = new HBox(8, genLabel, genSpinner, trainButton);
        trainRow.setAlignment(Pos.CENTER_LEFT);

        epochLabel = valueLabel("Epochs: 1  (total: 20 generations)");
        HintUtils.attachHint(epochLabel, HintKey.EPOCHS);
        epochSlider = buildSlider(1, 100, 1, 10);
        epochSlider.setMinorTickCount(4);
        epochSlider.setSnapToTicks(true);
        epochSlider.setBlockIncrement(1);
        epochSlider.valueProperty().addListener((o, ov, nv) -> {
            int ep = (int) Math.round(nv.doubleValue());
            epochLabel.setText(String.format("Epochs: %d  (total: %d generations)",
                ep, ep * genSpinner.getValue()));
        });

        mutationLabel = valueLabel("Base Mutation: 20%");
        HintUtils.attachHint(mutationLabel, HintKey.BASE_MUTATION);
        mutationSlider = buildSlider(0.01, 1.0, 0.20, 0.1);
        mutationSlider.valueProperty().addListener((o, ov, nv) ->
            mutationLabel.setText(String.format("Base Mutation: %.0f%%", nv.doubleValue() * 100)));

        progressBar = new ProgressBar(0);
        progressBar.setPrefWidth(Double.MAX_VALUE);
        progressBar.setVisible(false);

        statusLabel = new Label("No model loaded. Create or load a model to start.");
        statusLabel.setStyle("-fx-text-fill: #888; -fx-font-size: 11px;");
        statusLabel.setWrapText(true);

        autoGenCheckbox = new CheckBox("Auto-generate best after each epoch");
        autoGenCheckbox.setSelected(true);
        autoGenCheckbox.setStyle("-fx-text-fill: #bbb; -fx-font-size: 11px;");

        box.getChildren().addAll(title, trainRow,
            epochLabel, epochSlider,
            mutationLabel, mutationSlider,
            autoGenCheckbox,
            progressBar, statusLabel);
        return box;
    }

    // ── ADVANCED ─────────────────────────────────────────────────────────────

    private VBox createAdvancedSection() {
        VBox box = card();
        Label title = sectionTitle("⚙  Advanced GA Settings");

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(6);

        popSizeSpinner    = advSpinner(20, 500, 80, 10,
            "Genomes per generation. Larger → better coverage, slower.");
        tournamentSpinner = advSpinner(2, 20, 4, 1,
            "Candidates in selection tournament. Higher → more pressure on best.");
        eliteSpinner      = advSpinner(0, 50, 5, 1,
            "Top N genomes copied unchanged to next generation.");
        freshBloodSpinner = advSpinner(0, 20, 3, 1,
            "Random genomes injected each generation for diversity.");
        stagnationSpinner = advSpinner(5, 200, 15, 5,
            "Generations without improvement before boosting mutation rate.");

        int row = 0;
        row = addGridRow(grid, row, "Population size",   popSizeSpinner, HintKey.POPULATION_SIZE);
        row = addGridRow(grid, row, "Tournament size",   tournamentSpinner, HintKey.TOURNAMENT_SIZE);
        row = addGridRow(grid, row, "Elite count",       eliteSpinner, HintKey.ELITE_COUNT);
        row = addGridRow(grid, row, "Fresh blood / gen", freshBloodSpinner, HintKey.FRESH_BLOOD);
        addGridRow(grid, row, "Stagnation limit",  stagnationSpinner, HintKey.STAGNATION_LIMIT);

        maxMutationLabel = valueLabel("Max mutation burst: 60%");
        HintUtils.attachHint(maxMutationLabel, HintKey.MAX_MUTATION_BURST);
        maxMutationSlider = buildSlider(0.20, 1.0, 0.60, 0.20);
        maxMutationSlider.valueProperty().addListener((o, ov, nv) ->
            maxMutationLabel.setText(String.format("Max mutation burst: %.0f%%", nv.doubleValue() * 100)));

        box.getChildren().addAll(title, grid, maxMutationLabel, maxMutationSlider);
        return box;
    }

    private int addGridRow(GridPane grid, int row, String text, Spinner<Integer> ctrl, HintKey hint) {
        Label lbl = bodyLabel(text);
        HintUtils.attachHint(lbl, hint);
        HintUtils.attachHint(ctrl.getEditor(), hint);
        grid.add(lbl, 0, row);
        grid.add(ctrl, 1, row);
        GridPane.setHgrow(lbl, Priority.ALWAYS);
        return row + 1;
    }

    // ── STATS ────────────────────────────────────────────────────────────────

    private VBox createStatsSection() {
        VBox box = card();
        box.setStyle(CARD_STYLE + " -fx-background-color: #2e2e2e;");
        Label title = sectionTitle("📊  Last Evaluation");
        raidStatsLabel.setStyle(VALUE_LABEL);
        HintUtils.attachHint(raidStatsLabel, HintKey.RAID_RESISTANCE);
        costStatsLabel.setStyle(VALUE_LABEL);
        HintUtils.attachHint(costStatsLabel, HintKey.RESOURCES);
        logisticsStatsLabel.setStyle(VALUE_LABEL);
        HintUtils.attachHint(logisticsStatsLabel, HintKey.LOGISTICS);
        workingAreaStatsLabel.setStyle(VALUE_LABEL);
        HintUtils.attachHint(workingAreaStatsLabel, HintKey.WORKING_AREA);
        safeZoneStatsLabel.setStyle(VALUE_LABEL);
        HintUtils.attachHint(safeZoneStatsLabel, HintKey.SAFE_ZONE);
        box.getChildren().addAll(title, raidStatsLabel, costStatsLabel, logisticsStatsLabel, workingAreaStatsLabel, safeZoneStatsLabel);
        return box;
    }

    // ── ACTIONS ──────────────────────────────────────────────────────────────

    private HBox createActionSection() {
        HBox box = new HBox(10);
        box.setAlignment(Pos.CENTER);

        generateButton = styledBtn("🏠 Generate", "#9b59b6");
        generateButton.setStyle(generateButton.getStyle() +
            " -fx-font-weight: bold; -fx-font-size: 13px;");
        generateButton.setPrefWidth(140);
        generateButton.setDisable(true);
        HintUtils.attachHint(generateButton, "Генерувати", "Перенести найкращу згенеровану базу на ігрове поле.");
        generateButton.setOnAction(e -> generateBest());

        Button saveBtn = styledBtn("💾 Save", "#1abc9c");
        HintUtils.attachHint(saveBtn, "Зберегти", "Зберегти поточну модель.");
        saveBtn.setOnAction(e -> saveCurrentModel());

        Button closeBtn = styledBtn("✕ Close", "#555");
        closeBtn.setOnAction(e -> dialogStage.close());

        box.getChildren().addAll(generateButton, saveBtn, closeBtn);
        return box;
    }

    // ── LOGIC ─────────────────────────────────────────────────────────────────

    private void refreshModelList() {
        List<String> models = AIModelManager.listModels();
        modelComboBox.getItems().clear();
        modelComboBox.getItems().addAll(models);
    }

    private void createNewModel() {
        String name = newModelField.getText().trim();
        if (name.isEmpty()) { showAlert("Enter a model name."); return; }
        if (name.contains(" ") || name.contains("/") || name.contains("\\")) {
            showAlert("Model name cannot contain spaces or slashes."); return;
        }
        currentModelName = name;
        gaService = new GeneticAlgorithmService();
        gaService.initializePopulation();
        generateButton.setDisable(true);
        setStatus(String.format("✅ Model '%s' created — %d genomes ready to train.",
            name, gaService.getPopulationSize()), "#2ecc71");
        saveCurrentModel();
        refreshModelList();
    }

    private void loadSelectedModel() {
        String name = modelComboBox.getValue();
        if (name == null || name.isEmpty()) { showAlert("Select a model to load."); return; }
        try {
            AIModel model = AIModelManager.loadModel(name);
            currentModelName = name;
            gaService = new GeneticAlgorithmService();
            AIModelManager.restoreFromModel(gaService, model);
            double lw = model.logisticsWeight;
            double cw = model.costWeight;
            double rw = model.raidWeight;
            double ww = model.workingAreaWeight;
            double sw = model.safeZoneWeight;
            double sum = lw + cw + rw + ww + sw;
            if (sum < 0.0001) {
                lw = 0.22;
                cw = 0.18;
                rw = 0.28;
                ww = 0.22;
                sw = 0.10;
            } else {
                lw /= sum;
                cw /= sum;
                rw /= sum;
                ww /= sum;
                sw /= sum;
            }
            logisticsSlider.setValue(lw);
            costSlider.setValue(cw);
            raidSlider.setValue(rw);
            workingAreaSlider.setValue(ww);
            safeZoneSlider.setValue(sw);
            updateSliderLabels();
            generateButton.setDisable(gaService.getBestGenome() == null);
            setStatus(String.format("📂 Loaded '%s' — Gen %d, Best: %.4f",
                name, model.generation, model.bestFitness), "#3498db");
        } catch (Exception ex) {
            showAlert("Failed to load model: " + ex.getMessage());
        }
    }

    private void deleteSelectedModel() {
        String name = modelComboBox.getValue();
        if (name == null || name.isEmpty()) return;
        if (AIModelManager.deleteModel(name)) {
            refreshModelList();
            if (name.equals(currentModelName)) {
                currentModelName = null;
                gaService = new GeneticAlgorithmService();
                generateButton.setDisable(true);
                setStatus("🗑 Model deleted.", "#aaa");
            }
        }
    }

    private void saveCurrentModel() {
        if (currentModelName == null) { showAlert("No model active. Create or load a model first."); return; }
        try {
            double[] w = getNormalizedWeights();
            AIModel model = AIModelManager.createSnapshot(currentModelName, gaService, w[0], w[1], w[2], w[3], w[4]);
            AIModelManager.saveModel(model);
            setStatus("💾 Model '" + currentModelName + "' saved.", "#aaa");
        } catch (Exception ex) {
            showAlert("Failed to save model: " + ex.getMessage());
        }
    }

    private void startTraining() {
        if (currentModelName == null) { showAlert("Create or load a model first."); return; }

        int generations = genSpinner.getValue();
        int epochs      = (int) Math.round(epochSlider.getValue());
        int totalGens   = generations * epochs;
        double[] w      = getNormalizedWeights();

        trainButton.setDisable(true);
        generateButton.setDisable(true);
        progressBar.setVisible(true);
        progressBar.setProgress(0);
        setStatus("⏳ Training…", "#e67e22");

        final double logW = w[0], costW = w[1], raidW = w[2], workingAreaW = w[3], safeZoneW = w[4];

        gaService.setMutationRate(mutationSlider.getValue());
        gaService.setPopulationSize(popSizeSpinner.getValue());
        gaService.setTournamentSize(tournamentSpinner.getValue());
        gaService.setEliteCount(eliteSpinner.getValue());
        gaService.setFreshBloodCount(freshBloodSpinner.getValue());
        gaService.setStagnationLimit(stagnationSpinner.getValue());
        gaService.setMaxMutationRate(maxMutationSlider.getValue());

        if (gaService.getPopulation().isEmpty()) gaService.initializePopulation();

        final int startGen = gaService.getGeneration();

        Thread trainThread = new Thread(() -> {
            for (int epoch = 0; epoch < epochs; epoch++) {
                final int currentEpoch = epoch + 1;
                gaService.evolve(generations, logW, costW, raidW, workingAreaW, safeZoneW, progress -> {
                    double absGen    = progress[0];
                    double best      = progress[1];
                    double mutRate   = progress[2];
                    int    stagCount = (int) progress[3];

                    int done = (int) absGen - startGen;
                    Platform.runLater(() -> {
                        progressBar.setProgress((double) done / totalGens);
                        String stagInfo = stagCount > 0
                            ? String.format(" | Stag %d | Mut %.0f%%", stagCount, mutRate * 100)
                            : String.format(" | Mut %.0f%%", mutRate * 100);
                        setStatus(String.format("Epoch %d/%d — %d/%d gen%s — Best: %.4f",
                            currentEpoch, epochs, done, totalGens, stagInfo, best), "#e67e22");
                        updateStatsLabels();
                    });
                });

                final int ep = currentEpoch;
                try {
                    AIModel snapshot = AIModelManager.createSnapshot(currentModelName, gaService, logW, costW, raidW, workingAreaW, safeZoneW);
                    AIModelManager.saveModel(snapshot);
                    Platform.runLater(() -> {
                        setStatus(String.format("💾 Epoch %d/%d saved — Best: %.4f", ep, epochs, snapshot.bestFitness), "#aaa");
                        // Auto-generate: decode best genome onto the main grid
                        if (autoGenCheckbox.isSelected() && gaService.getBestGenome() != null) {
                            gaService.getBestGenome().decode(gridModel);
                            gameCanvas.draw();
                        }
                    });
                } catch (Exception ex) {
                    final String err = ex.getMessage();
                    Platform.runLater(() -> setStatus("⚠ Auto-save failed: " + err, "#e74c3c"));
                }
            }

            Platform.runLater(() -> {
                trainButton.setDisable(false);
                generateButton.setDisable(gaService.getBestGenome() == null);
                progressBar.setVisible(false);
                setStatus(String.format("✅ Done — %d epochs, Gen %d, Best: %.4f",
                    epochs, gaService.getGeneration(), gaService.getBestFitness()), "#2ecc71");
                saveCurrentModel();
                refreshModelList();
            });
        });
        trainThread.setDaemon(true);
        trainThread.start();
    }

    private void generateBest() {
        BaseGenome best = gaService.getBestGenome();
        if (best == null) { showAlert("No trained genome available. Train first."); return; }
        boolean placed = best.decode(gridModel);
        if (placed) {
            gameCanvas.draw();
            setStatus(String.format("🏠 Generated — %d blocks (fitness: %.4f)",
                gridModel.getAllBlocks().size(), best.getFitness()), "#9b59b6");
            updateStatsLabels();
        } else {
            showAlert("Failed to generate base. Try training more generations.");
        }
    }

    private void updateStatsLabels() {
        com.rustbuilder.service.evaluator.HouseEvaluator.EvaluationResult result = gaService.getLastBestResult();
        if (result == null) return;

        int tcS = result.raid.sulfurToTC;
        int lrS = result.raid.sulfurToLootRoom;
        raidStatsLabel.setText(String.format("🔴 Raid Cost:  TC = %s   LR = %s",
            tcS == Integer.MAX_VALUE ? "—" : String.valueOf(tcS),
            lrS == Integer.MAX_VALUE ? "—" : String.valueOf(lrS)));
        int stone = result.cost.totalCost.getOrDefault(com.rustbuilder.model.core.ResourceType.STONE, 0);
        int metal = result.cost.totalCost.getOrDefault(com.rustbuilder.model.core.ResourceType.METAL, 0);
        int hqm   = result.cost.totalCost.getOrDefault(com.rustbuilder.model.core.ResourceType.HQM,   0);
        costStatsLabel.setText(String.format("🪨 Build Cost:  %d Stone  %d Metal  %d HQM",
            stone, metal, hqm));
        double dist = result.logistics.entranceToTC;
        logisticsStatsLabel.setText(String.format("📐 Logistics:  Entrance→TC %.1f",
            dist == Double.MAX_VALUE ? 0 : dist));
        workingAreaStatsLabel.setText(String.format("🟩 Working Area:  area %.2f  perim %d  score %.3f",
            result.workingArea.protectedArea, result.workingArea.openEdgePerimeter, result.workingArea.score));
        safeZoneStatsLabel.setText(String.format("🛡 Safe Zone: closed %d  score %.3f",
            result.safeZone.closedBlocks, result.safeZone.score));
    }

    public void show() { dialogStage.showAndWait(); }

    // ── SHARED HELPERS ────────────────────────────────────────────────────────

    private VBox card() {
        VBox v = new VBox(7);
        v.setStyle(CARD_STYLE);
        return v;
    }

    private Label sectionTitle(String text) {
        Label l = new Label(text);
        l.setStyle(SECTION_TITLE);
        return l;
    }

    private Label bodyLabel(String text) {
        Label l = new Label(text);
        l.setStyle(BODY_LABEL);
        return l;
    }

    private Label valueLabel(String text) {
        Label l = new Label(text);
        l.setStyle(VALUE_LABEL);
        return l;
    }

    private Button styledBtn(String text, String color) {
        Button b = new Button(text);
        b.setStyle("-fx-background-color: " + color + "; -fx-text-fill: white;");
        return b;
    }

    private Slider buildSlider(double min, double max, double def, double tickUnit) {
        Slider s = new Slider(min, max, def);
        s.setShowTickLabels(true);
        s.setShowTickMarks(true);
        s.setMajorTickUnit(tickUnit);
        return s;
    }

    private Spinner<Integer> advSpinner(int min, int max, int def, int step, String tip) {
        Spinner<Integer> s = new Spinner<>(min, max, def, step);
        s.setPrefWidth(90);
        s.setEditable(true);
        Tooltip.install(s, new Tooltip(tip));
        return s;
    }

    private void setStatus(String text, String color) {
        statusLabel.setText(text);
        statusLabel.setStyle("-fx-text-fill: " + color + "; -fx-font-size: 11px;");
    }

    private void showAlert(String msg) {
        Alert alert = new Alert(Alert.AlertType.WARNING);
        alert.setTitle("AI Generator");
        alert.setHeaderText(null);
        alert.setContentText(msg);
        alert.showAndWait();
    }
}
