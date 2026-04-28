package com.rustbuilder.ui;

import java.io.IOException;

import com.rustbuilder.ai.core.TrainingMetrics;
import com.rustbuilder.ai.rl.RLModelManager;
import com.rustbuilder.ai.rl.RLModelManager.RLModel;
import com.rustbuilder.ai.rl.RLTrainingService;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
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
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.Modality;
import javafx.stage.Stage;

/**
 * A dialog to configure and run RL training (JavaFX implementation).
 * Updated to include Multi-Discrete support, diagnostic step, and improved stats.
 */
public class RLGeneratorDialog {

    private final GridModel mainGrid;
    private final GameCanvas gameCanvas; // Optional, can be null
    private final Stage dialogStage;
    private final Runnable refreshCallback;

    private RLTrainingService rlService;
    private String currentModelName;
    private boolean trainingRunning = false;

    // UI Components
    private ComboBox<String> modelComboBox;
    private TextField newModelField;
    
    private Slider logisticsSlider;
    private Slider costSlider;
    private Slider raidSlider;
    private Slider workingAreaSlider;
    
    private Label logLabel;
    private Label costLabel;
    private Label raidLabel;
    private Label workingAreaLabel;
    
    private Spinner<Integer> episodesSpinner;
    private Spinner<Integer> stepsSpinner;
    private Spinner<Integer> epochsSpinner;
    
    private CheckBox multiDiscreteCheck;
    private ProgressBar progressBar;
    private Label statusLabel;
    private TextArea diagnosticLogArea;
    
    private Button trainButton;
    private Button stopButton;
    private Button generateButton;
    private Button diagnosticButton;
    
    // Summary Labels
    private Label bestScoreLbl;
    private Label avgEvalScoreLbl;
    private Label avgStepRewardLbl;
    private Label invalidRateLbl;
    private Label memorySizeLbl;
    private Label episodesLbl;
    private Label epsilonLbl;
    private Label lossLbl;
    private Label bestBaseLbl;
    private Label ramUsageLbl;
    private Label timerLbl;
    
    // Reward Config UI Map
    private java.util.Map<String, Spinner<Double>> rewardSpinners = new java.util.HashMap<>();

    // shared styles
    private static final String CARD_STYLE =
        "-fx-background-color: #333; -fx-background-radius: 6; -fx-border-color: #4a4a4a; -fx-border-radius: 6; -fx-padding: 10 12 10 12;";
    private static final String SECTION_TITLE =
        "-fx-font-size: 13px; -fx-font-weight: bold; -fx-text-fill: #ddd;";
    private static final String BODY_LABEL =
        "-fx-text-fill: #bbb; -fx-font-size: 11px;";
    private static final String VALUE_LABEL =
        "-fx-text-fill: #e8e8e8; -fx-font-size: 11px; -fx-font-family: 'Consolas', monospace;";

    public RLGeneratorDialog(Stage owner, RLTrainingService rlService, GridModel mainGrid, Runnable refreshCallback, GameCanvas gameCanvas) {
        this.rlService = rlService;
        this.mainGrid = mainGrid;
        this.refreshCallback = refreshCallback;
        this.gameCanvas = gameCanvas;

        dialogStage = new Stage();
        dialogStage.initModality(Modality.NONE);
        if (owner != null) dialogStage.initOwner(owner);
        dialogStage.setTitle("RL Base Generator");
        dialogStage.setResizable(true);

        VBox mainContent = new VBox(10);
        mainContent.setPadding(new Insets(14));
        mainContent.setStyle("-fx-background-color: #2b2b2b;");

        mainContent.getChildren().addAll(
            createModelSection(),
            new Separator(),
            createPrioritySection(),
            new Separator(),
            createTrainingSection(),
            new Separator(),
            createRewardConfigSection(),
            new Separator(),
            createExperimentalSection(),
            new Separator(),
            createStatsSection(),
            new Separator(),
            createSummarySection(),
            new Separator(),
            createActionSection()
        );

        ScrollPane scroll = new ScrollPane(mainContent);
        scroll.setFitToWidth(true);
        scroll.setStyle("-fx-background: #2b2b2b; -fx-background-color: #2b2b2b;");
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);

        Scene scene = new Scene(scroll, 500, 700);
        dialogStage.setScene(scene);
        
        dialogStage.setOnCloseRequest(e -> cleanup());
    }

    /**
     * Static helper for opening the dialog, compatible with existing call sites.
     */
    public static void showDialog(Stage owner, RLTrainingService service, GridModel grid, Runnable refresh) {
        RLGeneratorDialog dialog = new RLGeneratorDialog(owner, service, grid, refresh, null);
        dialog.show();
    }

    private VBox createModelSection() {
        VBox box = card();
        box.getChildren().add(sectionTitle("📁  Model Management"));

        HBox selectRow = new HBox(8);
        selectRow.setAlignment(Pos.CENTER_LEFT);
        modelComboBox = new ComboBox<>();
        modelComboBox.setPrefWidth(210);
        modelComboBox.setPromptText("Select model...");
        HintUtils.attachHint(modelComboBox, "Вибір моделі", "Оберіть раніше збережену модель зі списку.");
        refreshModelList();

        Button loadBtn = styledBtn("Load", "#3498db");
        HintUtils.attachHint(loadBtn, "Завантажити", "Завантажити обрану модель для подальшого використання або тренування.");
        loadBtn.setOnAction(e -> loadModel());

        Button deleteBtn = styledBtn("🗑", "#e74c3c");
        HintUtils.attachHint(deleteBtn, "Видалити", "Безповоротно видалити обрану модель.");
        deleteBtn.setOnAction(e -> deleteModel());

        selectRow.getChildren().addAll(modelComboBox, loadBtn, deleteBtn);

        HBox newRow = new HBox(8);
        newRow.setAlignment(Pos.CENTER_LEFT);
        newModelField = new TextField();
        newModelField.setPromptText("New model name...");
        HintUtils.attachHint(newModelField, "Назва нової моделі", "Введіть назву для створення нової або збереження поточної моделі.");
        HBox.setHgrow(newModelField, Priority.ALWAYS);

        Button saveBtn = styledBtn("Save As", "#2ecc71");
        HintUtils.attachHint(saveBtn, "Зберегти як", "Зберегти поточний стан нейромережі під новою назвою.");
        saveBtn.setOnAction(e -> saveModel());
        newRow.getChildren().addAll(newModelField, saveBtn);

        box.getChildren().addAll(selectRow, newRow);
        return box;
    }

    private VBox createPrioritySection() {
        VBox box = card();
        box.getChildren().add(sectionTitle("⚖  Reward Weights"));

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(5);

        logLabel = valueLabel("Logistics: 1.00");
        HintUtils.attachHint(logLabel, HintKey.LOGISTICS);
        logisticsSlider = buildSlider(0, 2, 1.0, 0.5);
        logisticsSlider.valueProperty().addListener((o, ov, nv) -> logLabel.setText(String.format("Logistics: %.2f", nv)));

        costLabel = valueLabel("Resources: 0.80");
        HintUtils.attachHint(costLabel, HintKey.RESOURCES);
        costSlider = buildSlider(0, 2, 0.8, 0.5);
        costSlider.valueProperty().addListener((o, ov, nv) -> costLabel.setText(String.format("Resources: %.2f", nv)));

        raidLabel = valueLabel("Raid Resistance: 1.20");
        HintUtils.attachHint(raidLabel, HintKey.RAID_RESISTANCE);
        raidSlider = buildSlider(0, 2, 1.2, 0.5);
        raidSlider.valueProperty().addListener((o, ov, nv) -> raidLabel.setText(String.format("Raid Resistance: %.2f", nv)));

        workingAreaLabel = valueLabel("Working Area: 1.00");
        HintUtils.attachHint(workingAreaLabel, HintKey.WORKING_AREA);
        workingAreaSlider = buildSlider(0, 2, 1.0, 0.5);
        workingAreaSlider.valueProperty().addListener((o, ov, nv) -> workingAreaLabel.setText(String.format("Working Area: %.2f", nv)));

        grid.add(logLabel, 0, 0); grid.add(logisticsSlider, 1, 0);
        grid.add(costLabel, 0, 1); grid.add(costSlider, 1, 1);
        grid.add(raidLabel, 0, 2); grid.add(raidSlider, 1, 2);
        grid.add(workingAreaLabel, 0, 3); grid.add(workingAreaSlider, 1, 3);

        box.getChildren().add(grid);
        return box;
    }

    private VBox createTrainingSection() {
        VBox box = card();
        box.getChildren().add(sectionTitle("🧠  Training Parameters"));

        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(6);

        Label epLabel = bodyLabel("Episodes per Epoch:");
        HintUtils.attachHint(epLabel, HintKey.EPISODES_PER_EPOCH);
        grid.add(epLabel, 0, 0);
        episodesSpinner = new Spinner<>(10, 5000, 200, 50);
        episodesSpinner.setEditable(true);
        HintUtils.attachHint(episodesSpinner.getEditor(), HintKey.EPISODES_PER_EPOCH);
        grid.add(episodesSpinner, 1, 0);

        Label stepsLabel = bodyLabel("Max Steps per Ep:");
        HintUtils.attachHint(stepsLabel, HintKey.MAX_STEPS_PER_EP);
        grid.add(stepsLabel, 0, 1);
        stepsSpinner = new Spinner<>(10, 200, 40, 5);
        stepsSpinner.setEditable(true);
        HintUtils.attachHint(stepsSpinner.getEditor(), HintKey.MAX_STEPS_PER_EP);
        grid.add(stepsSpinner, 1, 1);

        Label epochsLabel = bodyLabel("Epochs:");
        HintUtils.attachHint(epochsLabel, HintKey.EPOCHS);
        grid.add(epochsLabel, 0, 2);
        epochsSpinner = new Spinner<>(1, 500, 50, 10);
        epochsSpinner.setEditable(true);
        HintUtils.attachHint(epochsSpinner.getEditor(), HintKey.EPOCHS);
        grid.add(epochsSpinner, 1, 2);

        box.getChildren().add(grid);
        return box;
    }

    private VBox createExperimentalSection() {
        VBox box = card();
        box.getChildren().add(sectionTitle("🧪  Experimental Settings"));

        multiDiscreteCheck = new CheckBox("Enable Multi-Discrete Flow");
        HintUtils.attachHint(multiDiscreteCheck, HintKey.MULTI_DISCRETE);
        multiDiscreteCheck.setStyle("-fx-text-fill: #ecf0f1;");
        multiDiscreteCheck.setSelected(rlService.isUseMultiDiscreteFlow());
        multiDiscreteCheck.setOnAction(e -> rlService.setUseMultiDiscreteFlow(multiDiscreteCheck.isSelected()));

        CheckBox aimSectorCheck = new CheckBox("Learn Aim Sector (Phase 5)");
        HintUtils.attachHint(aimSectorCheck, HintKey.MULTI_DISCRETE);
        aimSectorCheck.setStyle("-fx-text-fill: #ecf0f1;");
        aimSectorCheck.setSelected(rlService.isUseAimSectorLearning());
        aimSectorCheck.setOnAction(e -> rlService.setUseAimSectorLearning(aimSectorCheck.isSelected()));

        box.getChildren().addAll(multiDiscreteCheck, aimSectorCheck);
        return box;
    }

    private VBox createStatsSection() {
        VBox box = card();
        box.getChildren().add(sectionTitle("📊  Status & Diagnostics"));

        statusLabel = new Label("Status: Idle");
        statusLabel.setStyle("-fx-text-fill: #2ecc71; -fx-font-weight: bold;");

        progressBar = new ProgressBar(0);
        progressBar.setPrefWidth(Double.MAX_VALUE);

        diagnosticLogArea = new TextArea();
        diagnosticLogArea.setEditable(false);
        diagnosticLogArea.setPrefHeight(200);
        diagnosticLogArea.setStyle("-fx-font-family: 'Consolas', monospace; -fx-font-size: 11px; -fx-control-inner-background: #1e1e1e; -fx-text-fill: #d4d4d4;");

        box.getChildren().addAll(statusLabel, progressBar, diagnosticLogArea);
        return box;
    }

    private HBox createActionSection() {
        HBox box = new HBox(10);
        box.setAlignment(Pos.CENTER);
        box.setPadding(new Insets(5, 0, 5, 0));

        trainButton = styledBtn("Start Training", "#e67e22");
        HintUtils.attachHint(trainButton, "Почати", "Запустити процес тренування нейромережі.");
        trainButton.setOnAction(e -> startTraining());

        stopButton = styledBtn("Stop", "#e74c3c");
        stopButton.setDisable(true);
        HintUtils.attachHint(stopButton, "Зупинити", "Зупинити тренування.");
        stopButton.setOnAction(e -> stopTraining());

        generateButton = styledBtn("Apply Best", "#9b59b6");
        HintUtils.attachHint(generateButton, "Застосувати", "Перенести найкращу згенеровану базу на ігрове поле.");
        generateButton.setOnAction(e -> applyBest());

        diagnosticButton = styledBtn("▶ Run Single Diagnostic Step", "#f39c12");
        HintUtils.attachHint(diagnosticButton, "Діагностика", "Запустити один крок тренування з виведенням детальної інформації.");
        diagnosticButton.setOnAction(e -> runDiagnosticStep());

        box.getChildren().addAll(trainButton, stopButton, generateButton, diagnosticButton);
        return box;
    }

    private void startTraining() {
        if (trainingRunning) return;
        
        String modelName = newModelField.getText().trim();
        if (modelName.isEmpty()) {
            if (currentModelName != null) modelName = currentModelName;
            else {
                showAlert("Please enter a model name.");
                return;
            }
        }
        currentModelName = modelName;
        
        trainingRunning = true;
        trainButton.setDisable(true);
        stopButton.setDisable(false);
        diagnosticButton.setDisable(true);
        statusLabel.setText("Status: Preparing...");
        diagnosticLogArea.clear();

        int ep = episodesSpinner.getValue();
        int steps = stepsSpinner.getValue();
        int epochs = epochsSpinner.getValue();
        double lw = logisticsSlider.getValue();
        double cw = costSlider.getValue();
        double rw = raidSlider.getValue();
        double ww = workingAreaSlider.getValue();
        
        rlService.setUseMultiDiscreteFlow(multiDiscreteCheck.isSelected());
        rlService.setLogFile(modelName);

        Thread trainingThread = new Thread(() -> {
            try {
                rlService.train(ep, steps, lw, cw, rw, ww, epochs,
                    this::onTrainingProgress, 
                    () -> Platform.runLater(() -> statusLabel.setText("Status: Epoch Complete")));
                
                Platform.runLater(() -> {
                    statusLabel.setText("Status: Finished");
                    trainingRunning = false;
                    trainButton.setDisable(false);
                    stopButton.setDisable(true);
                    diagnosticButton.setDisable(false);
                    updateStats();
                });
            } catch (Throwable ex) {
                java.io.StringWriter sw = new java.io.StringWriter();
                ex.printStackTrace(new java.io.PrintWriter(sw));
                String stack = sw.toString();

                Platform.runLater(() -> {
                    statusLabel.setText("Status: Error - " + ex.getMessage());
                    appendStatus("ERROR:\n" + stack);
                    trainingRunning = false;
                    trainButton.setDisable(false);
                    stopButton.setDisable(true);
                    diagnosticButton.setDisable(false);
                });
            }
        });
        trainingThread.setDaemon(true);
        trainingThread.start();
    }

    private void onTrainingProgress(TrainingMetrics m) {
        Platform.runLater(() -> {
            statusLabel.setText(String.format("Status: Training Epoch %d/%d, Ep %d/%d", 
                m.currentEpoch, m.totalEpochs, m.currentEpisodeInEpoch, m.totalEpisodesPerEpoch));
            
            int totalEps = m.totalEpochs * m.totalEpisodesPerEpoch;
            int doneEps = (m.currentEpoch - 1) * m.totalEpisodesPerEpoch + m.currentEpisodeInEpoch;
            progressBar.setProgress((double) doneEps / totalEps);

            StringBuilder sb = new StringBuilder();
            sb.append(String.format("[%02d:%02d:%02d] EPOCH %d/%d EP %d/%d\n", 
                java.time.LocalTime.now().getHour(), java.time.LocalTime.now().getMinute(), java.time.LocalTime.now().getSecond(),
                m.currentEpoch, m.totalEpochs, m.currentEpisodeInEpoch, m.totalEpisodesPerEpoch));
            sb.append(String.format("  Best Score: %.4f | Avg Eval: %.4f\n", m.bestScore, m.avgEvalScore));
            sb.append(String.format("  Epsilon:    %.4f | Loss:     %.6f\n", m.epsilon, m.lastTrainLoss));
            sb.append(String.format("  Reward Sum: %.2f | Invalid:  %.1f%%\n", m.currentEpisodeStepReward, m.invalidActionRate * 100));
            sb.append(String.format("  Best Base:  Blocks %d, TC %s, Doors %d\n", m.bestBaseBlocks, m.bestBaseHasTC ? "YES" : "NO", m.bestBaseDoors));
            sb.append("------------------------------------------\n");
            
            diagnosticLogArea.setText(sb.toString() + diagnosticLogArea.getText());
            
            updateSummaryUI();
        });
    }

    private void updateSummaryUI() {
        if (rlService == null) return;
        
        bestScoreLbl.setText(String.format("%.4f", rlService.getBestScore()));
        avgEvalScoreLbl.setText(String.format("%.4f", rlService.getAvgEvalScore()));
        avgStepRewardLbl.setText(String.format("%.4f", rlService.getAvgReward()));
        invalidRateLbl.setText(String.format("%.1f%%", rlService.getInvalidActionRate() * 100));
        memorySizeLbl.setText(String.valueOf(rlService.getMemorySize()));
        episodesLbl.setText(String.valueOf(rlService.getEpisodesTrained()));
        epsilonLbl.setText(String.format("%.4f", rlService.getEpsilon()));
        lossLbl.setText(String.format("%.6f", rlService.getLastTrainLoss()));
        
        String bestBase = String.format("B: %d, TC: %s, D: %d", 
            rlService.getBestBaseBlocks(), 
            rlService.isBestBaseHasTC() ? "Yes" : "No", 
            rlService.getBestBaseDoors());
        bestBaseLbl.setText(bestBase);
        
        ramUsageLbl.setText(rlService.getRamUsage());
        timerLbl.setText(rlService.getFormattedTrainingTime());
    }

    private void stopTraining() {
        if (!trainingRunning) return;
        rlService.requestStop();
        statusLabel.setText("Status: Stop Requested...");
    }

    private void applyBest() {
        GridModel best = rlService.getBestGridModel();
        if (best == null || best.getAllBlocks().isEmpty()) {
            showAlert("No successful base trained yet!");
            return;
        }
        
        mainGrid.clear();
        for (BuildingBlock b : best.getAllBlocks()) {
            mainGrid.addBlockSilent(cloneBlock(b));
        }
        mainGrid.finalizeLoad();
        if (refreshCallback != null) refreshCallback.run();
        if (gameCanvas != null) {
            gameCanvas.invalidateCache();
            gameCanvas.draw();
        }
        
        appendStatus("Applied best trained layout (Score: " + String.format("%.2f", rlService.getBestScore()) + ")");
    }

    private void runDiagnosticStep() {
        if (trainingRunning) return;
        
        appendStatus("--- Starting Diagnostic Step ---");
        
        com.rustbuilder.ai.rl.multidiscrete.MultiDiscretePhaseContext context = 
            new com.rustbuilder.ai.rl.multidiscrete.MultiDiscretePhaseContext(
                mainGrid, true, false, 0, 1
            );
            
        com.rustbuilder.ai.rl.multidiscrete.MultiDiscretePhasePolicy policy = rlService.getMultiDiscretePolicy();
        if (policy == null) {
            showAlert("No active RL policy found.");
            return;
        }
        
        com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteStateObserver guiObserver = 
            new com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteStateObserver() {
                @Override
                public void observePhase(com.rustbuilder.ai.rl.multidiscrete.MultiDiscretePhaseContext ctx, String phaseName, int phaseIdx, int choice) {
                    appendStatus(String.format("  [%s] choice=%d", phaseName, choice));
                }
                
                @Override
                public void onActionAssembled(com.rustbuilder.ai.rl.multidiscrete.MultiDiscretePhaseContext ctx, com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteAction action) {
                    appendStatus("  Action Assembled: " + action);
                }
            };
        
        Thread t = new Thread(() -> {
            try {
                com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteAction action = policy.chooseAction(context, guiObserver);
                if (action == null || action.getTypeIndex() == -1) {
                    appendStatus("  No action selected (STOP).");
                    return;
                }
                
                com.rustbuilder.ai.ea.BaseGenome.BuildAction legacy = 
                    com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteActionMapper.toBuildAction(action);
                    
                com.rustbuilder.ai.rl.RLTrainingService.PlacementResult res = rlService.placeBlock(mainGrid, legacy);
                
                Platform.runLater(() -> {
                    if (res.inserted) {
                        appendStatus("  Placement successful: " + legacy.actionType);
                    } else {
                        appendStatus("  Placement FAILED: " + res.failReason);
                    }
                    if (refreshCallback != null) refreshCallback.run();
                    if (gameCanvas != null) {
                        gameCanvas.invalidateCache();
                        gameCanvas.draw();
                    }
                });
            } catch (Exception ex) {
                appendStatus("  Diagnostic Error: " + ex.toString());
            }
        });
        t.setDaemon(true);
        t.start();
    }

    private BuildingBlock cloneBlock(BuildingBlock b) {
        BuildingBlock clone = null;
        double x = b.getX();
        double y = b.getY();
        int z = b.getZ();
        double rot = b.getRotation();
        
        if (b instanceof com.rustbuilder.model.structure.Foundation) clone = new com.rustbuilder.model.structure.Foundation(x, y, z);
        else if (b instanceof com.rustbuilder.model.structure.TriangleFoundation) clone = new com.rustbuilder.model.structure.TriangleFoundation(x, y, z, rot);
        else if (b instanceof com.rustbuilder.model.structure.Wall) {
            com.rustbuilder.model.structure.Wall w = (com.rustbuilder.model.structure.Wall)b;
            clone = new com.rustbuilder.model.structure.Wall(x, y, z, w.getOrientation());
            ((com.rustbuilder.model.structure.Wall)clone).setType(w.getType());
            if (w.getType() == com.rustbuilder.model.core.BuildingType.DOORWAY) ((com.rustbuilder.model.structure.Wall)clone).setDoorType(w.getDoorType());
        }
        else if (b instanceof com.rustbuilder.model.structure.Floor) clone = new com.rustbuilder.model.structure.Floor(x, y, z, rot);
        else if (b instanceof com.rustbuilder.model.structure.TriangleFloor) clone = new com.rustbuilder.model.structure.TriangleFloor(x, y, z, rot);
        else if (b instanceof com.rustbuilder.model.structure.Door) clone = new com.rustbuilder.model.structure.Door(x, y, z, ((com.rustbuilder.model.structure.Door)b).getOrientation(), ((com.rustbuilder.model.structure.Door)b).getDoorType());
        else if (b instanceof com.rustbuilder.model.deployable.ToolCupboard) clone = new com.rustbuilder.model.deployable.ToolCupboard(x, y, z, rot);
        else if (b instanceof com.rustbuilder.model.deployable.Workbench) clone = new com.rustbuilder.model.deployable.Workbench(x, y, z, rot);
        else if (b instanceof com.rustbuilder.model.deployable.LootRoom) clone = new com.rustbuilder.model.deployable.LootRoom(x, y, z, rot);

        if (clone != null) {
            clone.setRotation(rot);
            clone.setTier(b.getTier());
        }
        return clone;
    }

    private void refreshModelList() {
        modelComboBox.getItems().clear();
        modelComboBox.getItems().addAll(RLModelManager.listModels());
    }

    private void saveModel() {
        String name = newModelField.getText().trim();
        if (name.isEmpty()) {
            showAlert("Enter a model name.");
            return;
        }
        try {
            double lw = logisticsSlider.getValue();
            double cw = costSlider.getValue();
            double rw = raidSlider.getValue();
            double ww = workingAreaSlider.getValue();
            
            RLModel snapshot = RLModelManager.createSnapshot(name, rlService, lw, cw, rw, ww);
            RLModelManager.saveModel(snapshot, rlService);
            refreshModelList();
            appendStatus("Model '" + name + "' saved successfully.");
        } catch (IOException ex) {
            showAlert("Save failed: " + ex.getMessage());
        }
    }

    private void loadModel() {
        String name = modelComboBox.getValue();
        if (name == null) return;
        try {
            rlService.resetRuntimeState(); // Hard reset derived stats
            RLModel meta = RLModelManager.loadModel(name, rlService);
            RLModelManager.restoreFromModel(rlService, meta); // Also calls resetRuntimeState() but double checking
            
            currentModelName = name;
            newModelField.setText(name);
            
            logisticsSlider.setValue(meta.logisticsWeight);
            costSlider.setValue(meta.costWeight);
            raidSlider.setValue(meta.raidWeight);
            workingAreaSlider.setValue(meta.workingAreaWeight);
            
            if (meta.rewardConfig != null) {
                rlService.setRewardConfig(meta.rewardConfig.clone());
            }
            syncRewardUIFromConfig();
            
            multiDiscreteCheck.setSelected(rlService.isUseMultiDiscreteFlow());
            updateStats();
            updateSummaryUI(); // Important: refresh UI after load
            
            appendStatus("Model loaded successfully: " + meta.toString());
        } catch (IOException | ClassNotFoundException ex) {
            showAlert("Load failed: " + ex.getMessage());
        }
    }

    private void deleteModel() {
        String name = modelComboBox.getValue();
        if (name == null) return;
        if (RLModelManager.deleteModel(name)) {
            refreshModelList();
            appendStatus("Model '" + name + "' deleted.");
        }
    }

    private void updateStats() {
        if (rlService == null) return;
        
        String policyName = "None";
        if (rlService.getMultiDiscretePolicy() != null) {
            policyName = rlService.getMultiDiscretePolicy().getClass().getSimpleName();
        }
        
        boolean hasBaseline = rlService.getMultiDiscreteLearningProvider() != null;
        
        appendStatus(String.format("Active Policy: %s", policyName));
        appendStatus(String.format("Q-Table Baseline: %s", hasBaseline ? "YES" : "NO"));
        appendStatus(String.format("Best Score: %.4f | Episodes: %d", rlService.getBestScore(), rlService.getEpisodesTrained()));
    }

    public void show() {
        refreshModelList();
        multiDiscreteCheck.setSelected(rlService.isUseMultiDiscreteFlow());
        updateStats();
        updateSummaryUI();
        reportMode();
        dialogStage.show();
    }

    public void appendStatus(String text) {
        Platform.runLater(() -> {
            diagnosticLogArea.appendText(text + "\n");
        });
    }

    public void reportMode() {
        boolean activeMD = rlService.isUseMultiDiscreteFlow();
        boolean learnsMD = rlService.isUseMultiDiscreteLearning();
        
        String report = String.format("[RL UI REPORT] mode=%s, training=%s", 
            activeMD ? "MULTI_DISCRETE" : "LEGACY",
            learnsMD ? "ENABLED" : "DISABLED");
        
        appendStatus(report);
    }

    public void cleanup() {
        if (trainingRunning) {
            stopTraining();
        }
    }

    private VBox createSummarySection() {
        VBox box = card();
        box.getChildren().add(sectionTitle("📈  Training Summary"));

        GridPane grid = new GridPane();
        grid.setHgap(20);
        grid.setVgap(8);
        grid.setPadding(new Insets(5, 5, 5, 5));

        // Column 1
        bestScoreLbl = valueLabel("0.0000");
        avgEvalScoreLbl = valueLabel("0.0000");
        avgStepRewardLbl = valueLabel("0.0000");
        invalidRateLbl = valueLabel("0.0%");
        timerLbl = valueLabel("00:00:00");
        ramUsageLbl = valueLabel("0 MB");

        addSummaryRow(grid, 0, "Best Score:", bestScoreLbl, HintKey.BEST_SCORE);
        addSummaryRow(grid, 1, "Avg Eval Score:", avgEvalScoreLbl, HintKey.AVG_EVAL_SCORE);
        addSummaryRow(grid, 2, "Avg Step Reward:", avgStepRewardLbl, HintKey.AVG_STEP_REWARD);
        addSummaryRow(grid, 3, "Invalid Rate:", invalidRateLbl, HintKey.INVALID_RATE);
        addSummaryRow(grid, 4, "Training Time:", timerLbl, HintKey.TRAINING_TIME);
        addSummaryRow(grid, 5, "RAM Usage:", ramUsageLbl, HintKey.RAM_USAGE);

        // Column 2
        episodesLbl = valueLabel("0");
        epsilonLbl = valueLabel("1.0000");
        lossLbl = valueLabel("0.000000");
        memorySizeLbl = valueLabel("0");
        bestBaseLbl = valueLabel("Blocks: 0, TC: No");

        addSummaryRow(grid, 0, 2, "Total Episodes:", episodesLbl, HintKey.TOTAL_EPISODES);
        addSummaryRow(grid, 1, 2, "Current Epsilon:", epsilonLbl, HintKey.EPSILON);
        addSummaryRow(grid, 2, 2, "Train Loss:", lossLbl, HintKey.TRAIN_LOSS);
        addSummaryRow(grid, 3, 2, "Replay Memory:", memorySizeLbl, HintKey.REPLAY_MEMORY);
        addSummaryRow(grid, 4, 2, "Best Base:", bestBaseLbl, HintKey.BEST_BASE);

        box.getChildren().add(grid);
        return box;
    }

    private void addSummaryRow(GridPane grid, int row, String labelText, Label valueLbl, HintKey hint) {
        addSummaryRow(grid, row, 0, labelText, valueLbl, hint);
    }

    private void addSummaryRow(GridPane grid, int row, int colOffset, String labelText, Label valueLbl, HintKey hint) {
        Label l = bodyLabel(labelText);
        l.setMinWidth(100);
        if (hint != null) {
            HintUtils.attachHint(l, hint);
            HintUtils.attachHint(valueLbl, hint);
        }
        grid.add(l, colOffset, row);
        grid.add(valueLbl, colOffset + 1, row);
    }

    private VBox createRewardConfigSection() {
        VBox box = card();
        box.getChildren().add(sectionTitle("🎯  Reward / Penalty Tuning"));

        TabPane tabs = new TabPane();
        tabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE);
        tabs.setStyle("-fx-background-color: transparent;");

        tabs.getTabs().addAll(
            new Tab("Step Rewards", createStepRewardsGrid()),
            new Tab("Step Penalties", createStepPenaltiesGrid()),
            new Tab("Final Eval", createEvalRewardsGrid())
        );

        HBox btnBox = new HBox(8);
        btnBox.setAlignment(Pos.CENTER_RIGHT);
        btnBox.setPadding(new Insets(5, 0, 0, 0));

        Button applyBtn = styledBtn("Apply Changes", "#27ae60");
        applyBtn.setOnAction(e -> applyRewardConfig());

        Button resetBtn = styledBtn("Reset Defaults", "#7f8c8d");
        resetBtn.setOnAction(e -> resetRewardConfig());

        btnBox.getChildren().addAll(resetBtn, applyBtn);

        box.getChildren().addAll(tabs, btnBox);
        
        syncRewardUIFromConfig();
        
        return box;
    }

    private GridPane createStepRewardsGrid() {
        GridPane grid = rewardGrid();
        int row = 0;
        addRewardRow(grid, row++, "Розміщення бази", "basePlacementReward", "Нагорода за будь-яке валідне розміщення блоку.");
        addRewardRow(grid, row++, "З'єднання (Socket)", "socketConnectionReward", "Нагорода за кожне з'єднання між блоками.");
        addRewardRow(grid, row++, "Множник стабільності", "stabilityRewardMult", "Множник для структурної стабільності (0-1).");
        addRewardRow(grid, row++, "Розміщення шафи", "tcPlacementBonus", "Одноразовий бонус за розміщення шафи (TC).");
        addRewardRow(grid, row++, "Бонус за предмети", "secondaryDeployableBonus", "Бонус за розміщення верстака або лутової.");
        addRewardRow(grid, row++, "Компактність", "spatialCompactnessBonus", "Нагорода за будівництво впритул до існуючих блоків.");
        addRewardRow(grid, row++, "Штраф за розсіювання", "spatialScatteredPenalty", "Штраф за початок будівництва надто далеко.");
        addRewardRow(grid, row++, "Штраф за від'єднання", "disconnectedSegmentPenalty", "Штраф за розміщення блоку без з'єднання з базою.");
        return grid;
    }

    private GridPane createStepPenaltiesGrid() {
        GridPane grid = rewardGrid();
        int row = 0;
        addRewardRow(grid, row++, "Без опори", "penaltyNoSupport", "Штраф за розміщення без фундаменту або стабільності.");
        addRewardRow(grid, row++, "Поганий сокет", "penaltyBadSocket", "Штраф за неправильне вирівнювання або тип з'єднання.");
        addRewardRow(grid, row++, "Колізія", "penaltyCollision", "Штраф за перекриття з існуючими блоками.");
        addRewardRow(grid, row++, "Поверх. обмеження", "penaltyFloorConstraint", "Штраф за порушення специфічних правил поверхів.");
        addRewardRow(grid, row++, "Загальна помилка", "penaltyGenericInvalid", "Штраф за інші помилки розміщення.");
        addRewardRow(grid, row++, "Штраф за недобудовані", "stopUnbuiltBlockPenalty", "Штраф за кожен недобудований блок до максимального ліміту при виборі дії STOP.");
        return grid;
    }

    private GridPane createEvalRewardsGrid() {
        GridPane grid = rewardGrid();
        int row = 0;
        addRewardRow(grid, row++, "Множник рахунку", "finalScoreMultiplier", "Множник для базової оцінки HouseEvaluator.");
        addRewardRow(grid, row++, "Бонус логістики", "logisticsBonus", "Бонус, якщо база має хорошу логістику (скрині/печі).");
        addRewardRow(grid, row++, "Бонус рейду", "raidBonusMultiplier", "Множник для оцінки стійкості до рейду.");
        addRewardRow(grid, row++, "Зв'язність бази", "connectivityBonus", "Бонус, якщо всі блоки утворюють єдину цілісну структуру.");
        addRewardRow(grid, row++, "Штраф фрагментації", "fragmentBasePenalty", "Штраф, якщо структура фрагментована.");
        addRewardRow(grid, row++, "Штраф за дострокове", "earlyStopPenaltyMult", "Штраф за кожен невикористаний крок, якщо тренування зупинилось рано.");
        addRewardRow(grid, row++, "Штраф за крах", "totalFailurePenalty", "Штраф, якщо побудована база занадто мала або невалідна.");
        return grid;
    }

    private GridPane rewardGrid() {
        GridPane grid = new GridPane();
        grid.setHgap(8);
        grid.setVgap(4);
        grid.setPadding(new Insets(10));
        return grid;
    }

    private void addRewardRow(GridPane grid, int row, String labelText, String fieldName, String tooltip) {
        Label label = bodyLabel(labelText);
        if (tooltip != null) {
            HintUtils.attachHint(label, labelText, tooltip);
        }
        Spinner<Double> spinner = new Spinner<>(-100.0, 100.0, 0.0, 0.01);
        spinner.setEditable(true);
        spinner.setPrefWidth(90);
        if (tooltip != null) {
            HintUtils.attachHint(spinner.getEditor(), labelText, tooltip);
        }
        
        rewardSpinners.put(fieldName, spinner);
        
        grid.add(label, 0, row);
        grid.add(spinner, 1, row);
    }

    private void syncRewardUIFromConfig() {
        com.rustbuilder.ai.rl.RLRewardConfig config = rlService.getRewardConfig();
        if (config == null) return;

        updateSpinner("basePlacementReward", config.basePlacementReward);
        updateSpinner("socketConnectionReward", config.socketConnectionReward);
        updateSpinner("stabilityRewardMult", config.stabilityRewardMult);
        updateSpinner("tcPlacementBonus", config.tcPlacementBonus);
        updateSpinner("secondaryDeployableBonus", config.secondaryDeployableBonus);
        updateSpinner("spatialCompactnessBonus", config.spatialCompactnessBonus);
        updateSpinner("spatialScatteredPenalty", config.spatialScatteredPenalty);
        updateSpinner("disconnectedSegmentPenalty", config.disconnectedSegmentPenalty);

        updateSpinner("penaltyNoSupport", config.penaltyNoSupport);
        updateSpinner("penaltyBadSocket", config.penaltyBadSocket);
        updateSpinner("penaltyCollision", config.penaltyCollision);
        updateSpinner("penaltyFloorConstraint", config.penaltyFloorConstraint);
        updateSpinner("penaltyGenericInvalid", config.penaltyGenericInvalid);
        updateSpinner("stopUnbuiltBlockPenalty", config.stopUnbuiltBlockPenalty);

        updateSpinner("finalScoreMultiplier", config.finalScoreMultiplier);
        updateSpinner("logisticsBonus", config.logisticsBonus);
        updateSpinner("raidBonusMultiplier", config.raidBonusMultiplier);
        updateSpinner("connectivityBonus", config.connectivityBonus);
        updateSpinner("fragmentBasePenalty", config.fragmentBasePenalty);
        updateSpinner("earlyStopPenaltyMult", config.earlyStopPenaltyMult);
        updateSpinner("totalFailurePenalty", config.totalFailurePenalty);
    }

    private void updateSpinner(String key, double val) {
        Spinner<Double> s = rewardSpinners.get(key);
        if (s != null) {
            s.getValueFactory().setValue(val);
        }
    }

    private void applyRewardConfig() {
        com.rustbuilder.ai.rl.RLRewardConfig config = rlService.getRewardConfig();
        if (config == null) config = new com.rustbuilder.ai.rl.RLRewardConfig();
        
        config.basePlacementReward = rewardSpinners.get("basePlacementReward").getValue();
        config.socketConnectionReward = rewardSpinners.get("socketConnectionReward").getValue();
        config.stabilityRewardMult = rewardSpinners.get("stabilityRewardMult").getValue();
        config.tcPlacementBonus = rewardSpinners.get("tcPlacementBonus").getValue();
        config.secondaryDeployableBonus = rewardSpinners.get("secondaryDeployableBonus").getValue();
        config.spatialCompactnessBonus = rewardSpinners.get("spatialCompactnessBonus").getValue();
        config.spatialScatteredPenalty = rewardSpinners.get("spatialScatteredPenalty").getValue();
        config.disconnectedSegmentPenalty = rewardSpinners.get("disconnectedSegmentPenalty").getValue();

        config.penaltyNoSupport = rewardSpinners.get("penaltyNoSupport").getValue();
        config.penaltyBadSocket = rewardSpinners.get("penaltyBadSocket").getValue();
        config.penaltyCollision = rewardSpinners.get("penaltyCollision").getValue();
        config.penaltyFloorConstraint = rewardSpinners.get("penaltyFloorConstraint").getValue();
        config.penaltyGenericInvalid = rewardSpinners.get("penaltyGenericInvalid").getValue();
        config.stopUnbuiltBlockPenalty = rewardSpinners.get("stopUnbuiltBlockPenalty").getValue();

        config.finalScoreMultiplier = rewardSpinners.get("finalScoreMultiplier").getValue();
        config.logisticsBonus = rewardSpinners.get("logisticsBonus").getValue();
        config.raidBonusMultiplier = rewardSpinners.get("raidBonusMultiplier").getValue();
        config.connectivityBonus = rewardSpinners.get("connectivityBonus").getValue();
        config.fragmentBasePenalty = rewardSpinners.get("fragmentBasePenalty").getValue();
        config.earlyStopPenaltyMult = rewardSpinners.get("earlyStopPenaltyMult").getValue();
        config.totalFailurePenalty = rewardSpinners.get("totalFailurePenalty").getValue();
        
        rlService.setRewardConfig(config);
        appendStatus("Reward Configuration applied to Service.");
    }

    private void resetRewardConfig() {
        rlService.setRewardConfig(com.rustbuilder.ai.rl.RLRewardConfig.createDefault());
        syncRewardUIFromConfig();
        appendStatus("Reward Configuration reset to defaults.");
    }

    private void focusDefault() {
        newModelField.requestFocus();
    }

    // Helper UI builders
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

    private void showAlert(String msg) {
        Alert alert = new Alert(Alert.AlertType.WARNING);
        alert.setTitle("RL AI Generator");
        alert.setHeaderText(null);
        alert.setContentText(msg);
        alert.showAndWait();
    }
}
