package com.rustbuilder.ui;

import java.io.IOException;

import com.rustbuilder.ai.core.TrainingMetrics;
import com.rustbuilder.ai.rl.RLModelManager;
import com.rustbuilder.ai.rl.RLModelManager.RLModel;
import com.rustbuilder.ai.rl.RLTrainingConfig;
import com.rustbuilder.ai.rl.RLTrainingService;
import com.rustbuilder.ai.rl.supervisor.ExternalCommandLlmSupervisor;
import com.rustbuilder.ai.rl.supervisor.LlmSupervisorApplyMode;
import com.rustbuilder.ai.rl.supervisor.LlmSupervisorConfig;
import com.rustbuilder.ai.rl.supervisor.NoOpLlmSupervisor;
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
import javafx.scene.control.PasswordField;
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

    /** API-ключ зберігається в пам'яті до закриття програми. */
    private static String savedApiKey = "";

    // UI Components
    private ComboBox<String> modelComboBox;
    private ComboBox<RLTrainingService.EncoderMode> encoderModeComboBox;
    private TextField newModelField;
    private boolean syncingEncoderMode = false;
    
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
    
    private Spinner<Integer> episodesSpinner;
    private Spinner<Integer> stepsSpinner;
    private Spinner<Integer> epochsSpinner;
    private TextField trainingTimeLimitField;
    private ComboBox<RLTrainingService.TrainingLoadProfile> loadProfileComboBox;
    private Label loadProfileLabel;
    private CheckBox supervisorEnabledCheck;
    private Spinner<Integer> supervisorIntervalSpinner;
    private PasswordField supervisorApiKeyField;
    private TextField supervisorCommandField;
    private ComboBox<LlmSupervisorApplyMode> supervisorApplyModeComboBox;
    private Button supervisorApplyPendingButton;
    private CheckBox use2dCnnCheck;
    
    // Legacy multiDiscreteCheck removed
    private ProgressBar progressBar;
    private Label statusLabel;
    private TextArea diagnosticLogArea;
    private TextArea llmLogArea;
    
    private Button trainButton;
    private Button stopButton;
    private Button generateButton;
    private Button bestRewardButton;
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

        TabPane tabPane = new TabPane();
        tabPane.setStyle("-fx-background: #2b2b2b; -fx-background-color: #2b2b2b;");

        Tab manualTab = new Tab("Manual Control");
        manualTab.setClosable(false);
        VBox manualBox = new VBox(10);
        manualBox.setPadding(new Insets(14));
        manualBox.getChildren().addAll(
            createModelSection(),
            new Separator(),
            createPrioritySection(),
            new Separator(),
            createTrainingSection(),
            new Separator(),
            createLoadControlSection(),
            new Separator(),
            createStatsSection(),
            new Separator(),
            createSummarySection(),
            new Separator(),
            createActionSection()
        );
        manualTab.setContent(new ScrollPane(manualBox));

        Tab advancedTab = new Tab("Advanced Settings");
        advancedTab.setClosable(false);
        VBox advancedBox = new VBox(10);
        advancedBox.setPadding(new Insets(14));
        advancedBox.getChildren().addAll(
            createRewardConfigSection(),
            new Separator(),
            createExperimentalSection()
        );
        advancedTab.setContent(new ScrollPane(advancedBox));

        Tab llmTab = new Tab("LLM Autopilot");
        llmTab.setClosable(false);
        VBox llmBox = new VBox(10);
        llmBox.setPadding(new Insets(14));
        llmBox.getChildren().addAll(
            createSupervisorSection(),
            new Separator(),
            createLlmLogSection()
        );
        llmTab.setContent(new ScrollPane(llmBox));

        tabPane.getTabs().addAll(manualTab, advancedTab, llmTab);
        
        // Disable scrollpane background on the internal scrollpanes
        for (Tab t : tabPane.getTabs()) {
            ((ScrollPane) t.getContent()).setFitToWidth(true);
            ((ScrollPane) t.getContent()).setStyle("-fx-background: #2b2b2b; -fx-background-color: #2b2b2b;");
            ((ScrollPane) t.getContent()).setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        }

        VBox mainContent = new VBox(tabPane);
        mainContent.setStyle("-fx-background-color: #2b2b2b;");
        VBox.setVgrow(tabPane, Priority.ALWAYS);
        Scene scene = new Scene(mainContent, 550, 800);
        dialogStage.setScene(scene);
        this.rlService.setSupervisorLogCallback(this::appendLlmStatus);
        
        dialogStage.setOnCloseRequest(e -> cleanup());
    }

    /**
     * Static helper for opening the dialog, compatible with existing call sites.
     */
    public static void showDialog(Stage owner, RLTrainingService service, GridModel grid, Runnable refresh) {
        RLGeneratorDialog dialog = new RLGeneratorDialog(owner, service, grid, refresh, null);
        dialog.show();
    }

    public static void showDialog(Stage owner, RLTrainingService service, GridModel grid, Runnable refresh, GameCanvas gameCanvas) {
        RLGeneratorDialog dialog = new RLGeneratorDialog(owner, service, grid, refresh, gameCanvas);
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

        safeZoneLabel = valueLabel("Safe Zone: 0.50");
        HintUtils.attachHint(safeZoneLabel, HintKey.SAFE_ZONE);
        safeZoneSlider = buildSlider(0, 2, 0.5, 0.5);
        safeZoneSlider.valueProperty().addListener((o, ov, nv) -> safeZoneLabel.setText(String.format("Safe Zone: %.2f", nv)));

        grid.add(logLabel, 0, 0); grid.add(logisticsSlider, 1, 0);
        grid.add(costLabel, 0, 1); grid.add(costSlider, 1, 1);
        grid.add(raidLabel, 0, 2); grid.add(raidSlider, 1, 2);
        grid.add(workingAreaLabel, 0, 3); grid.add(workingAreaSlider, 1, 3);
        grid.add(safeZoneLabel, 0, 4); grid.add(safeZoneSlider, 1, 4);

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

        Label timeLimitLabel = bodyLabel("Time limit:");
        HintUtils.attachHint(timeLimitLabel, "Training time limit", "Optional wall-clock budget. Examples: 30m, 2h, 01:30. Empty or 0 means no time limit.");
        grid.add(timeLimitLabel, 0, 3);
        trainingTimeLimitField = new TextField();
        trainingTimeLimitField.setPromptText("0, 30m, 2h, 01:30");
        HintUtils.attachHint(trainingTimeLimitField, "Training time limit", "Training stops cleanly after this duration, and the LLM supervisor sees elapsed time, current time, remaining time, and deadline.");
        grid.add(trainingTimeLimitField, 1, 3);

        box.getChildren().add(grid);
        return box;
    }

    private VBox createLoadControlSection() {
        VBox box = card();
        box.getChildren().add(sectionTitle("Training Load"));

        HBox profileRow = new HBox(8);
        profileRow.setAlignment(Pos.CENTER_LEFT);

        Label profileLabel = bodyLabel("Load profile:");
        HintUtils.attachHint(profileLabel, "Training load", "Controls how aggressively RL training uses CPU/GPU time. Can be changed while training is running.");

        loadProfileComboBox = new ComboBox<>();
        loadProfileComboBox.getItems().setAll(RLTrainingService.TrainingLoadProfile.values());
        loadProfileComboBox.setValue(rlService.getTrainingLoadProfile());
        loadProfileComboBox.setPrefWidth(190);
        HintUtils.attachHint(loadProfileComboBox, "Training load", "Low adds more pauses for games/heavy apps. Maximum removes throttling for overnight training.");

        loadProfileLabel = valueLabel(loadProfileText(loadProfileComboBox.getValue()));
        HBox.setHgrow(loadProfileLabel, Priority.ALWAYS);

        loadProfileComboBox.setOnAction(e -> applyLoadProfileFromUI(true));

        profileRow.getChildren().addAll(profileLabel, loadProfileComboBox);
        box.getChildren().addAll(profileRow, loadProfileLabel);
        return box;
    }

    private VBox createExperimentalSection() {
        VBox box = card();
        box.getChildren().add(sectionTitle("🧪  Experimental Settings"));

        HBox encoderRow = new HBox(8);
        encoderRow.setAlignment(Pos.CENTER_LEFT);

        Label encoderLabel = bodyLabel("State Encoder:");
        HintUtils.attachHint(encoderLabel, "State Encoder", "Select the state encoder architecture used for new RL training runs.");

        encoderModeComboBox = new ComboBox<>();
        encoderModeComboBox.getItems().setAll(RLTrainingService.EncoderMode.values());
        encoderModeComboBox.setValue(rlService.getEncoderMode());
        encoderModeComboBox.setPrefWidth(170);
        HintUtils.attachHint(encoderModeComboBox, "State Encoder", "V1 uses the legacy voxel encoder. V2 uses bucketed voxels. V3 uses the hybrid voxel/global encoder.");
        encoderModeComboBox.setOnAction(e -> handleEncoderModeSelection());

        encoderRow.getChildren().addAll(encoderLabel, encoderModeComboBox);

        CheckBox aimSectorCheck = new CheckBox("Learn Aim Sector (Phase 5)");
        HintUtils.attachHint(aimSectorCheck, HintKey.MULTI_DISCRETE);
        aimSectorCheck.setStyle("-fx-text-fill: #ecf0f1;");
        aimSectorCheck.setSelected(rlService.isUseAimSectorLearning());
        aimSectorCheck.setOnAction(e -> rlService.setUseAimSectorLearning(aimSectorCheck.isSelected()));

        use2dCnnCheck = new CheckBox("Use 2D CNN Architecture");
        HintUtils.attachHint(use2dCnnCheck, "Use 2D CNN Architecture", "Flattens the 3D voxel input along the Z-axis, creating a 2D map with dense channels. Potentially faster and easier to train.");
        use2dCnnCheck.setStyle("-fx-text-fill: #ecf0f1;");
        use2dCnnCheck.setSelected(rlService.isUse2dCnn());
        use2dCnnCheck.setOnAction(e -> rlService.setUse2dCnn(use2dCnnCheck.isSelected()));

        box.getChildren().addAll(encoderRow, aimSectorCheck, use2dCnnCheck);
        return box;
    }

    private VBox createSupervisorSection() {
        VBox box = card();
        box.getChildren().add(sectionTitle("LLM Supervisor"));

        LlmSupervisorConfig config = rlService.getSupervisorConfig();

        supervisorEnabledCheck = new CheckBox("Enable LLM supervisor");
        supervisorEnabledCheck.setStyle("-fx-text-fill: #ecf0f1;");
        supervisorEnabledCheck.setSelected(config.isEnabled());
        HintUtils.attachHint(supervisorEnabledCheck, "LLM Supervisor", "Allows a configured LLM supervisor to review compact training metrics between episodes.");

        HBox intervalRow = new HBox(8);
        intervalRow.setAlignment(Pos.CENTER_LEFT);
        Label intervalLabel = bodyLabel("Call every episodes:");
        HintUtils.attachHint(intervalLabel, "Supervisor frequency", "How often the LLM supervisor is called. Default is every 1000 trained episodes.");

        supervisorIntervalSpinner = new Spinner<>(1, 1_000_000,
            config.getCallIntervalEpisodes() > 0
                ? config.getCallIntervalEpisodes()
                : LlmSupervisorConfig.DEFAULT_CALL_INTERVAL_EPISODES,
            100);
        supervisorIntervalSpinner.setEditable(true);
        supervisorIntervalSpinner.setPrefWidth(130);
        HintUtils.attachHint(supervisorIntervalSpinner.getEditor(), "Supervisor frequency", "Default: 1000 episodes. Lower values react faster but can slow training.");

        HBox modeRow = new HBox(8);
        modeRow.setAlignment(Pos.CENTER_LEFT);
        Label modeLabel = bodyLabel("Apply mode:");
        HintUtils.attachHint(modeLabel, "Supervisor apply mode", "Auto Apply changes training immediately. Log Only records decisions without applying them.");
        supervisorApplyModeComboBox = new ComboBox<>();
        supervisorApplyModeComboBox.getItems().setAll(LlmSupervisorApplyMode.values());
        supervisorApplyModeComboBox.setValue(config.getApplyMode());
        supervisorApplyModeComboBox.setPrefWidth(170);
        HintUtils.attachHint(supervisorApplyModeComboBox, "Supervisor apply mode", "Use Log Only first when testing a new LLM prompt or command.");

        HBox apiKeyRow = new HBox(8);
        apiKeyRow.setAlignment(Pos.CENTER_LEFT);
        Label apiKeyLabel = bodyLabel("API key:");
        HintUtils.attachHint(apiKeyLabel, "Supervisor API key", "Optional key passed only to the external command environment. It is not saved in model metadata.");
        supervisorApiKeyField = new PasswordField();
        supervisorApiKeyField.setPromptText("Optional Gemini/API key...");
        // Restore API key saved in memory for this session
        if (!savedApiKey.isEmpty()) {
            supervisorApiKeyField.setText(savedApiKey);
        }
        HBox.setHgrow(supervisorApiKeyField, Priority.ALWAYS);
        HintUtils.attachHint(supervisorApiKeyField, "Supervisor API key", "For Gemini, this becomes GEMINI_API_KEY and GOOGLE_API_KEY for the wrapper process.");

        Button saveApiKeyBtn = styledBtn("💾", "#27ae60");
        saveApiKeyBtn.setMinWidth(36);
        HintUtils.attachHint(saveApiKeyBtn, "Зберегти API-ключ", "Зберігає ключ у пам'яті до закриття програми. Ключ не записується на диск.");
        saveApiKeyBtn.setOnAction(e -> {
            savedApiKey = supervisorApiKeyField.getText();
            saveApiKeyBtn.setText("✅");
            javafx.animation.PauseTransition pause = new javafx.animation.PauseTransition(javafx.util.Duration.seconds(2));
            pause.setOnFinished(ev -> saveApiKeyBtn.setText("💾"));
            pause.play();
        });

        HBox commandRow = new HBox(8);
        commandRow.setAlignment(Pos.CENTER_LEFT);
        Label commandLabel = bodyLabel("Command:");
        HintUtils.attachHint(commandLabel, "Supervisor command", "Optional external command. It receives observation JSON on stdin and returns decision JSON on stdout.");
        String defaultCommand = "powershell -ExecutionPolicy Bypass -File scripts/llm_supervisor_gemini.ps1";
        String existingCommand = config.getExternalCommand();
        supervisorCommandField = new TextField(
            (existingCommand == null || existingCommand.isBlank()) ? defaultCommand : existingCommand);
        supervisorCommandField.setPromptText("Optional command/script...");
        HBox.setHgrow(supervisorCommandField, Priority.ALWAYS);
        HintUtils.attachHint(supervisorCommandField, "Supervisor command", "Use a local script/wrapper for your LLM provider. Leave empty to run the safe no-op supervisor.");

        intervalRow.getChildren().addAll(intervalLabel, supervisorIntervalSpinner);
        modeRow.getChildren().addAll(modeLabel, supervisorApplyModeComboBox);
        apiKeyRow.getChildren().addAll(apiKeyLabel, supervisorApiKeyField, saveApiKeyBtn);
        commandRow.getChildren().addAll(commandLabel, supervisorCommandField);

        Button triggerLlmButton = styledBtn("Start Auto-Pilot / Trigger Check", "#f39c12");
        HintUtils.attachHint(triggerLlmButton, "Почати авто-пілот", "Негайно відправити стан системи до LLM для аналізу (якщо навчання не запущено).");
        triggerLlmButton.setOnAction(e -> {
            supervisorEnabledCheck.setSelected(true);
            applySupervisorConfigFromUI(currentModelName == null ? "auto_run" : currentModelName);
            rlService.getLlmOrchestrator().forceCheck(this::appendLlmStatus, this::onTrainingProgress);
        });

        supervisorApplyPendingButton = styledBtn("Apply Pending", "#16a085");
        HintUtils.attachHint(supervisorApplyPendingButton, "Apply pending supervisor decision", "Applies the last validated LLM decision when apply mode is MANUAL_APPROVAL.");
        supervisorApplyPendingButton.setOnAction(e -> applyPendingSupervisorDecision());
        
        HBox llmActionRow = new HBox(8);
        llmActionRow.getChildren().addAll(triggerLlmButton, supervisorApplyPendingButton);

        box.getChildren().addAll(supervisorEnabledCheck, intervalRow, modeRow, apiKeyRow, commandRow, llmActionRow);
        return box;
    }

    private VBox createLlmLogSection() {
        VBox box = card();
        box.getChildren().add(sectionTitle("LLM Log"));

        llmLogArea = new TextArea();
        llmLogArea.setEditable(false);
        llmLogArea.setWrapText(true);
        llmLogArea.setPrefHeight(260);
        llmLogArea.setStyle("-fx-font-family: 'Consolas', monospace; -fx-font-size: 11px; -fx-control-inner-background: #151515; -fx-text-fill: #d4d4d4;");

        box.getChildren().add(llmLogArea);
        return box;
    }

    private void handleEncoderModeSelection() {
        if (syncingEncoderMode || encoderModeComboBox == null) return;
        RLTrainingService.EncoderMode selectedMode = encoderModeComboBox.getValue();
        if (selectedMode == null || selectedMode == rlService.getEncoderMode()) return;

        if (trainingRunning) {
            showAlert("Stop training before switching the state encoder.");
            syncEncoderModeCombo();
            return;
        }

        switchEncoderMode(selectedMode, true);
    }

    private void switchEncoderMode(RLTrainingService.EncoderMode mode, boolean userInitiated) {
        if (mode == null) mode = RLTrainingService.EncoderMode.V1;
        RLTrainingService.EncoderMode previousMode = rlService.getEncoderMode();
        if (previousMode == mode) {
            syncEncoderModeCombo();
            return;
        }

        rlService.setEncoderMode(mode);
        syncEncoderModeCombo();
        updateSummaryUI();

        String source = userInitiated ? "Selected" : "Loaded";
        appendStatus(String.format("%s state encoder %s. Runtime stats and replay buffers were reset.", source, mode));
    }

    private void syncEncoderModeCombo() {
        if (encoderModeComboBox == null) return;
        syncingEncoderMode = true;
        try {
            encoderModeComboBox.setValue(rlService.getEncoderMode());
        } finally {
            syncingEncoderMode = false;
        }
    }

    private RLTrainingService.EncoderMode encoderModeFromVersion(String encoderVersion) {
        if ("v3".equalsIgnoreCase(encoderVersion)) {
            return RLTrainingService.EncoderMode.V3;
        }
        if ("v2".equalsIgnoreCase(encoderVersion)) {
            return RLTrainingService.EncoderMode.V2;
        }
        return RLTrainingService.EncoderMode.V1;
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

        bestRewardButton = styledBtn("Apply Best Reward", "#8e44ad");
        HintUtils.attachHint(bestRewardButton, "Найкраща винагорода", "Показати будинок з найбільшою сумарною винагородою епізоду: step reward + final reward.");
        bestRewardButton.setOnAction(e -> applyBestReward());

        diagnosticButton = styledBtn("▶ Run Single AI Step", "#f39c12");
        HintUtils.attachHint(diagnosticButton, "Один крок ШІ", "Завантажена RL-модель читає поточну базу на полі та виконує одну дію.");
        diagnosticButton.setOnAction(e -> runDiagnosticStep());

        box.getChildren().addAll(trainButton, stopButton, generateButton, bestRewardButton, diagnosticButton);
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
        applyLoadProfileFromUI(false);

        int ep = episodesSpinner.getValue();
        int steps = stepsSpinner.getValue();
        int epochs = epochsSpinner.getValue();
        long trainingDurationMs = parseTrainingDurationMs();
        if (trainingDurationMs < 0) {
            showAlert("Training time limit should be empty, 0, minutes, or values like 30m, 2h, 01:30.");
            resetTrainingButtonsAfterRejectedStart();
            return;
        }
        double lw = logisticsSlider.getValue();
        double cw = costSlider.getValue();
        double rw = raidSlider.getValue();
        double ww = workingAreaSlider.getValue();
        double safeZoneW = safeZoneSlider.getValue();
        applySupervisorConfigFromUI(modelName);
        if (trainingDurationMs > 0) {
            appendStatus("Training time limit: " + formatDuration(trainingDurationMs));
        }
        
        final String finalModelName = modelName;
        Thread trainingThread = new Thread(() -> {
            try {
                RLTrainingConfig trainingConfig = new RLTrainingConfig(
                    finalModelName, ep, steps, lw, cw, rw, ww, safeZoneW, epochs,
                    rlService.getSupervisorConfig(), trainingDurationMs,
                    use2dCnnCheck != null && use2dCnnCheck.isSelected());
                rlService.train(trainingConfig,
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

    private void applyLoadProfileFromUI(boolean announce) {
        if (loadProfileComboBox == null) {
            return;
        }

        RLTrainingService.TrainingLoadProfile profile = loadProfileComboBox.getValue();
        if (profile == null) {
            profile = RLTrainingService.TrainingLoadProfile.MAXIMUM;
            loadProfileComboBox.setValue(profile);
        }

        rlService.setTrainingLoadProfile(profile);
        if (loadProfileLabel != null) {
            loadProfileLabel.setText(loadProfileText(profile));
        }
        if (announce) {
            appendStatus("Training load profile: " + profile.getDisplayName());
        }
    }

    private String loadProfileText(RLTrainingService.TrainingLoadProfile profile) {
        RLTrainingService.TrainingLoadProfile safeProfile = profile != null
            ? profile
            : RLTrainingService.TrainingLoadProfile.MAXIMUM;
        if (safeProfile == RLTrainingService.TrainingLoadProfile.MAXIMUM) {
            return "Target: no throttle; best for overnight runs.";
        }
        return String.format("Target: about %d%% active training time; pauses between episodes.",
            safeProfile.getTargetPercent());
    }

    private long parseTrainingDurationMs() {
        if (trainingTimeLimitField == null) {
            return 0L;
        }
        String raw = trainingTimeLimitField.getText();
        if (raw == null || raw.trim().isEmpty()) {
            return 0L;
        }
        String value = raw.trim().toLowerCase(java.util.Locale.ROOT);
        if ("0".equals(value)) {
            return 0L;
        }

        try {
            if (value.contains(":")) {
                String[] parts = value.split(":");
                if (parts.length == 2) {
                    long hours = Long.parseLong(parts[0].trim());
                    long minutes = Long.parseLong(parts[1].trim());
                    return ((hours * 60L) + minutes) * 60_000L;
                }
                if (parts.length == 3) {
                    long hours = Long.parseLong(parts[0].trim());
                    long minutes = Long.parseLong(parts[1].trim());
                    long seconds = Long.parseLong(parts[2].trim());
                    return (((hours * 60L) + minutes) * 60L + seconds) * 1_000L;
                }
                return -1L;
            }

            long multiplier = 60_000L;
            if (value.endsWith("ms")) {
                multiplier = 1L;
                value = value.substring(0, value.length() - 2).trim();
            } else if (value.endsWith("s")) {
                multiplier = 1_000L;
                value = value.substring(0, value.length() - 1).trim();
            } else if (value.endsWith("m")) {
                multiplier = 60_000L;
                value = value.substring(0, value.length() - 1).trim();
            } else if (value.endsWith("h")) {
                multiplier = 3_600_000L;
                value = value.substring(0, value.length() - 1).trim();
            }
            double numeric = Double.parseDouble(value.replace(',', '.'));
            if (numeric < 0) {
                return -1L;
            }
            return Math.round(numeric * multiplier);
        } catch (NumberFormatException ex) {
            return -1L;
        }
    }

    private void resetTrainingButtonsAfterRejectedStart() {
        trainingRunning = false;
        trainButton.setDisable(false);
        stopButton.setDisable(true);
        diagnosticButton.setDisable(false);
        statusLabel.setText("Status: Ready");
    }

    private void applySupervisorConfigFromUI(String modelName) {
        if (supervisorEnabledCheck == null || supervisorIntervalSpinner == null) {
            return;
        }

        LlmSupervisorConfig config = rlService.getSupervisorConfig();
        config.setEnabled(supervisorEnabledCheck.isSelected());
        config.setCallIntervalEpisodes(supervisorIntervalSpinner.getValue());
        config.setBranchId(modelName + "_candidate");
        config.setExternalCommand(supervisorCommandField != null ? supervisorCommandField.getText() : "");
        config.setApiKey(supervisorApiKeyField != null ? supervisorApiKeyField.getText() : "");
        config.setApplyMode(supervisorApplyModeComboBox != null
            ? supervisorApplyModeComboBox.getValue()
            : LlmSupervisorApplyMode.AUTO_APPLY);
        rlService.setSupervisorConfig(config);
        installSupervisorProvider(config);

        if (config.isEnabled()) {
            appendLlmStatus(String.format("LLM Supervisor enabled: every %d episodes.", config.getCallIntervalEpisodes()));
        } else {
            appendLlmStatus(String.format("LLM Supervisor disabled (default interval %d episodes).", config.getCallIntervalEpisodes()));
        }
    }

    private void installSupervisorProvider(LlmSupervisorConfig config) {
        if (config.isEnabled() && !config.getExternalCommand().isBlank()) {
            try {
                java.util.Map<String, String> environmentOverrides = createSupervisorEnvironmentOverrides(config);
                rlService.setLlmSupervisor(new ExternalCommandLlmSupervisor(
                    config.getExternalCommand(),
                    rlService::getRewardConfig,
                    environmentOverrides));
                appendLlmStatus(environmentOverrides.isEmpty()
                    ? "LLM Supervisor command connected."
                    : "LLM Supervisor command connected with UI API key.");
            } catch (IllegalArgumentException ex) {
                rlService.setLlmSupervisor(new NoOpLlmSupervisor());
                appendLlmStatus("LLM Supervisor command rejected: " + ex.getMessage());
            }
        } else {
            rlService.setLlmSupervisor(new NoOpLlmSupervisor());
        }
    }

    private java.util.Map<String, String> createSupervisorEnvironmentOverrides(LlmSupervisorConfig config) {
        String apiKey = config != null ? config.getApiKey() : "";
        if (apiKey.isBlank()) {
            return java.util.Map.of();
        }

        java.util.Map<String, String> environment = new java.util.HashMap<>();
        environment.put("GEMINI_API_KEY", apiKey);
        environment.put("GOOGLE_API_KEY", apiKey);
        environment.put("LLM_API_KEY", apiKey);
        return environment;
    }

    private void applyPendingSupervisorDecision() {
        if (rlService.applyPendingSupervisorDecision()) {
            appendLlmStatus("Applied pending LLM supervisor decision.");
            updateSummaryUI();
        } else {
            appendLlmStatus("No pending LLM supervisor decision to apply.");
        }
    }

    private void onTrainingProgress(TrainingMetrics m) {
        Platform.runLater(() -> {
            String statusText = String.format("Status: Training Epoch %d/%d, Ep %d/%d",
                m.currentEpoch, m.totalEpochs, m.currentEpisodeInEpoch, m.totalEpisodesPerEpoch);
            if (rlService != null && rlService.getTrainingLoadProfile() != null) {
                statusText += " | Load " + rlService.getTrainingLoadProfile().getTargetPercent() + "%";
            }
            if (m.trainingTimeLimitEnabled) {
                statusText += " | Left " + formatDuration(m.trainingRemainingMs);
            }
            statusLabel.setText(statusText);
            
            int totalEps = m.totalEpochs * m.totalEpisodesPerEpoch;
            int doneEps = (m.currentEpoch - 1) * m.totalEpisodesPerEpoch + m.currentEpisodeInEpoch;
            progressBar.setProgress((double) doneEps / totalEps);

            StringBuilder sb = new StringBuilder();
            sb.append(String.format("[%02d:%02d:%02d] EPOCH %d/%d EP %d/%d\n", 
                java.time.LocalTime.now().getHour(), java.time.LocalTime.now().getMinute(), java.time.LocalTime.now().getSecond(),
                m.currentEpoch, m.totalEpochs, m.currentEpisodeInEpoch, m.totalEpisodesPerEpoch));
            double totalReward = m.currentEpisodeStepReward + m.currentEpisodeFinalReward;
            sb.append(String.format("  Best Score: %.4f | Avg Eval Score: %.4f\n", m.bestScore, m.avgEvalScore));
            sb.append(String.format("  Eval Score:  %.4f | Total Reward: %.2f\n", m.currentEpisodeEvalScore, totalReward));
            sb.append(String.format("  Epsilon:    %.4f | Loss:     %.6f\n", m.epsilon, m.lastTrainLoss));
            if (m.trainingTimeLimitEnabled) {
                sb.append(String.format("  Time:       elapsed %s | left %s | deadline %s\n",
                    formatDuration(m.trainingElapsedMs),
                    formatDuration(m.trainingRemainingMs),
                    formatDeadline(m.trainingDeadlineEpochMs)));
            } else {
                sb.append(String.format("  Time:       elapsed %s | no limit\n", formatDuration(m.trainingElapsedMs)));
            }
            sb.append(String.format("  Step Reward: %.2f | Final Reward: %.2f\n", m.currentEpisodeStepReward, m.currentEpisodeFinalReward));
            if (m.currentEpisodeStepRewardBreakdown != null && !m.currentEpisodeStepRewardBreakdown.isEmpty()) {
                sb.append("  Step By:    ").append(m.currentEpisodeStepRewardBreakdown).append("\n");
            }
            if (m.currentEpisodeFinalRewardBreakdown != null && !m.currentEpisodeFinalRewardBreakdown.isEmpty()) {
                sb.append("  Final By:   ").append(m.currentEpisodeFinalRewardBreakdown).append("\n");
            }
            sb.append(String.format("  Invalid:    %d/%d (%.1f%%)\n", m.lastEpisodeInvalidActions, m.lastEpisodeTotalActions, m.invalidActionRate * 100));
            sb.append(String.format("  Best Base:  Blocks %d, TC %s, Doors %d\n", m.bestBaseBlocks, m.bestBaseHasTC ? "YES" : "NO", m.bestBaseDoors));
            sb.append(String.format("  Best Reward: total %.2f | best-base step %.2f final %.2f total %.2f\n",
                m.bestTotalReward, m.bestBaseStepReward, m.bestBaseFinalReward, m.bestBaseTotalReward));
            sb.append("------------------------------------------\n");
            
            diagnosticLogArea.setText(sb.toString() + diagnosticLogArea.getText());

            updateSummaryUI(m);
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

    private void updateSummaryUI(TrainingMetrics m) {
        if (m == null) {
            updateSummaryUI();
            return;
        }

        bestScoreLbl.setText(String.format("%.4f", m.bestScore));
        avgEvalScoreLbl.setText(String.format("%.4f", m.avgEvalScore));
        avgStepRewardLbl.setText(String.format("%.4f", m.avgReward));
        invalidRateLbl.setText(String.format("%.1f%%", m.invalidActionRate * 100));
        memorySizeLbl.setText(String.valueOf(m.memorySize));
        episodesLbl.setText(String.valueOf(m.totalEpisodesTrained));
        epsilonLbl.setText(String.format("%.4f", m.epsilon));
        lossLbl.setText(String.format("%.6f", m.lastTrainLoss));

        String bestBase = String.format("B: %d, TC: %s, D: %d",
            m.bestBaseBlocks,
            m.bestBaseHasTC ? "Yes" : "No",
            m.bestBaseDoors);
        bestBaseLbl.setText(bestBase);

        if (rlService != null) {
            ramUsageLbl.setText(rlService.getRamUsage());
            timerLbl.setText(m.trainingTimeLimitEnabled
                ? formatDuration(m.trainingElapsedMs) + " / left " + formatDuration(m.trainingRemainingMs)
                : rlService.getFormattedTrainingTime());
        }
    }

    private String formatDuration(long ms) {
        if (ms < 0) {
            return "--:--:--";
        }
        long totalSeconds = Math.max(0L, ms / 1000L);
        long seconds = totalSeconds % 60L;
        long minutes = (totalSeconds / 60L) % 60L;
        long hours = totalSeconds / 3600L;
        return String.format("%02d:%02d:%02d", hours, minutes, seconds);
    }

    private String formatDeadline(long epochMs) {
        if (epochMs <= 0) {
            return "none";
        }
        return java.time.format.DateTimeFormatter.ofPattern("HH:mm:ss")
            .withZone(java.time.ZoneId.systemDefault())
            .format(java.time.Instant.ofEpochMilli(epochMs));
    }

    private void stopTraining() {
        if (!trainingRunning) return;
        rlService.requestStop();
        statusLabel.setText("Status: Stop Requested...");
    }

    private void applyBest() {
        GridModel best = rlService.getBestGridModelSnapshot();
        if (best == null || best.getAllBlocks().isEmpty()) {
            showAlert("No successful base trained yet!");
            return;
        }

        applyGridToMain(best);
        appendStatus("Applied best trained layout (Score: " + String.format("%.2f", rlService.getBestScore()) + ", Blocks: " + mainGrid.getAllBlocks().size() + ")");
    }

    private void applyBestReward() {
        GridModel best = rlService.getBestRewardGridModelSnapshot();
        if (best == null || best.getAllBlocks().isEmpty()) {
            showAlert("No best-reward episode recorded yet!");
            return;
        }

        applyGridToMain(best);
        appendStatus("Applied best reward layout (Total Reward: " + String.format("%.2f", rlService.getBestTotalReward()) + ", Blocks: " + mainGrid.getAllBlocks().size() + ")");
    }

    private void applyGridToMain(GridModel source) {
        mainGrid.clear();
        for (BuildingBlock b : source.getAllBlocks()) {
            BuildingBlock clone = cloneBlock(b);
            if (clone != null) {
                mainGrid.addBlockSilent(clone);
            }
        }
        mainGrid.finalizeLoad();
        if (refreshCallback != null) {
            refreshCallback.run();
        }
        if (gameCanvas != null) {
            gameCanvas.invalidateCache();
            gameCanvas.draw();
        }
    }

    private void runDiagnosticStep() {
        if (trainingRunning) return;
        
        com.rustbuilder.ai.rl.multidiscrete.MultiDiscretePhasePolicy policy = rlService.getMultiDiscretePolicy();
        if (policy == null) {
            showAlert("No active RL policy found.");
            return;
        }

        GridModel stepGrid = mainGrid.clone();
        diagnosticButton.setDisable(true);
        appendStatus("--- Running AI Single Step ---");
        
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
                com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteAction action = rlService.chooseSingleStepAction(stepGrid, guiObserver);
                if (action == null || !action.isValid() || action.getTypeIndex() == com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteActionSpace.STOP_TYPE_INDEX) {
                    appendStatus("  AI selected STOP. No block placed.");
                    Platform.runLater(() -> diagnosticButton.setDisable(false));
                    return;
                }
                
                com.rustbuilder.core.action.BuildAction bAction =
                    com.rustbuilder.ai.rl.multidiscrete.MultiDiscreteActionMapper.toBuildAction(action);
                
                Platform.runLater(() -> {
                    com.rustbuilder.ai.rl.RLTrainingService.PlacementResult res = rlService.placeBlock(mainGrid, bAction);
                    if (res.inserted) {
                        appendStatus("  Placement successful: " + bAction.actionType);
                    } else {
                        appendStatus("  Placement FAILED: " + res.failReason);
                    }
                    if (refreshCallback != null) refreshCallback.run();
                    if (gameCanvas != null) {
                        gameCanvas.invalidateCache();
                        gameCanvas.draw();
                    }
                    diagnosticButton.setDisable(false);
                });
            } catch (Exception ex) {
                appendStatus("  Single Step Error: " + ex.toString());
                Platform.runLater(() -> diagnosticButton.setDisable(false));
            }
        });
        t.setDaemon(true);
        t.start();
    }

    private BuildingBlock cloneBlock(BuildingBlock b) {
        return b == null ? null : b.clone();
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
            double sw = safeZoneSlider.getValue();
            
            RLModel snapshot = RLModelManager.createSnapshot(name, rlService, lw, cw, rw, ww, sw);
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
            RLModel meta = RLModelManager.loadMetadata(name);
            switchEncoderMode(encoderModeFromVersion(meta.stateEncoderVersion), false);
            RLModelManager.restoreFromModel(rlService, meta); // Validates action-space shape before loading weights.
            RLModelManager.loadNetworkWeights(name, rlService);
            syncEncoderModeCombo();
            
            currentModelName = name;
            newModelField.setText(name);
            
            logisticsSlider.setValue(meta.logisticsWeight);
            costSlider.setValue(meta.costWeight);
            raidSlider.setValue(meta.raidWeight);
            workingAreaSlider.setValue(meta.workingAreaWeight);
            safeZoneSlider.setValue(meta.safeZoneWeight);
            
            if (meta.rewardConfig != null) {
                rlService.setRewardConfig(meta.rewardConfig.clone());
            }
            syncRewardUIFromConfig();
            syncSupervisorUIFromConfig();
            
            // Legacy multiDiscrete check removed
            updateStats();
            updateSummaryUI(); // Important: refresh UI after load
            
            appendStatus("Model loaded successfully: " + meta.toString());
        } catch (IOException | ClassNotFoundException | IllegalArgumentException ex) {
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
        
        appendStatus(String.format("Active Policy: %s", policyName));
        appendStatus(String.format("Best Score: %.4f | Episodes: %d", rlService.getBestScore(), rlService.getEpisodesTrained()));
    }

    public void show() {
        refreshModelList();
        syncSupervisorUIFromConfig();
        updateStats();
        updateSummaryUI();
        reportMode();
        dialogStage.show();
    }

    private void syncSupervisorUIFromConfig() {
        if (supervisorEnabledCheck == null || supervisorIntervalSpinner == null || rlService == null) {
            return;
        }
        LlmSupervisorConfig config = rlService.getSupervisorConfig();
        supervisorEnabledCheck.setSelected(config.isEnabled());
        supervisorIntervalSpinner.getValueFactory().setValue(config.getCallIntervalEpisodes());
        if (supervisorCommandField != null) {
            supervisorCommandField.setText(config.getExternalCommand());
        }
        if (supervisorApiKeyField != null) {
            supervisorApiKeyField.setText(config.getApiKey());
        }
        if (supervisorApplyModeComboBox != null) {
            supervisorApplyModeComboBox.setValue(config.getApplyMode());
        }
    }

    public void appendStatus(String text) {
        Platform.runLater(() -> {
            diagnosticLogArea.appendText(text + "\n");
        });
    }

    public void appendLlmStatus(String text) {
        Platform.runLater(() -> {
            if (llmLogArea == null) {
                return;
            }
            llmLogArea.appendText(String.format("[%02d:%02d:%02d] %s%n",
                java.time.LocalTime.now().getHour(),
                java.time.LocalTime.now().getMinute(),
                java.time.LocalTime.now().getSecond(),
                text));
        });
    }

    public void reportMode() {
        boolean learnsMD = true;
        
        String report = String.format("[RL UI REPORT] mode=MULTI_DISCRETE, training=%s", 
            learnsMD ? "ENABLED" : "DISABLED");
        
        appendStatus(report);
    }

    public void cleanup() {
        if (rlService != null) {
            rlService.setSupervisorLogCallback(null);
        }
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
        addRewardRow(grid, row++, "Закрита шафа", "tcEnclosedBonus", "Бонус, якщо TC неможливо досягти ззовні без рейд-витрат.");
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
        addRewardRow(grid, row++, "Покрокова дельта оцінки", "stepEvalDeltaMultiplier", "Множник покрокової зміни HouseEvaluator: логістика, ресурси, рейдостійкість, робочий простір, safe-zone.");
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
        updateSpinner("tcEnclosedBonus", config.tcEnclosedBonus);
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

        updateSpinner("stepEvalDeltaMultiplier", config.stepEvalDeltaMultiplier);
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
        config.tcEnclosedBonus = rewardSpinners.get("tcEnclosedBonus").getValue();
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

        config.stepEvalDeltaMultiplier = rewardSpinners.get("stepEvalDeltaMultiplier").getValue();
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
