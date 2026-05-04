package com.rustbuilder;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import com.rustbuilder.ai.rl.RLTrainingService;
import com.rustbuilder.controller.GameController;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.ui.GameCanvas;
import com.rustbuilder.ui.hints.HintUtils;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Separator;
import javafx.scene.control.TextArea;
import javafx.scene.control.ToolBar;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;

public class MainApp extends Application {

    private GameController gameController;
    private RLTrainingService rlTrainingService;

    @Override
    public void start(Stage stage) {
        try {
            GridModel gridModel = new GridModel();
            GameCanvas gameCanvas = new GameCanvas(gridModel, 800, 600);
            gameController = new GameController(gridModel, gameCanvas);
            rlTrainingService = new RLTrainingService();
            gameCanvas.setController(gameController);

            // === Building Tools ===
            ToolBar toolBar = new ToolBar();

            Button btnFoundation = new Button("Foundation");
            btnFoundation.setOnAction(e -> gameController.setSelectedTool("FOUNDATION"));
            HintUtils.attachHint(btnFoundation, "Фундамент", "Квадратний базовий блок для початку будівництва.");

            Button btnTriFoundation = new Button("Tri Foundation");
            btnTriFoundation.setOnAction(e -> gameController.setSelectedTool("TRIANGLE"));
            HintUtils.attachHint(btnTriFoundation, "Трикутний фундамент", "Трикутний базовий блок для початку будівництва.");

            Button btnWall = new Button("Wall");
            btnWall.setOnAction(e -> gameController.setSelectedTool("WALL"));
            HintUtils.attachHint(btnWall, "Стіна", "Захисний блок. Розміщується на краю фундаменту або підлоги.");

            Button btnFloor = new Button("Floor");
            btnFloor.setOnAction(e -> gameController.setSelectedTool("FLOOR"));
            HintUtils.attachHint(btnFloor, "Підлога / Стеля", "Розміщується на стінах як перекриття.");

            Button btnTriFloor = new Button("Tri Floor");
            btnTriFloor.setOnAction(e -> gameController.setSelectedTool("TRIANGLE_FLOOR"));
            HintUtils.attachHint(btnTriFloor, "Трикутна підлога", "Розміщується на стінах як трикутне перекриття.");

            Button btnDoorFrame = new Button("Door Frame");
            btnDoorFrame.setOnAction(e -> gameController.setSelectedTool("DOOR_FRAME"));
            HintUtils.attachHint(btnDoorFrame, "Дверний проріз", "Стіна з отвором для встановлення дверей.");

            Button btnWindowFrame = new Button("Window");
            btnWindowFrame.setOnAction(e -> gameController.setSelectedTool("WINDOW_FRAME"));
            HintUtils.attachHint(btnWindowFrame, "Вікно", "Стіна з отвором для вікна.");

            Button btnDoor = new Button("Door");
            btnDoor.setOnAction(e -> gameController.setSelectedTool("DOOR"));
            HintUtils.attachHint(btnDoor, "Двері", "Захист входу. Розміщується у дверному прорізі.");

            Button btnToolCupboard = new Button("TC");
            btnToolCupboard.setOnAction(e -> gameController.setSelectedTool("TC"));
            HintUtils.attachHint(btnToolCupboard, "Шафа (TC)", "Запобігає гниттю бази і блокує будівництво іншим гравцям.");

            Button btnWorkbench = new Button("Workbench");
            btnWorkbench.setOnAction(e -> gameController.setSelectedTool("WORKBENCH"));
            HintUtils.attachHint(btnWorkbench, "Верстак", "Необхідний для створення предметів та проведення досліджень.");

            Button btnLootRoom = new Button("Loot Room");
            btnLootRoom.setOnAction(e -> gameController.setSelectedTool("LOOT_ROOM"));
            HintUtils.attachHint(btnLootRoom, "Скрині (Лутова)", "Зона для зберігання ресурсів та цінностей.");

            // === Navigation ===
            Button btnUp = new Button("▲");
            btnUp.setOnAction(e -> gameController.moveFloorUp());
            HintUtils.attachHint(btnUp, "Поверх вгору", "Перейти на вищий рівень.");

            Button btnDown = new Button("▼");
            btnDown.setOnAction(e -> gameController.moveFloorDown());
            HintUtils.attachHint(btnDown, "Поверх вниз", "Перейти на нижчий рівень.");

            Button btnDelete = new Button("Delete");
            btnDelete.setOnAction(e -> gameController.setSelectedTool("DELETE"));
            HintUtils.attachHint(btnDelete, "Видалити", "Видалення вибраного блоку з сітки.");

            Button btnClear = new Button("Clear");
            btnClear.setOnAction(e -> gameController.clearGrid());
            HintUtils.attachHint(btnClear, "Очистити все", "Повне очищення сітки від усіх блоків.");

            // === Tier Selection ===
            ComboBox<String> tierBox = new ComboBox<>();
            tierBox.getItems().addAll("Twig", "Wood", "Stone", "Metal", "HQM");
            tierBox.setValue("Stone");
            HintUtils.attachHint(tierBox, "Матеріал бази", "Вибір матеріалу для будівництва блоків.");
            tierBox.setOnAction(e -> {
                String sel = tierBox.getValue();
                switch (sel) {
                    case "Twig":  gameController.setSelectedTier(BuildingTier.TWIG); break;
                    case "Wood":  gameController.setSelectedTier(BuildingTier.WOOD); break;
                    case "Stone": gameController.setSelectedTier(BuildingTier.STONE); break;
                    case "Metal": gameController.setSelectedTier(BuildingTier.METAL); break;
                    case "HQM":   gameController.setSelectedTier(BuildingTier.HQM); break;
                }
            });

            // === Door Type Selection ===
            ComboBox<String> doorBox = new ComboBox<>();
            doorBox.getItems().addAll("Sheet Metal", "Garage", "Armored");
            doorBox.setValue("Sheet Metal");
            HintUtils.attachHint(doorBox, "Тип дверей", "Вибір типу дверей для встановлення.");
            doorBox.setOnAction(e -> {
                String sel = doorBox.getValue();
                switch (sel) {
                    case "Sheet Metal": gameController.setSelectedDoorType(DoorType.SHEET_METAL); break;
                    case "Garage":      gameController.setSelectedDoorType(DoorType.GARAGE); break;
                    case "Armored":     gameController.setSelectedDoorType(DoorType.ARMORED); break;
                }
            });

            // === Evaluate Button ===
            Button btnEvaluate = new Button("⚡ Evaluate");
            btnEvaluate.setStyle("-fx-background-color: #4CAF50; -fx-text-fill: white; -fx-font-weight: bold;");
            HintUtils.attachHint(btnEvaluate, "Оцінити (Evaluate)", "Розрахувати вартість, логістику та стійкість поточної бази.");
            btnEvaluate.setOnAction(e -> {
                btnEvaluate.setDisable(true);
                btnEvaluate.setText("⏳ Evaluating...");

                CompletableFuture
                    .supplyAsync(gameController::evaluateHouse)
                    .orTimeout(8, TimeUnit.SECONDS)
                    .whenComplete((result, error) -> Platform.runLater(() -> {
                        btnEvaluate.setDisable(false);
                        btnEvaluate.setText("⚡ Evaluate");

                        if (error != null) {
                            Throwable ex = error.getCause() != null ? error.getCause() : error;
                            StringWriter sw = new StringWriter();
                            ex.printStackTrace(new PrintWriter(sw));
                            String stack = sw.toString();

                            Alert err = new Alert(Alert.AlertType.ERROR);
                            err.setTitle("Evaluation Error");
                            err.setHeaderText("Failed to evaluate current base");

                            String msg = ex.getClass().getSimpleName() + ": " +
                                (ex.getMessage() == null ? "(no message)" : ex.getMessage()) +
                                "\n\n" + stack;
                            TextArea details = new TextArea(msg);
                            details.setEditable(false);
                            details.setWrapText(true);
                            details.setPrefColumnCount(80);
                            details.setPrefRowCount(8);

                            err.getDialogPane().setContent(details);
                            err.getDialogPane().setMinWidth(700);
                            err.showAndWait();
                            return;
                        }

                        if (result == null) {
                            Alert err = new Alert(Alert.AlertType.ERROR);
                            err.setTitle("Evaluation Error");
                            err.setHeaderText("Failed to evaluate current base");
                            err.setContentText("House evaluation returned null result.");
                            err.showAndWait();
                            return;
                        }

                        Alert alert = new Alert(Alert.AlertType.INFORMATION);
                        alert.setTitle("House Evaluation");
                        alert.setHeaderText(String.format("Final Score: %.2f / 1.00", result.finalScore));
                        alert.setContentText(result.toString());
                        alert.getDialogPane().setMinWidth(500);
                        alert.showAndWait();
                    }));
            });

            // === AI Generate Button ===
            Button btnAI = new Button("🧬 GA Generate");
            btnAI.setStyle("-fx-background-color: #2980b9; -fx-text-fill: white; -fx-font-weight: bold;");
            HintUtils.attachHint(btnAI, "Генетичний алгоритм", "Відкриває вікно для генерації бази за допомогою еволюційного підходу.");
            btnAI.setOnAction(e -> {
                com.rustbuilder.ui.GeneratorDialog dialog = 
                    new com.rustbuilder.ui.GeneratorDialog(gridModel, gameCanvas, stage);
                dialog.show();
                gameCanvas.draw(); // refresh canvas after dialog closes
            });

            // === RL AI Button ===
            Button btnRL = new Button("🤖 RL Generate");
            btnRL.setStyle("-fx-background-color: #8e44ad; -fx-text-fill: white; -fx-font-weight: bold;");
            HintUtils.attachHint(btnRL, "Навчання з підкріпленням (RL)", "Відкриває вікно для навчання нейромережі будувати бази.");
            btnRL.setOnAction(e -> {
                com.rustbuilder.ui.RLGeneratorDialog.showDialog(
                    null,
                    rlTrainingService,
                    gridModel,
                    () -> Platform.runLater(() -> {
                        gameCanvas.invalidateCache();
                        gameCanvas.draw();
                    }),
                    gameCanvas
                );
            });

            // === Stability Tool ===
            Button btnStability = new Button("🛡 Stability");
            HintUtils.attachHint(btnStability, "Стабільність", "Інструмент для перевірки структурної цілісності бази. Показує відсоток стабільності для блоків.");
            btnStability.setOnAction(e -> {
                boolean newState = btnStability.getUserData() == null ? true : !(boolean)btnStability.getUserData();
                btnStability.setUserData(newState);

                gameCanvas.setShowStabilityTool(newState);
                if (newState) {
                    btnStability.setStyle("-fx-background-color: #2ecc71; -fx-text-fill: white; -fx-font-weight: bold;");
                    gameCanvas.setGhost(0, 0, 0, null, false);
                    gameCanvas.draw();
                } else {
                    btnStability.setStyle("");
                    gameCanvas.setHoveredBlock(null);
                }
            });

            toolBar.getItems().addAll(
                    btnFoundation, btnTriFoundation, btnWall, btnFloor, btnTriFloor,
                    btnDoorFrame, btnDoor, btnWindowFrame,
                    new Separator(),
                    btnToolCupboard, btnWorkbench, btnLootRoom,
                    new Separator(),
                    btnUp, btnDown, btnDelete, btnClear,
                    new Separator(),
                    tierBox, doorBox,
                    new Separator(),
                    btnEvaluate, btnAI, btnRL,
                    new Separator(),
                    btnStability
            );

            // Mouse Events
            gameCanvas.setOnMouseMoved(e -> {
                boolean stabilityActive = btnStability.getUserData() != null && (boolean)btnStability.getUserData();
                if (stabilityActive) {
                    BuildingBlock b = gameCanvas.hitTest(e.getX(), e.getY());
                    gameCanvas.setHoveredBlock(b);
                } else {
                    gameController.handleMouseMove(e.getX(), e.getY());
                }
            });
            gameCanvas.setOnMouseClicked(e -> {
                boolean stabilityActive = btnStability.getUserData() != null && (boolean)btnStability.getUserData();
                if (!stabilityActive) {
                    gameController.handleMouseClick(e.getX(), e.getY(),
                        e.getButton() == javafx.scene.input.MouseButton.PRIMARY);
                }
            });
            gameCanvas.setOnScroll(e -> gameController.handleScroll(e.getDeltaY(), e.getX(), e.getY()));
            gameCanvas.setOnMousePressed(e -> gameController.handleMousePressed(e.getX(), e.getY(), 
                    e.getButton() == javafx.scene.input.MouseButton.MIDDLE));
            gameCanvas.setOnMouseDragged(e -> gameController.handleMouseDragged(e.getX(), e.getY(), 
                    e.getButton() == javafx.scene.input.MouseButton.MIDDLE));

            BorderPane root = new BorderPane();
            root.setTop(toolBar);
            root.setCenter(gameCanvas);

            // Bind canvas to container so it fills the window and redraws on resize
            gameCanvas.widthProperty().bind(root.widthProperty());
            gameCanvas.heightProperty().bind(root.heightProperty());
            gameCanvas.widthProperty().addListener(o -> gameCanvas.draw());
            gameCanvas.heightProperty().addListener(o -> gameCanvas.draw());

            Scene scene = new Scene(root, 1000, 700);
            stage.setTitle("Rust Base Builder");
            stage.setScene(scene);
            stage.show();
            gameCanvas.draw(); // draw after show() so getWidth()/getHeight() are correct
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle("Startup Error");
            alert.setHeaderText("Failed to initialize application");
            alert.setContentText(e.getMessage() != null ? e.getMessage() : e.toString());
            alert.showAndWait();
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
