package com.rustbuilder.controller;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.core.ResourceType;
import com.rustbuilder.service.evaluator.HouseEvaluator;
import com.rustbuilder.service.evaluator.HouseEvaluationService;
import com.rustbuilder.service.evaluator.HouseEvaluatorFactory;
import com.rustbuilder.service.physics.SnapResolver;
import com.rustbuilder.service.physics.SnappingService;
import com.rustbuilder.service.physics.SnappingService.SnapResult;
import com.rustbuilder.ui.GameCanvas;
import com.rustbuilder.util.BlockFactory;
import com.rustbuilder.util.BuildingTypeUtils;
import com.rustbuilder.config.GameConstants;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import javafx.geometry.Point2D;

public class GameController {

    private final GridModel gridModel;
    private final SnapResolver snappingService;
    private final GameCanvas gameCanvas;
    private static final String DELETE_TOOL = "DELETE";

    private String selectedTool = BuildingTypeUtils.toToolId(BuildingType.FOUNDATION);
    private BuildingType selectedBuildingType = BuildingType.FOUNDATION;
    private BuildingTier selectedTier = BuildingTier.STONE;
    private DoorType selectedDoorType = DoorType.SHEET_METAL;
    private final HouseEvaluationService houseEvaluator;
    private int currentFloor = 0;

    // Ghost State
    private double ghostX;
    private double ghostY;
    private double ghostRotation;
    private Orientation ghostOrientation = Orientation.NORTH;
    private boolean ghostValid = true;

    public void setSelectedTier(BuildingTier tier) {
        this.selectedTier = tier;
    }

    public void setSelectedDoorType(DoorType doorType) {
        this.selectedDoorType = doorType;
    }

    public DoorType getSelectedDoorType() {
        return selectedDoorType;
    }

    public HouseEvaluator.EvaluationResult evaluateHouse() {
        return houseEvaluator.evaluate(gridModel);
    }

    public GameController(GridModel gridModel, GameCanvas gameCanvas) {
        this(gridModel, gameCanvas, new SnappingService(gridModel), HouseEvaluatorFactory.createDefault());
    }

    public GameController(GridModel gridModel,
                          GameCanvas gameCanvas,
                          SnapResolver snappingService,
                          HouseEvaluationService houseEvaluator) {
        this.gridModel = gridModel;
        this.gameCanvas = gameCanvas;
        this.snappingService = Objects.requireNonNull(snappingService, "snappingService");
        this.houseEvaluator = Objects.requireNonNull(houseEvaluator, "houseEvaluator");
    }

    private double lastDragX;
    private double lastDragY;

    public void handleScroll(double deltaY, double mouseX, double mouseY) {
        double zoomFactor = 1.05;
        if (deltaY < 0) {
            zoomFactor = 1 / zoomFactor;
        }
        gameCanvas.zoom(zoomFactor, mouseX, mouseY);
        // Update ghost after zoom
        handleMouseMove(mouseX, mouseY);
    }

    public void handleMousePressed(double mouseX, double mouseY, boolean isMiddleButton) {
        if (isMiddleButton) {
            lastDragX = mouseX;
            lastDragY = mouseY;
        }
    }

    public void handleMouseDragged(double mouseX, double mouseY, boolean isMiddleButton) {
        if (isMiddleButton) {
            double dx = mouseX - lastDragX;
            double dy = mouseY - lastDragY;
            gameCanvas.pan(dx, dy);
            lastDragX = mouseX;
            lastDragY = mouseY;
        } else {
            // Dragging with other buttons? Maybe standard mouse move logic just to update ghost
            handleMouseMove(mouseX, mouseY);
        }
    }

    public void handleMouseMove(double mouseX, double mouseY) {
        if (isDeleteTool()) {
            gameCanvas.setGhost(0, 0, 0, null, false, null, null);
            gameCanvas.draw();
            return;
        }

        // Convert Screen to World
        Point2D world = gameCanvas.toGrid(mouseX, mouseY);
        double worldX = world.getX();
        double worldY = world.getY();

        SnapResult result = snappingService.calculateSnap(worldX, worldY, selectedTool, currentFloor);
        this.ghostX = result.x;
        this.ghostY = result.y;
        this.ghostRotation = result.rotation;
        this.ghostOrientation = result.orientation;

        // Check Collision
        BuildingBlock tempBlock = createBlock(ghostX, ghostY, currentFloor, ghostRotation, ghostOrientation, selectedBuildingType);
        
        if (tempBlock != null && shouldApplyWallRotation(selectedBuildingType)) {
            tempBlock.setRotation(ghostRotation);
        }
        this.ghostValid = result.valid && tempBlock != null && gridModel.canPlace(tempBlock);

        gameCanvas.setGhost(ghostX, ghostY, ghostRotation, selectedTool, ghostValid, ghostOrientation, selectedTier);
        gameCanvas.draw();
    }

    private BuildingBlock selectedBlock;
    private final Map<ResourceType, Integer> totalConstructionCost = new HashMap<>();

    public BuildingBlock getSelectedBlock() {
        return selectedBlock;
    }
    
    public Map<ResourceType, Integer> getTotalConstructionCost() {
        return totalConstructionCost;
    }

    public void handleMouseClick(double mouseX, double mouseY, boolean isPrimaryButton) {
        if (isPrimaryButton) {
            if (isDeleteTool()) {
                deleteBlockAt(mouseX, mouseY);
            } else {
                if (!placeBlock()) {
                    // Only select block if we didn't place anything
                    Point2D world = gameCanvas.toGrid(mouseX, mouseY);
                    BuildingBlock clicked = findBlockAt(world.getX(), world.getY());
                    selectedBlock = clicked;
                    gameCanvas.draw();
                } else {
                    selectedBlock = null;
                }
            }
        } else {
            // Right click logic (e.g., rotate)
        }
    }

    private BuildingBlock findBlockAt(double worldX, double worldY) {
         for (BuildingBlock b : gridModel.getAllBlocks()) {
            if (b.getZ() != currentFloor) continue;
            
            double[] poly = b.getCollisionPoints();
            if (poly != null && poly.length >= 6) {
                if (isPointInPolygon(worldX, worldY, poly)) {
                    return b;
                }
            } else {
                // Simple bbox check for blocks with no collision polygon
                if (worldX >= b.getX() && worldX <= b.getX() + GameConstants.TILE_SIZE &&
                    worldY >= b.getY() && worldY <= b.getY() + GameConstants.TILE_SIZE) {
                    return b;
                }
            }
        }
        return null;
    }

    private boolean isPointInPolygon(double x, double y, double[] poly) {
        return com.rustbuilder.util.GeometryUtils.isPointInPolygon(x, y, poly);
    }

    private void deleteBlockAt(double mouseX, double mouseY) {
        Point2D world = gameCanvas.toGrid(mouseX, mouseY);
        double worldX = world.getX();
        double worldY = world.getY();
        BuildingBlock toDelete = findBlockAt(worldX, worldY);
        if (toDelete != null) {
            Map<ResourceType, Integer> cost = toDelete.getBuildCost();
            cost.forEach((k, v) -> totalConstructionCost.merge(k, -v, (a, b) -> a + b)); // Subtract
            
            gridModel.removeBlock(toDelete);
            gameCanvas.invalidateCache();
            gameCanvas.draw();
        }
    }

    private boolean placeBlock() {
        if (!ghostValid) {
            return false;
        }
        BuildingBlock newBlock = createBlock(ghostX, ghostY, currentFloor, ghostRotation, ghostOrientation, selectedBuildingType);

        if (newBlock != null) {
            if (shouldApplyWallRotation(selectedBuildingType)) {
                newBlock.setRotation(ghostRotation);
            }

            if (gridModel.canPlace(newBlock)) {
                boolean added = gridModel.addBlock(newBlock);
                if (!added) {
                    gameCanvas.invalidateCache();
                    gameCanvas.draw();
                    return false;
                }
                
                Map<ResourceType, Integer> cost = newBlock.getBuildCost();
                cost.forEach((k, v) -> totalConstructionCost.merge(k, v, (a, b) -> a + b));
                
                gameCanvas.invalidateCache();
                gameCanvas.draw();
                return true;
            }
        }
        return false;
    }

    private BuildingBlock createBlock(double x, double y, int z, double rotation, Orientation orientation, BuildingType type) {
        if (type == null) {
            return null;
        }

        // Prevent foundations on upper floors
        if (z > 0 && BuildingTypeUtils.isFoundation(type)) {
            return null;
        }

        BuildingBlock block = BlockFactory.create(type, x, y, z, rotation, orientation, selectedDoorType);
        
        if (block != null) {
            if (!BuildingTypeUtils.isFurniture(block.getType()) && block.getType() != BuildingType.DOOR) {
                block.setTier(selectedTier);
            }
        }
        
        return block;
    }

    private boolean shouldApplyWallRotation(BuildingType type) {
        return BuildingTypeUtils.isWall(type) || type == BuildingType.DOOR;
    }

    public void setSelectedTool(String tool) {
        this.selectedTool = tool;
        this.selectedBuildingType = BuildingTypeUtils.fromToolId(tool);
    }

    public void setSelectedTool(BuildingType type) {
        this.selectedBuildingType = type;
        this.selectedTool = BuildingTypeUtils.toToolId(type);
    }

    public void selectDeleteTool() {
        this.selectedTool = DELETE_TOOL;
        this.selectedBuildingType = null;
    }

    private boolean isDeleteTool() {
        return DELETE_TOOL.equals(selectedTool);
    }

    public void moveFloorUp() {
        currentFloor++;
        gameCanvas.setCurrentFloor(currentFloor);
    }

    public void moveFloorDown() {
        if (currentFloor > 0) {
            currentFloor--;
            gameCanvas.setCurrentFloor(currentFloor);
        }
    }

    public void clearGrid() {
        gridModel.clear();
        totalConstructionCost.clear();
        gameCanvas.invalidateCache();
        gameCanvas.draw();
    }
}
