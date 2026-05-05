package com.rustbuilder.controller;

import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.structure.Door;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.model.structure.Floor;
import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.deployable.LootRoom;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.core.ResourceType;
import com.rustbuilder.model.deployable.ToolCupboard;
import com.rustbuilder.model.structure.TriangleFloor;
import com.rustbuilder.model.structure.TriangleFoundation;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.model.deployable.Workbench;
import com.rustbuilder.service.evaluator.HouseEvaluator;
import com.rustbuilder.service.physics.SnappingService;
import com.rustbuilder.service.physics.SnappingService.SnapResult;
import com.rustbuilder.ui.GameCanvas;
import com.rustbuilder.util.BuildingTypeUtils;
import com.rustbuilder.config.GameConstants;
import java.util.HashMap;
import java.util.Map;

import javafx.geometry.Point2D;

public class GameController {

    private final GridModel gridModel;
    private final SnappingService snappingService;
    private final GameCanvas gameCanvas;

    private String selectedTool = "FOUNDATION";
    private BuildingTier selectedTier = BuildingTier.STONE;
    private DoorType selectedDoorType = DoorType.SHEET_METAL;
    private final HouseEvaluator houseEvaluator = new HouseEvaluator();
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
        this.gridModel = gridModel;
        this.gameCanvas = gameCanvas;
        this.snappingService = new SnappingService(gridModel);
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
        if ("DELETE".equals(selectedTool)) {
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
        BuildingBlock tempBlock = createBlock(ghostX, ghostY, currentFloor, ghostRotation, ghostOrientation, selectedTool);
        
        if (tempBlock != null && (isWallType(selectedTool) || "DOOR".equals(selectedTool))) {
            tempBlock.setRotation(ghostRotation);
        }
        this.ghostValid = result.valid;
        if (this.ghostValid && tempBlock != null) {
            this.ghostValid = gridModel.canPlace(tempBlock);
        }

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
            if ("DELETE".equals(selectedTool)) {
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
        BuildingBlock newBlock = createBlock(ghostX, ghostY, currentFloor, ghostRotation, ghostOrientation, selectedTool);

        if (newBlock != null) {
            if (isWallType(selectedTool) || "DOOR".equals(selectedTool)) {
                newBlock.setRotation(ghostRotation);
            }

            if (gridModel.canPlace(newBlock)) {
                gridModel.addBlock(newBlock);
                
                Map<ResourceType, Integer> cost = newBlock.getBuildCost();
                cost.forEach((k, v) -> totalConstructionCost.merge(k, v, (a, b) -> a + b));
                
                gameCanvas.invalidateCache();
                gameCanvas.draw();
                return true;
            }
        }
        return false;
    }

    private BuildingBlock createBlock(double x, double y, int z, double rotation, Orientation orientation, String tool) {
        // Prevent foundations on upper floors
        if (z > 0 && ("FOUNDATION".equals(tool) || "TRIANGLE".equals(tool))) {
            return null;
        }

        BuildingBlock block = null;
        if ("FOUNDATION".equals(tool)) {
            block = new Foundation(x, y, z, rotation);
        } else if ("TRIANGLE".equals(tool)) {
            block = new TriangleFoundation(x, y, z, rotation);
        } else if ("WALL".equals(tool)) {
            block = new Wall(x, y, z, orientation);
        } else if ("DOOR_FRAME".equals(tool)) {
            Wall doorWall = new Wall(x, y, z, orientation);
            doorWall.setType(BuildingType.DOORWAY);
            doorWall.setDoorType(selectedDoorType);
            block = doorWall;
        } else if ("WINDOW_FRAME".equals(tool)) {
            block = new Wall(x, y, z, orientation);
            block.setType(BuildingType.WINDOW_FRAME);
        } else if ("FLOOR".equals(tool)) {
            block = new Floor(x, y, z, rotation);
        } else if ("TRIANGLE_FLOOR".equals(tool)) {
            block = new TriangleFloor(x, y, z, rotation);
        } else if ("TC".equals(tool)) {
            block = new ToolCupboard(x, y, z, rotation);
        } else if ("WORKBENCH".equals(tool)) {
            block = new Workbench(x, y, z, rotation);
        } else if ("LOOT_ROOM".equals(tool)) {
            block = new LootRoom(x, y, z, rotation);
        } else if ("DOOR".equals(tool)) {
            block = new Door(x, y, z, orientation, selectedDoorType);
        }
        
        if (block != null) {
            if (!BuildingTypeUtils.isFurniture(block.getType()) && block.getType() != BuildingType.DOOR) {
                block.setTier(selectedTier);
            }
        }
        
        return block;
    }

    private boolean isWallType(String tool) {
        return BuildingTypeUtils.isWallTool(tool);
    }

    public void setSelectedTool(String tool) {
        this.selectedTool = tool;
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
