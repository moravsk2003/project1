package com.rustbuilder.ui;

import com.rustbuilder.config.GameConstants;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.BuildingType;

import javafx.geometry.Point2D;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

public class GameCanvas extends Canvas {
    private static final int TILE_SIZE = (int) GameConstants.TILE_SIZE;
    private static final double PSEUDO_X_SCALE = 0.75 * Math.sqrt(2.0);
    private static final double PSEUDO_Y_SCALE = 0.38 * Math.sqrt(2.0);
    private static final double PSEUDO_FLOOR_HEIGHT = 42.0;
    private static final double PSEUDO_SLAB_HEIGHT = 8.0;
    private static final double PSEUDO_WALL_HEIGHT = 46.0;
    private static final double PSEUDO_FURNITURE_HEIGHT = 24.0;
    private static final double DEFAULT_PSEUDO_YAW_DEGREES = 45.0;

    private enum RenderMode {
        TOP_DOWN,
        PSEUDO_3D
    }

    private final GridModel gridModel;
    private int currentFloor = 0;
    private com.rustbuilder.controller.GameController controller;
    private RenderMode renderMode = RenderMode.TOP_DOWN;

    // Ghost State
    private double ghostX;
    private double ghostY;
    private double ghostRotation;
    private String ghostType;
    private boolean ghostValid = true;
    private com.rustbuilder.model.core.Orientation ghostOrientation;
    private BuildingTier ghostTier;

    // Camera State
    private double offsetX = 0;
    private double offsetY = 0;
    private double scale = 1.0;
    private double topDownRotationDegrees = 0.0;
    private double topDownRotationCos = 1.0;
    private double topDownRotationSin = 0.0;
    private double pseudoYawDegrees = DEFAULT_PSEUDO_YAW_DEGREES;
    private double pseudoYawCos = Math.cos(Math.toRadians(DEFAULT_PSEUDO_YAW_DEGREES));
    private double pseudoYawSin = Math.sin(Math.toRadians(DEFAULT_PSEUDO_YAW_DEGREES));

    // Reusable objects for rendering to avoid GC pressure
    private final com.rustbuilder.model.structure.Wall reusableWall = new com.rustbuilder.model.structure.Wall(0, 0, 0, com.rustbuilder.model.core.Orientation.NORTH);
    private final double[] cachedXPts = new double[10];
    private final double[] cachedYPts = new double[10];
    private final double[] triangleXPts = new double[3];
    private final double[] triangleYPts = new double[3];
    private final java.util.ArrayList<PseudoFace> pseudoFacePool = new java.util.ArrayList<>(512);
    private final double[] pseudoBottomX = new double[4];
    private final double[] pseudoBottomY = new double[4];
    private final double[] pseudoTopX = new double[4];
    private final double[] pseudoTopY = new double[4];
    private final double[] pseudoDepth = new double[4];
    private int pseudoFaceCount = 0;

    private static final class PseudoFace {
        private final double[] x = new double[4];
        private final double[] y = new double[4];
        private int count;
        private double depth;
        private Color fill;
        private Color stroke;
        private double lineWidth;
    }

    public GameCanvas(GridModel gridModel, double width, double height) {
        super(width, height);
        this.gridModel = gridModel;
        draw();
    }
    
    public void setController(com.rustbuilder.controller.GameController controller) {
        this.controller = controller;
    }

    public void setGhost(double x, double y, double rotation, String type, boolean valid) {
        setGhost(x, y, rotation, type, valid, com.rustbuilder.model.core.Orientation.NORTH, null);
    }

    public void setGhost(double x, double y, double rotation, String type, boolean valid,
            com.rustbuilder.model.core.Orientation orientation, BuildingTier tier) {
        this.ghostX = x;
        this.ghostY = y;
        this.ghostRotation = rotation;
        this.ghostType = type;
        this.ghostValid = valid;
        this.ghostOrientation = orientation;
        this.ghostTier = tier;
    }

    public void setCurrentFloor(int floor) {
        this.currentFloor = floor;
        invalidateCache();
        draw();
    }

    private java.util.List<BuildingBlock> cachedRenderBlocks = null;

    public void invalidateCache() {
        this.cachedRenderBlocks = null;
    }

    public void pan(double dx, double dy) {
        this.offsetX += dx;
        this.offsetY += dy;
        draw();
    }

    public void moveCamera(double dx, double dy) {
        pan(dx, dy);
    }

    public void rotateCamera(double degrees) {
        double pivotX = getWidth() / 2.0;
        double pivotY = getHeight() / 2.0;
        Point2D pivotWorld = toGrid(pivotX, pivotY);

        if (isPseudo3D()) {
            setPseudoYawDegrees(pseudoYawDegrees + degrees);
        } else {
            setTopDownRotationDegrees(topDownRotationDegrees + degrees);
        }

        keepWorldPointAtScreenPoint(pivotWorld.getX(), pivotWorld.getY(), pivotX, pivotY);
        draw();
    }

    public void zoom(double factor, double pivotX, double pivotY) {
        Point2D pivotWorld = toGrid(pivotX, pivotY);
        double newScale = scale * factor;
        
        // Clamp scale
        if (newScale < 0.1) newScale = 0.1;
        if (newScale > 5.0) newScale = 5.0;
        
        this.scale = newScale;
        keepWorldPointAtScreenPoint(pivotWorld.getX(), pivotWorld.getY(), pivotX, pivotY);
        
        draw();
    }

    public void toggleRenderMode() {
        double pivotX = getWidth() / 2.0;
        double pivotY = getHeight() / 2.0;
        Point2D pivotWorld = toGrid(pivotX, pivotY);
        renderMode = renderMode == RenderMode.TOP_DOWN ? RenderMode.PSEUDO_3D : RenderMode.TOP_DOWN;
        keepWorldPointAtScreenPoint(pivotWorld.getX(), pivotWorld.getY(), pivotX, pivotY);
        draw();
    }

    public boolean isPseudo3D() {
        return renderMode == RenderMode.PSEUDO_3D;
    }

    public Point2D toGrid(double screenX, double screenY) {
        double renderX = (screenX - offsetX) / scale;
        double renderY = (screenY - offsetY) / scale;

        if (!isPseudo3D()) {
            double worldX = renderX * topDownRotationCos + renderY * topDownRotationSin;
            double worldY = -renderX * topDownRotationSin + renderY * topDownRotationCos;
            return new Point2D(worldX, worldY);
        }

        double viewX = renderX / PSEUDO_X_SCALE;
        double viewDepth = (renderY + currentFloor * PSEUDO_FLOOR_HEIGHT) / PSEUDO_Y_SCALE;
        double worldX = viewX * pseudoYawCos + viewDepth * pseudoYawSin;
        double worldY = -viewX * pseudoYawSin + viewDepth * pseudoYawCos;
        return new Point2D(worldX, worldY);
    }

    public double toGridX(double screenX) {
        return (screenX - offsetX) / scale;
    }

    public double toGridY(double screenY) {
        return (screenY - offsetY) / scale;
    }

    private void keepWorldPointAtScreenPoint(double worldX, double worldY, double screenX, double screenY) {
        double renderX;
        double renderY;

        if (isPseudo3D()) {
            renderX = projectX(worldX, worldY);
            renderY = projectY(worldX, worldY, currentFloor, 0.0);
        } else {
            renderX = worldX * topDownRotationCos - worldY * topDownRotationSin;
            renderY = worldX * topDownRotationSin + worldY * topDownRotationCos;
        }

        offsetX = screenX - renderX * scale;
        offsetY = screenY - renderY * scale;
    }

    private void setTopDownRotationDegrees(double degrees) {
        topDownRotationDegrees = normalizeDegrees(degrees);
        double radians = Math.toRadians(topDownRotationDegrees);
        topDownRotationCos = Math.cos(radians);
        topDownRotationSin = Math.sin(radians);
    }

    private void setPseudoYawDegrees(double degrees) {
        pseudoYawDegrees = normalizeDegrees(degrees);
        double radians = Math.toRadians(pseudoYawDegrees);
        pseudoYawCos = Math.cos(radians);
        pseudoYawSin = Math.sin(radians);
    }

    private double normalizeDegrees(double degrees) {
        double normalized = degrees % 360.0;
        if (normalized < 0) {
            normalized += 360.0;
        }
        return normalized;
    }

    public void draw() {
        GraphicsContext gc = getGraphicsContext2D();
        gc.clearRect(0, 0, getWidth(), getHeight());

        gc.save();
        // Apply Camera Transform
        gc.translate(offsetX, offsetY);
        gc.scale(scale, scale);

        if (isPseudo3D()) {
            drawPseudo3DGrid(gc);
            drawPseudo3DBlocks(gc);
            drawPseudo3DGhost(gc);

            if (showStabilityTool) {
                drawPseudo3DStabilityOverlay(gc);
            }
        } else {
            if (topDownRotationDegrees != 0.0) {
                gc.rotate(topDownRotationDegrees);
            }
            drawGrid(gc);
            drawBlocks(gc);
            drawGhost(gc);
            
            if (showStabilityTool) {
                drawStabilityOverlay(gc);
            }
        }
        
        gc.restore();
        
        // Draw UI (Screen Space)
        drawFloorIndicator(gc);
    }

    private void drawFloorIndicator(GraphicsContext gc) {
        gc.setFill(Color.BLACK);
        gc.fillText("Floor: " + currentFloor, 10, 20);
        
        if (controller == null) return;
        
        int y = 50;
        
        // 1. Total Construction Cost (Always Visible)
        gc.fillText("Construction Cost:", 10, y);
        y += 20;
        java.util.Map<com.rustbuilder.model.core.ResourceType, Integer> totalCost = controller.getTotalConstructionCost();
        if (totalCost != null) {
            for (com.rustbuilder.model.core.ResourceType type : com.rustbuilder.model.core.ResourceType.values()) {
                Integer amount = totalCost.getOrDefault(type, 0);
                if (amount > 0) {
                     gc.fillText("Total " + type.name() + ": " + amount, 20, y);
                     y += 20;
                }
            }
        }
        
        // 2. Upkeep (Only if TC selected)
        BuildingBlock selected = controller.getSelectedBlock();
        if (selected != null && selected.getType() == BuildingType.TC) {
            y += 10;
            gc.fillText("Upkeep / 24h:", 10, y);
            y += 20;
            
            // Re-calculate upkeep or fetch?
            java.util.Map<com.rustbuilder.model.core.ResourceType, Integer> upkeep = 
                com.rustbuilder.service.economy.UpkeepService.calculateUpkeep(gridModel.getAllBlocks());
    
            for (com.rustbuilder.model.core.ResourceType type : com.rustbuilder.model.core.ResourceType.values()) {
                Integer amount = upkeep.getOrDefault(type, 0);
                if (amount > 0) {
                     gc.fillText(type.name() + ": " + amount, 20, y);
                     y += 20;
                }
            }
        }
    }

    private void drawGhost(GraphicsContext gc) {
        if (ghostType == null)
            return;

        gc.setGlobalAlpha(0.5); // Transparent
        
        // Ghost Color based on Tier (if valid)
        Color baseColor = Color.BLUE;
        if (ghostTier != null) {
             baseColor = getColorForTier(ghostTier);
        }
        
        if (ghostValid) {
            gc.setFill(baseColor);
        } else {
            gc.setFill(Color.RED);
        }

        if ("FOUNDATION".equals(ghostType)) {
            drawRotatedRect(gc, ghostX, ghostY, TILE_SIZE, TILE_SIZE, ghostRotation);
        } else if ("TRIANGLE_FOUNDATION".equals(ghostType) || "TRIANGLE".equals(ghostType)) {
            drawTriangle(gc, ghostX, ghostY, ghostRotation);
        } else if ("WALL".equals(ghostType) || "DOOR_FRAME".equals(ghostType) || "WINDOW_FRAME".equals(ghostType) || "DOORWAY".equals(ghostType) || "DOOR".equals(ghostType)) {
            gc.save();
            gc.translate(ghostX, ghostY); // Малюємо відносно координат миші

            // ОНОВЛЮЄМО СТІНУ ЛИШЕ ЯКЩО ЗМІНИЛИСЯ ПАРАМЕТРИ
            if (reusableWall.getOrientation() != ghostOrientation || reusableWall.getRotation() != ghostRotation || 
                reusableWall.getX() != 0 || reusableWall.getY() != 0) {
                reusableWall.setX(0);
                reusableWall.setY(0);
                reusableWall.setZ(0);
                reusableWall.setOrientation(ghostOrientation);
                reusableWall.setRotation(ghostRotation);
                // Тут під капотом спрацює invalidateGeometryCache, але ТІЛЬКИ при реальній зміні
            }

            // Використовуємо кешовані масиви (без алокації пам'яті)
            double[] points = reusableWall.getPolygonPoints();
            int n = points.length / 2;
            for (int i = 0; i < n; i++) {
                cachedXPts[i] = points[i * 2];
                cachedYPts[i] = points[i * 2 + 1];
            }

            if ("DOOR".equals(ghostType) && ghostTier != null && ghostValid) {
                switch (controller.getSelectedDoorType()) {
                    case SHEET_METAL: gc.setFill(Color.SILVER); break;
                    case GARAGE:      gc.setFill(Color.SLATEGRAY); break;
                    case ARMORED:     gc.setFill(Color.DARKSLATEGRAY); break;
                }
                gc.setStroke(Color.BLACK);
                gc.strokePolygon(cachedXPts, cachedYPts, n);
            }

            gc.fillPolygon(cachedXPts, cachedYPts, n);
            gc.restore();
        } else if ("FLOOR".equals(ghostType)) {
            if (ghostRotation == 0) {
                gc.fillRect(ghostX, ghostY, TILE_SIZE, TILE_SIZE);
                gc.strokeRect(ghostX, ghostY, TILE_SIZE, TILE_SIZE);
            } else {
                drawRotatedRect(gc, ghostX, ghostY, TILE_SIZE, TILE_SIZE, ghostRotation);
            }
        } else if ("TRIANGLE_FLOOR".equals(ghostType)) {
            drawTriangle(gc, ghostX, ghostY, ghostRotation);
        } else if ("TC".equals(ghostType) || "WORKBENCH".equals(ghostType) || "LOOT_ROOM".equals(ghostType)) {
             gc.fillRect(ghostX + 18, ghostY + 22, TILE_SIZE - 36, TILE_SIZE - 44);
        }

        gc.setGlobalAlpha(1.0); // Reset
    }

    private void drawGrid(GraphicsContext gc) {
        gc.setStroke(Color.LIGHTGRAY);
        gc.setLineWidth(0.5);

        // Draw world grid within a fixed 2000x2000 area
        
        int worldSize = 2000;
        for (double x = 0; x < worldSize; x += TILE_SIZE) {
            gc.strokeLine(x, 0, x, worldSize);
        }
        for (double y = 0; y < worldSize; y += TILE_SIZE) {
            gc.strokeLine(0, y, worldSize, y);
        }
    }

    private void drawBlocks(GraphicsContext gc) {
        if (cachedRenderBlocks == null) {
            cachedRenderBlocks = gridModel.getAllBlocks().stream()
                    .filter(b -> b.getZ() <= currentFloor)
                    .sorted((b1, b2) -> {
                        int zCompare = Integer.compare(b1.getZ(), b2.getZ());
                        if (zCompare != 0) return zCompare;
                        
                        // Same floor: Draw Foundations/Floors first, then Walls
                        boolean b1IsBase = b1.getType() == BuildingType.FOUNDATION || 
                                           b1.getType() == BuildingType.TRIANGLE_FOUNDATION ||
                                           b1.getType() == BuildingType.FLOOR ||
                                           b1.getType() == BuildingType.TRIANGLE_FLOOR;
                                           
                        boolean b2IsBase = b2.getType() == BuildingType.FOUNDATION || 
                                           b2.getType() == BuildingType.TRIANGLE_FOUNDATION ||
                                           b2.getType() == BuildingType.FLOOR ||
                                           b2.getType() == BuildingType.TRIANGLE_FLOOR;
                                           
                        if (b1IsBase && !b2IsBase) return -1;
                        if (!b1IsBase && b2IsBase) return 1;
                        return 0;
                    })
                    .collect(java.util.stream.Collectors.toList());
        }
        cachedRenderBlocks.forEach(block -> drawBlock(gc, block));
    }

    private void drawPseudo3DGrid(GraphicsContext gc) {
        gc.setStroke(Color.rgb(190, 198, 205, 0.65));
        gc.setLineWidth(0.7);

        int worldSize = 2000;
        for (double x = 0; x <= worldSize; x += TILE_SIZE) {
            Point2D a = project(x, 0, currentFloor);
            Point2D b = project(x, worldSize, currentFloor);
            gc.strokeLine(a.getX(), a.getY(), b.getX(), b.getY());
        }
        for (double y = 0; y <= worldSize; y += TILE_SIZE) {
            Point2D a = project(0, y, currentFloor);
            Point2D b = project(worldSize, y, currentFloor);
            gc.strokeLine(a.getX(), a.getY(), b.getX(), b.getY());
        }
    }

    private void drawPseudo3DBlocks(GraphicsContext gc) {
        pseudoFaceCount = 0;

        for (BuildingBlock block : gridModel.getAllBlocks()) {
            if (block.getZ() <= currentFloor) {
                queuePseudo3DBlock(block);
            }
        }

        pseudoFacePool.subList(0, pseudoFaceCount)
                .sort((f1, f2) -> Double.compare(f1.depth, f2.depth));

        for (int i = 0; i < pseudoFaceCount; i++) {
            PseudoFace face = pseudoFacePool.get(i);
            gc.setFill(face.fill);
            gc.fillPolygon(face.x, face.y, face.count);
            gc.setStroke(face.stroke);
            gc.setLineWidth(face.lineWidth);
            gc.strokePolygon(face.x, face.y, face.count);
        }

        gc.setLineWidth(1.0);
        for (BuildingBlock block : gridModel.getAllBlocks()) {
            if (block.getZ() <= currentFloor) {
                drawPseudo3DLabel(gc, block);
            }
        }
    }

    private void queuePseudo3DBlock(BuildingBlock block) {
        double[] points = getPseudo3DFootprint(block);
        if (points == null || points.length < 6) {
            return;
        }

        Color color = getPseudo3DColor(block);
        if (block.getZ() < currentFloor) {
            color = color.deriveColor(0, 1, 0.58, 0.85);
        }

        queuePseudo3DPrismFaces(points, block.getZ(), getPseudo3DHeight(block.getType()), color);
    }

    private void drawPseudo3DGhost(GraphicsContext gc) {
        if (ghostType == null) {
            return;
        }

        double[] points = getPseudo3DGhostFootprint();
        if (points == null || points.length < 6) {
            return;
        }

        Color color = ghostValid ? (ghostTier == null ? Color.CORNFLOWERBLUE : getColorForTier(ghostTier)) : Color.RED;
        gc.save();
        gc.setGlobalAlpha(0.45);
        drawPseudo3DPrism(gc, points, currentFloor, getPseudo3DHeight(toBuildingType(ghostType)), color);
        gc.restore();
    }

    private void drawPseudo3DStabilityOverlay(GraphicsContext gc) {
        if (hoveredBlock == null) {
            return;
        }

        double[] points = getPseudo3DFootprint(hoveredBlock);
        if (points == null || points.length < 6) {
            return;
        }

        double[] projectedX = new double[points.length / 2];
        double[] projectedY = new double[points.length / 2];
        for (int i = 0; i < projectedX.length; i++) {
            Point2D p = projectElevated(points[i * 2], points[i * 2 + 1], hoveredBlock.getZ(), getPseudo3DHeight(hoveredBlock.getType()));
            projectedX[i] = p.getX();
            projectedY[i] = p.getY();
        }

        gc.setStroke(Color.CYAN);
        gc.setLineWidth(2.0);
        gc.strokePolygon(projectedX, projectedY, projectedX.length);

        Point2D labelPoint = projectElevated(hoveredBlock.getX() + TILE_SIZE / 2.0, hoveredBlock.getY() + TILE_SIZE / 2.0,
                hoveredBlock.getZ(), getPseudo3DHeight(hoveredBlock.getType()) + 14.0);
        gc.setFill(Color.rgb(0, 0, 0, 0.7));
        gc.fillRect(labelPoint.getX() - 38, labelPoint.getY() - 16, 82, 20);
        gc.setFill(Color.WHITE);
        gc.fillText(String.format("Stab: %.0f%%", hoveredBlock.getStability() * 100), labelPoint.getX() - 32, labelPoint.getY() - 2);
    }

    private void queuePseudo3DPrismFaces(double[] points, int floor, double height, Color color) {
        int n = points.length / 2;
        if (n < 3 || n > 4) {
            return;
        }

        double depthSum = 0.0;
        for (int i = 0; i < n; i++) {
            double x = points[i * 2];
            double y = points[i * 2 + 1];
            pseudoBottomX[i] = projectX(x, y);
            pseudoBottomY[i] = projectY(x, y, floor, 0.0);
            pseudoTopX[i] = pseudoBottomX[i];
            pseudoTopY[i] = projectY(x, y, floor, height);
            pseudoDepth[i] = viewDepth(x, y);
            depthSum += pseudoDepth[i];
        }

        double floorBias = floor * TILE_SIZE;
        Color stroke = Color.rgb(20, 24, 28, 0.82);
        for (int i = 0; i < n; i++) {
            int next = (i + 1) % n;
            PseudoFace side = acquirePseudoFace();
            side.count = 4;
            side.depth = ((pseudoDepth[i] + pseudoDepth[next]) / 2.0) + floorBias;
            side.fill = color.deriveColor(0, 0.85, 0.72 + (i % 2) * 0.12, 1.0);
            side.stroke = Color.rgb(35, 40, 45, 0.55);
            side.lineWidth = 1.0;
            side.x[0] = pseudoBottomX[i];
            side.y[0] = pseudoBottomY[i];
            side.x[1] = pseudoBottomX[next];
            side.y[1] = pseudoBottomY[next];
            side.x[2] = pseudoTopX[next];
            side.y[2] = pseudoTopY[next];
            side.x[3] = pseudoTopX[i];
            side.y[3] = pseudoTopY[i];
        }

        PseudoFace top = acquirePseudoFace();
        top.count = n;
        top.depth = (depthSum / n) + floorBias + 0.35;
        top.fill = color.brighter();
        top.stroke = stroke;
        top.lineWidth = 1.0;
        for (int i = 0; i < n; i++) {
            top.x[i] = pseudoTopX[i];
            top.y[i] = pseudoTopY[i];
        }
    }

    private PseudoFace acquirePseudoFace() {
        if (pseudoFaceCount == pseudoFacePool.size()) {
            pseudoFacePool.add(new PseudoFace());
        }
        return pseudoFacePool.get(pseudoFaceCount++);
    }

    private void drawPseudo3DPrism(GraphicsContext gc, double[] points, int floor, double height, Color color) {
        int n = points.length / 2;
        double[] bottomX = new double[n];
        double[] bottomY = new double[n];
        double[] topX = new double[n];
        double[] topY = new double[n];

        for (int i = 0; i < n; i++) {
            double x = points[i * 2];
            double y = points[i * 2 + 1];
            Point2D bottom = project(x, y, floor);
            Point2D top = projectElevated(x, y, floor, height);
            bottomX[i] = bottom.getX();
            bottomY[i] = bottom.getY();
            topX[i] = top.getX();
            topY[i] = top.getY();
        }

        for (int i = 0; i < n; i++) {
            int next = (i + 1) % n;
            double[] sideX = { bottomX[i], bottomX[next], topX[next], topX[i] };
            double[] sideY = { bottomY[i], bottomY[next], topY[next], topY[i] };
            gc.setFill(color.deriveColor(0, 0.85, 0.72 + (i % 2) * 0.12, 1));
            gc.fillPolygon(sideX, sideY, 4);
            gc.setStroke(Color.rgb(35, 40, 45, 0.55));
            gc.strokePolygon(sideX, sideY, 4);
        }

        gc.setFill(color.brighter());
        gc.fillPolygon(topX, topY, n);
        gc.setStroke(Color.rgb(20, 24, 28, 0.82));
        gc.setLineWidth(1.0);
        gc.strokePolygon(topX, topY, n);
    }

    private Point2D project(double x, double y, int floor) {
        return new Point2D(projectX(x, y), projectY(x, y, floor, 0.0));
    }

    private Point2D projectElevated(double x, double y, int floor, double elevation) {
        return new Point2D(projectX(x, y), projectY(x, y, floor, elevation));
    }

    private double projectX(double x, double y) {
        return (x * pseudoYawCos - y * pseudoYawSin) * PSEUDO_X_SCALE;
    }

    private double projectY(double x, double y, int floor, double elevation) {
        return viewDepth(x, y) * PSEUDO_Y_SCALE - floor * PSEUDO_FLOOR_HEIGHT - elevation;
    }

    private double viewDepth(double x, double y) {
        return x * pseudoYawSin + y * pseudoYawCos;
    }

    private double[] getPseudo3DFootprint(BuildingBlock block) {
        if (block.getType() == BuildingType.DOOR && block instanceof com.rustbuilder.model.structure.Door) {
            com.rustbuilder.model.structure.Door door = (com.rustbuilder.model.structure.Door) block;
            reusableWall.setX(block.getX());
            reusableWall.setY(block.getY());
            reusableWall.setZ(block.getZ());
            reusableWall.setOrientation(door.getOrientation());
            reusableWall.setRotation(block.getRotation());
            return reusableWall.getPolygonPoints();
        }

        return block.getPolygonPoints();
    }

    private double[] getPseudo3DGhostFootprint() {
        if ("FOUNDATION".equals(ghostType) || "FLOOR".equals(ghostType)) {
            return rectanglePoints(ghostX, ghostY, TILE_SIZE, TILE_SIZE, ghostRotation);
        }
        if ("TRIANGLE_FOUNDATION".equals(ghostType) || "TRIANGLE".equals(ghostType) || "TRIANGLE_FLOOR".equals(ghostType)) {
            return trianglePoints(ghostX, ghostY, ghostRotation);
        }
        if ("WALL".equals(ghostType) || "DOOR_FRAME".equals(ghostType) || "WINDOW_FRAME".equals(ghostType) || "DOORWAY".equals(ghostType) || "DOOR".equals(ghostType)) {
            reusableWall.setX(ghostX);
            reusableWall.setY(ghostY);
            reusableWall.setZ(currentFloor);
            reusableWall.setOrientation(ghostOrientation == null ? com.rustbuilder.model.core.Orientation.NORTH : ghostOrientation);
            reusableWall.setRotation(ghostRotation);
            return reusableWall.getPolygonPoints();
        }
        if ("TC".equals(ghostType)) {
            return rectanglePoints(ghostX + 18, ghostY + 22, TILE_SIZE - 36, TILE_SIZE - 44, 0);
        }
        if ("WORKBENCH".equals(ghostType)) {
            return rectanglePoints(ghostX + 8, ghostY + 12, TILE_SIZE - 16, TILE_SIZE - 24, 0);
        }
        if ("LOOT_ROOM".equals(ghostType)) {
            return rectanglePoints(ghostX + 10, ghostY + 14, TILE_SIZE - 20, TILE_SIZE - 28, 0);
        }
        return null;
    }

    private double[] rectanglePoints(double x, double y, double w, double h, double angle) {
        double cx = x + w / 2.0;
        double cy = y + h / 2.0;
        double rotRad = Math.toRadians(angle);
        double cos = Math.cos(rotRad);
        double sin = Math.sin(rotRad);
        double[] local = { -w / 2.0, -h / 2.0, w / 2.0, -h / 2.0, w / 2.0, h / 2.0, -w / 2.0, h / 2.0 };
        double[] points = new double[8];
        for (int i = 0; i < 4; i++) {
            double px = local[i * 2];
            double py = local[i * 2 + 1];
            points[i * 2] = cx + px * cos - py * sin;
            points[i * 2 + 1] = cy + px * sin + py * cos;
        }
        return points;
    }

    private double[] trianglePoints(double x, double y, double angle) {
        double cx = x + TILE_SIZE / 2.0;
        double cy = y + TILE_SIZE / 2.0;
        double[] local = {
                -TILE_SIZE / 2.0, TILE_SIZE / 2.0,
                0, (TILE_SIZE - TILE_SIZE * 0.866) - TILE_SIZE / 2.0,
                TILE_SIZE / 2.0, TILE_SIZE / 2.0
        };
        double rotRad = Math.toRadians(angle);
        double cos = Math.cos(rotRad);
        double sin = Math.sin(rotRad);
        double[] points = new double[6];
        for (int i = 0; i < 3; i++) {
            double px = local[i * 2];
            double py = local[i * 2 + 1];
            points[i * 2] = cx + px * cos - py * sin;
            points[i * 2 + 1] = cy + px * sin + py * cos;
        }
        return points;
    }

    private double getPseudo3DHeight(BuildingType type) {
        if (type == null) {
            return PSEUDO_SLAB_HEIGHT;
        }
        switch (type) {
            case WALL:
            case DOORWAY:
            case WINDOW_FRAME:
            case DOOR:
                return PSEUDO_WALL_HEIGHT;
            case TC:
            case WORKBENCH:
            case LOOT_ROOM:
                return PSEUDO_FURNITURE_HEIGHT;
            default:
                return PSEUDO_SLAB_HEIGHT;
        }
    }

    private BuildingType toBuildingType(String type) {
        if (type == null) {
            return null;
        }
        switch (type) {
            case "FOUNDATION":
                return BuildingType.FOUNDATION;
            case "TRIANGLE":
            case "TRIANGLE_FOUNDATION":
                return BuildingType.TRIANGLE_FOUNDATION;
            case "FLOOR":
                return BuildingType.FLOOR;
            case "TRIANGLE_FLOOR":
                return BuildingType.TRIANGLE_FLOOR;
            case "WALL":
                return BuildingType.WALL;
            case "DOOR_FRAME":
            case "DOORWAY":
                return BuildingType.DOORWAY;
            case "WINDOW_FRAME":
                return BuildingType.WINDOW_FRAME;
            case "DOOR":
                return BuildingType.DOOR;
            case "TC":
                return BuildingType.TC;
            case "WORKBENCH":
                return BuildingType.WORKBENCH;
            case "LOOT_ROOM":
                return BuildingType.LOOT_ROOM;
            default:
                return null;
        }
    }

    private Color getPseudo3DColor(BuildingBlock block) {
        if (block.getType() == BuildingType.TC) {
            return Color.RED;
        }
        if (block.getType() == BuildingType.WORKBENCH) {
            return Color.ORANGE;
        }
        if (block.getType() == BuildingType.LOOT_ROOM) {
            return Color.GOLD;
        }
        if (block.getType() == BuildingType.DOOR && block instanceof com.rustbuilder.model.structure.Door) {
            com.rustbuilder.model.structure.Door door = (com.rustbuilder.model.structure.Door) block;
            switch (door.getDoorType()) {
                case SHEET_METAL: return Color.SILVER;
                case GARAGE:      return Color.SLATEGRAY;
                case ARMORED:     return Color.DARKSLATEGRAY;
            }
        }
        return getColorForTier(block.getTier());
    }

    private void drawPseudo3DLabel(GraphicsContext gc, BuildingBlock block) {
        String label = null;
        if (block.getType() == BuildingType.TC) {
            label = "TC";
        } else if (block.getType() == BuildingType.WORKBENCH) {
            label = "WB";
        } else if (block.getType() == BuildingType.LOOT_ROOM) {
            label = "LR";
        } else if (block.getType() == BuildingType.DOORWAY) {
            label = "DF";
        } else if (block.getType() == BuildingType.WINDOW_FRAME) {
            label = "WF";
        }

        if (label == null) {
            return;
        }

        Point2D p = projectElevated(block.getX() + TILE_SIZE / 2.0, block.getY() + TILE_SIZE / 2.0,
                block.getZ(), getPseudo3DHeight(block.getType()) + 2.0);
        gc.setFill(Color.WHITE);
        gc.setFont(javafx.scene.text.Font.font("Arial", javafx.scene.text.FontWeight.BOLD, 9));
        gc.fillText(label, p.getX() - 7, p.getY() + 3);
    }

    private void drawBlock(GraphicsContext gc, BuildingBlock block) {
        double x = block.getX();
        double y = block.getY();

        Color color = getColorForTier(block.getTier());
        
        // Dim lower floors
        if (block.getZ() < currentFloor) {
             color = color.deriveColor(0, 1, 0.6, 1);
        }
        
        gc.setFill(color);
        gc.setStroke(Color.BLACK);

        if (block.getType() == BuildingType.FOUNDATION) {
            if (block.getRotation() == 0) {
                gc.fillRect(x, y, TILE_SIZE, TILE_SIZE);
                gc.strokeRect(x, y, TILE_SIZE, TILE_SIZE);
            } else {
                drawRotatedRect(gc, x, y, TILE_SIZE, TILE_SIZE, block.getRotation());
            }

        } else if (block.getType() == BuildingType.TRIANGLE_FOUNDATION
                || block.getType() == BuildingType.TRIANGLE_FLOOR) {
            drawTriangle(gc, x, y, block.getRotation());
        } else if (block.getType() == BuildingType.FLOOR) {
            if (block.getRotation() == 0) {
                gc.fillRect(x, y, TILE_SIZE, TILE_SIZE);
                gc.strokeRect(x, y, TILE_SIZE, TILE_SIZE);
            } else {
                drawRotatedRect(gc, x, y, TILE_SIZE, TILE_SIZE, block.getRotation());
            }
        } else if (block.getType() == BuildingType.WALL || block.getType() == BuildingType.DOORWAY
                || block.getType() == BuildingType.WINDOW_FRAME) {
            if (block instanceof com.rustbuilder.model.structure.Wall) {
                // Use cached split arrays — no allocation in hot render path
                double[] xPoints = block.getSplitPolygonX();
                double[] yPoints = block.getSplitPolygonY();

                if (block.getType() == BuildingType.DOORWAY) {
                    // --- DOORWAY: draw as a frame with an opening ---
                    // 1. Fill with a lighter, semi-transparent version of the tier color
                    gc.setFill(color.deriveColor(0, 0.4, 1.3, 0.35));
                    gc.fillPolygon(xPoints, yPoints, xPoints.length);

                    // 2. Draw thick colored border to represent the frame
                    gc.setStroke(color.darker());
                    gc.setLineWidth(3.0);
                    gc.strokePolygon(xPoints, yPoints, xPoints.length);

                    // 3. Draw a dashed center line to indicate the opening
                    gc.setStroke(color.darker().darker());
                    gc.setLineDashes(4, 4);
                    gc.setLineWidth(1.5);
                    double centerX = 0, centerY = 0;
                    for (int i = 0; i < xPoints.length; i++) {
                        centerX += xPoints[i];
                        centerY += yPoints[i];
                    }
                    centerX /= xPoints.length;
                    centerY /= yPoints.length;

                    // Draw "DF" label at center
                    gc.setLineDashes(null);
                    gc.setFill(Color.WHITE);
                    gc.setFont(javafx.scene.text.Font.font("Arial", javafx.scene.text.FontWeight.BOLD, 9));
                    gc.fillText("DF", centerX - 6, centerY + 3);

                    // Reset line width
                    gc.setLineWidth(1.0);
                    gc.setStroke(Color.BLACK);

                } else if (block.getType() == BuildingType.WINDOW_FRAME) {
                    // --- WINDOW FRAME: draw with cross-hatch pattern ---
                    gc.setFill(color.deriveColor(200, 0.6, 1.1, 0.7));
                    gc.fillPolygon(xPoints, yPoints, xPoints.length);

                    // Cross lines inside the polygon to indicate a window
                    gc.setStroke(color.darker());
                    gc.setLineDashes(3, 3);
                    gc.setLineWidth(1.0);
                    double cx = 0, cy = 0;
                    for (int i = 0; i < xPoints.length; i++) {
                        cx += xPoints[i];
                        cy += yPoints[i];
                    }
                    cx /= xPoints.length;
                    cy /= yPoints.length;

                    gc.setLineDashes(null);
                    gc.setStroke(Color.BLACK);
                    gc.strokePolygon(xPoints, yPoints, xPoints.length);

                    // "WF" label
                    gc.setFill(Color.WHITE);
                    gc.setFont(javafx.scene.text.Font.font("Arial", javafx.scene.text.FontWeight.BOLD, 9));
                    gc.fillText("WF", cx - 7, cy + 3);

                } else {
                    // --- Regular WALL: solid fill ---
                    gc.setFill(color);
                    gc.fillPolygon(xPoints, yPoints, xPoints.length);
                    gc.setStroke(Color.BLACK);
                    gc.strokePolygon(xPoints, yPoints, xPoints.length);
                }
            }
        } else if (block.getType() == BuildingType.TC) {
            gc.setFill(Color.RED);
            gc.fillRect(x + 18, y + 22, TILE_SIZE - 36, TILE_SIZE - 44);
            gc.setStroke(Color.DARKRED);
            gc.strokeRect(x + 18, y + 22, TILE_SIZE - 36, TILE_SIZE - 44);
            gc.setFill(Color.WHITE);
            gc.fillText("TC", x + TILE_SIZE / 2.0 - 7, y + TILE_SIZE / 2.0 + 4);
        } else if (block.getType() == BuildingType.WORKBENCH) {
            gc.setFill(Color.ORANGE);
            gc.fillRect(x + 8, y + 12, TILE_SIZE - 16, TILE_SIZE - 24);
            gc.setStroke(Color.DARKORANGE);
            gc.strokeRect(x + 8, y + 12, TILE_SIZE - 16, TILE_SIZE - 24);
            // Label
            gc.setFill(Color.BLACK);
            gc.fillText("WB", x + TILE_SIZE / 2 - 8, y + TILE_SIZE / 2 + 4);
        } else if (block.getType() == BuildingType.LOOT_ROOM) {
            gc.setFill(Color.GOLD);
            gc.fillRect(x + 10, y + 14, TILE_SIZE - 20, TILE_SIZE - 28);
            gc.setStroke(Color.GOLDENROD);
            gc.strokeRect(x + 10, y + 14, TILE_SIZE - 20, TILE_SIZE - 28);
            gc.setFill(Color.BLACK);
            gc.fillText("LR", x + TILE_SIZE / 2 - 8, y + TILE_SIZE / 2 + 4);
        } else if (block.getType() == BuildingType.DOOR) {
            // Draw door inside its doorway — same shape but different color
            if (block instanceof com.rustbuilder.model.structure.Door) {
                com.rustbuilder.model.structure.Door door = (com.rustbuilder.model.structure.Door) block;
                reusableWall.setX(x);
                reusableWall.setY(y);
                reusableWall.setZ(block.getZ());
                reusableWall.setOrientation(door.getOrientation());
                reusableWall.setRotation(block.getRotation());
                double[] points = reusableWall.getPolygonPoints();
                int n = points.length / 2;
                for (int i = 0; i < n; i++) {
                    cachedXPts[i] = points[i * 2];
                    cachedYPts[i] = points[i * 2 + 1];
                }
                // Color based on door type
                switch (door.getDoorType()) {
                    case SHEET_METAL: gc.setFill(Color.SILVER); break;
                    case GARAGE:      gc.setFill(Color.SLATEGRAY); break;
                    case ARMORED:     gc.setFill(Color.DARKSLATEGRAY); break;
                }
                gc.fillPolygon(cachedXPts, cachedYPts, n);
                gc.setStroke(Color.BLACK);
                gc.strokePolygon(cachedXPts, cachedYPts, n);
            }
        }
    }

    private void drawRotatedRect(GraphicsContext gc, double x, double y, double w, double h, double angle) {
        gc.save();
        double cx = x + w / 2;
        double cy = y + h / 2;
        gc.translate(cx, cy);
        gc.rotate(angle);
        gc.translate(-cx, -cy);
        gc.fillRect(x, y, w, h);
        gc.strokeRect(x, y, w, h);
        gc.restore();
    }

    private void drawTriangle(GraphicsContext gc, double x, double y, double rot) {
        gc.save();
        double cx = x + TILE_SIZE / 2.0;
        double cy = y + TILE_SIZE / 2.0;
        gc.translate(cx, cy);
        gc.rotate(rot);
        gc.translate(-cx, -cy);

        // Equilateral Triangle (Up/North by default)
        triangleXPts[0] = x;
        triangleYPts[0] = y + TILE_SIZE;
        triangleXPts[1] = x + TILE_SIZE / 2.0;
        triangleYPts[1] = y + (TILE_SIZE - (TILE_SIZE * 0.866));
        triangleXPts[2] = x + TILE_SIZE;
        triangleYPts[2] = y + TILE_SIZE;

        gc.fillPolygon(triangleXPts, triangleYPts, 3);
        gc.strokePolygon(triangleXPts, triangleYPts, 3);
        gc.restore();
    }

    private BuildingBlock hoveredBlock;
    private boolean showStabilityTool = false;

    public void setShowStabilityTool(boolean show) {
        this.showStabilityTool = show;
        draw();
    }
    
    public boolean isStabilityToolActive() {
        return showStabilityTool;
    }

    public void setHoveredBlock(BuildingBlock block) {
        this.hoveredBlock = block;
        draw();
    }

    public BuildingBlock hitTest(double screenX, double screenY) {
        Point2D world = toGrid(screenX, screenY);
        double worldX = world.getX();
        double worldY = world.getY();
        
        // Reverse iteration to check top-most blocks first
        java.util.List<BuildingBlock> blocks = new java.util.ArrayList<>(gridModel.getAllBlocks());
        java.util.Collections.reverse(blocks);
        
        for (BuildingBlock b : blocks) {
            // Only check blocks on current floor (or slightly below/above if visible?)
            // For now restrict to current floor for easy selection
            if (b.getZ() != currentFloor) continue;
            
            double[] points = b.getPolygonPoints();
            if (contains(points, worldX, worldY)) {
                return b;
            }
        }
        return null;
    }

    private boolean contains(double[] points, double x, double y) {
        return com.rustbuilder.util.GeometryUtils.isPointInPolygon(x, y, points);
    }

    private void drawStabilityOverlay(GraphicsContext gc) {
        if (hoveredBlock != null) {
            double x = hoveredBlock.getX();
            double y = hoveredBlock.getY();
            double stability = hoveredBlock.getStability();
            
            // Draw highlight border
            gc.setStroke(Color.CYAN);
            gc.setLineWidth(2);
            if (hoveredBlock.getType() == BuildingType.FOUNDATION || hoveredBlock.getType() == BuildingType.FLOOR) {
                 gc.strokeRect(x, y, TILE_SIZE, TILE_SIZE);
            } else if (hoveredBlock.getType() == BuildingType.TRIANGLE_FOUNDATION || hoveredBlock.getType() == BuildingType.TRIANGLE_FLOOR) {
                 // Simplified triangle highlight (box for now)
                 // Or re-use drawTriangle logic?
                 // Let's reuse drawTriangle logic manually
                 drawTriangleHighlight(gc, x, y, hoveredBlock.getRotation());
            } else if (hoveredBlock instanceof com.rustbuilder.model.structure.Wall) {
                 // Highlight logic for wall
            }

            // Draw Text
            gc.setFill(Color.BLACK);
            gc.setGlobalAlpha(0.7);
            gc.fillRect(x + 10, y + 10, 80, 20);
            gc.setGlobalAlpha(1.0);
            
            gc.setFill(Color.WHITE);
            // Limit decimals
            gc.fillText(String.format("Stab: %.0f%%", stability * 100), x + 15, y + 25);
        }
    }
    
    private void drawTriangleHighlight(GraphicsContext gc, double x, double y, double rot) {
        gc.save();
        double cx = x + TILE_SIZE / 2.0;
        double cy = y + TILE_SIZE / 2.0;
        gc.translate(cx, cy);
        gc.rotate(rot);
        gc.translate(-cx, -cy);

        triangleXPts[0] = x;
        triangleYPts[0] = y + TILE_SIZE;
        triangleXPts[1] = x + TILE_SIZE / 2.0;
        triangleYPts[1] = y + (TILE_SIZE - (TILE_SIZE * 0.866));
        triangleXPts[2] = x + TILE_SIZE;
        triangleYPts[2] = y + TILE_SIZE;

        gc.strokePolygon(triangleXPts, triangleYPts, 3);
        gc.restore();
    }

    private Color getColorForTier(BuildingTier tier) {
        if (tier == null) return Color.BLUE; // Default ghost color
        switch (tier) {
            case TWIG:
                return Color.BURLYWOOD;
            case WOOD:
                return Color.SADDLEBROWN;
            case STONE:
                return Color.GRAY;
            case METAL:
                return Color.DARKGRAY;
            case HQM:
                return Color.DARKSLATEGRAY;
            default:
                return Color.WHITE;
        }
    }
}
