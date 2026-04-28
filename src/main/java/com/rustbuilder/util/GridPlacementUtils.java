package com.rustbuilder.util;

import com.rustbuilder.ai.ea.BaseGenome.BuildAction;
import com.rustbuilder.ai.rl.PlacementError;
import com.rustbuilder.config.GameConstants;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.core.Socket;
import com.rustbuilder.model.structure.Floor;
import com.rustbuilder.model.structure.Foundation;
import com.rustbuilder.model.structure.TriangleFloor;
import com.rustbuilder.model.structure.TriangleFoundation;
import com.rustbuilder.model.structure.Wall;

public class GridPlacementUtils {

    public static class Placement {
        public double x, y, rotation;
        public Orientation orientation;
        public boolean valid;
        public PlacementError error = PlacementError.NONE;
        public double minDist = -1.0;
        public double socketDist = -1.0;
        
        public Placement(double x, double y, double rotation, Orientation orientation, boolean valid, PlacementError error) {
            this.x = x;
            this.y = y;
            this.rotation = rotation;
            this.orientation = orientation;
            this.valid = valid;
            this.error = error;
        }

        public Placement(double x, double y, double rotation, Orientation orientation, boolean valid) {
            this(x, y, rotation, orientation, valid, PlacementError.NONE);
        }
    }

    public static Placement calculatePlacement(GridModel grid, BuildAction action) {
        double tileSize = GameConstants.TILE_SIZE;
        double halfTile = GameConstants.HALF_TILE;
        double startX = GameConstants.GRID_ORIGIN_X;
        double startY = GameConstants.GRID_ORIGIN_Y;

        // Tile-local aiming offset (aimSector is a 5x5 grid, center=12)
        int sectorX = action.aimSector % 5;
        int sectorY = action.aimSector / 5;
        // Map [0..4] to [-2..2] then scale (step size = 30% of tile)
        double stepSize = tileSize * 0.3; // 60 * 0.3 = 18px
        double offsetX = (sectorX - 2) * stepSize;
        double offsetY = (sectorY - 2) * stepSize;

        // Note: Building models are centered on top-left X/Y actually
        // gridX, gridY means the center of the grid cell
        double rawCenterX = startX + action.gridX * tileSize + offsetX;
        double rawCenterY = startY + action.gridY * tileSize + offsetY;
        
        // Base X,Y of a block if placed exactly on grid
        double exactX = rawCenterX - halfTile;
        double exactY = rawCenterY - halfTile;
        
        int z = (action.actionType == BuildAction.ActionType.FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) ? 0 : action.floor;

        BuildingBlock target = null;
        double minDist = Double.MAX_VALUE;
        // Find closest block
        for (BuildingBlock b : grid.getAllBlocks()) {
            if (b.getZ() != z && b.getZ() != z - 1) continue; // Only same floor or floor below
            
            // Measure from center of block to rawCenter
            double d = Math.hypot(b.getX() + halfTile - rawCenterX, b.getY() + halfTile - rawCenterY);
            if (d < minDist) { 
                minDist = d; 
                target = b; 
            }
        }

        boolean isFirst = target == null || minDist > tileSize * 1.5;

        // 1. Initial / Free placement
        if (isFirst) {
            if (action.actionType == BuildAction.ActionType.FOUNDATION) {
                return new Placement(exactX, exactY, 0, Orientation.NORTH, true, PlacementError.NONE);
            }
            if (action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) {
                return new Placement(exactX, exactY, 0, Orientation.NORTH, true, PlacementError.NONE);
            }
            return new Placement(0, 0, 0, Orientation.NORTH, false, PlacementError.BAD_SOCKET_IS_FIRST);
        }

        // 2. Decor / Deployables (TC, Workbench, Loot) -> Center of Target
        if (action.actionType == BuildAction.ActionType.TC || action.actionType == BuildAction.ActionType.WORKBENCH || action.actionType == BuildAction.ActionType.LOOT_ROOM) {
            if (target != null) {
                return new Placement(target.getX(), target.getY(), target.getRotation(), Orientation.NORTH, true, PlacementError.NONE);
            }
            return new Placement(0, 0, 0, Orientation.NORTH, false, target == null ? PlacementError.BAD_SOCKET_NO_TARGET : PlacementError.BAD_SOCKET_WRONG_TARGET_TYPE);
        }

        // 4. Walls / Doorways / Windows
        if (isWallLike(action.actionType)) {
            if (target != null && isValidBase(target.getType(), true)) {
                Orientation orient = getOrientationFromAction(target.getType(), action.orientation);
                return new Placement(target.getX(), target.getY(), target.getRotation(), orient, true);
            }
            return new Placement(0,0,0,null,false, target == null ? PlacementError.BAD_SOCKET_NO_TARGET : PlacementError.BAD_SOCKET_WRONG_TARGET_TYPE); // Invalid attachment target for wall
        }

        // 5. Connecting Foundations/Floors mathematically
        if (target != null && isValidBase(target.getType(), true)) {
            // Instantiate a dummy in the center of the AI's requested cell
            BuildingBlock dummy = instantiateDummy(action.actionType, exactX, exactY, z);
            if (dummy == null) return new Placement(0,0,0,null,false);
            
            // Try 4 orientations to find the one that best connects sockets mathematically
            double bestShiftX = 0;
            double bestShiftY = 0;
            double bestRot = 0;
            double globalMinDist = Double.MAX_VALUE;
            
            int rotations = (action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FLOOR) ? 6 : 4;
            for (int rotGuess = 0; rotGuess < rotations; rotGuess++) {
                 double testRot = target.getRotation() + rotGuess * (rotations == 6 ? 60 : 90);
                 dummy.setRotation((testRot + 360) % 360);
                 
                 for (Socket tSock : target.getSockets()) {
                     if (tSock.getSide() == 10) continue; // Skip center socket
                     for (Socket dSock : dummy.getSockets()) {
                         if (dSock.getSide() == 10) continue;
                         double dist = Math.hypot(tSock.getX() - dSock.getX(), tSock.getY() - dSock.getY());
                         if (dist < globalMinDist) {
                             globalMinDist = dist;
                             bestShiftX = tSock.getX() - dSock.getX();
                             bestShiftY = tSock.getY() - dSock.getY();
                             bestRot = dummy.getRotation();
                         }
                     }
                 }
            }
            
            if (globalMinDist < tileSize * 0.5) {
                 double finalSnapX = exactX + bestShiftX;
                 double finalSnapY = exactY + bestShiftY;
                 
                 // Post-snap overlap guard: center-to-center distance must be >= 0.9 * tileSize
                 double cx1 = target.getX() + halfTile;
                 double cy1 = target.getY() + halfTile;
                 double cx2 = finalSnapX + halfTile;
                 double cy2 = finalSnapY + halfTile;
                 double centerDist = Math.hypot(cx2 - cx1, cy2 - cy1);
                 
                 if (centerDist >= tileSize * 0.9) {
                     Placement p = new Placement(finalSnapX, finalSnapY, bestRot, Orientation.NORTH, true); p.minDist = minDist; p.socketDist = globalMinDist; return p;
                 } else {
                     Placement p = new Placement(0,0,0,null,false, PlacementError.BAD_SOCKET_CENTERDIST_REJECT); p.minDist = minDist; p.socketDist = globalMinDist; return p;
                 }
            } else {
                 Placement p = new Placement(0,0,0,null,false, PlacementError.BAD_SOCKET_NO_SOCKET_ALIGNMENT); p.minDist = minDist; p.socketDist = globalMinDist; return p;
            }
        }

        Placement p = new Placement(0,0,0,null,false, target == null ? PlacementError.BAD_SOCKET_NO_TARGET : PlacementError.BAD_SOCKET_WRONG_TARGET_TYPE); p.minDist = minDist; return p;
    }
    
    private static boolean isWallLike(BuildAction.ActionType type) {
        return type == BuildAction.ActionType.WALL || type == BuildAction.ActionType.DOORWAY || type == BuildAction.ActionType.WINDOW_FRAME;
    }
    
    private static boolean isValidBase(BuildingType type, boolean includeWall) {
        if (type == BuildingType.FOUNDATION || type == BuildingType.TRIANGLE_FOUNDATION || type == BuildingType.FLOOR || type == BuildingType.TRIANGLE_FLOOR) return true;
        if (includeWall && (type == BuildingType.WALL || type == BuildingType.DOORWAY || type == BuildingType.WINDOW_FRAME)) return true;
        return false;
    }

    private static Orientation getOrientationFromAction(BuildingType baseType, int orientationSelection) {
        if (baseType == BuildingType.TRIANGLE_FOUNDATION || baseType == BuildingType.TRIANGLE_FLOOR) {
            Orientation[] triOrients = {Orientation.TRIANGLE_BASE, Orientation.TRIANGLE_LEFT, Orientation.TRIANGLE_RIGHT};
            return triOrients[orientationSelection % 3];
        } else {
            Orientation[] sqOrients = {Orientation.NORTH, Orientation.EAST, Orientation.SOUTH, Orientation.WEST};
            return sqOrients[orientationSelection % 4];
        }
    }
    
    private static BuildingBlock instantiateDummy(BuildAction.ActionType type, double x, double y, int z) {
        switch (type) {
             case FOUNDATION: return new Foundation(x, y, z);
             case TRIANGLE_FOUNDATION: return new TriangleFoundation(x, y, z, 0);
             case FLOOR: return new Floor(x, y, z, 0);
             case TRIANGLE_FLOOR: return new TriangleFloor(x, y, z, 0);
             default: return null;
        }
    }

    public static BuildingBlock createRealBlock(BuildAction action, Placement placement) {
        double finalX = placement.x;
        double finalY = placement.y;
        double finalRotation = placement.rotation;
        Orientation finalOrientation = placement.orientation;
        int z = (action.actionType == BuildAction.ActionType.FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) ? 0 : action.floor;
        if (z < 0) z = 0;
        
        // Delegate construction to BlockFactory (SHEET_METAL default for doorways in this context)
        BuildingBlock block = BlockFactory.create(
                action.actionType, finalX, finalY, z,
                finalRotation, finalOrientation, DoorType.SHEET_METAL);

        if (block != null) {
            block.setTier(BuildingTier.STONE);
            if (!(block instanceof Wall)) {
                block.setRotation(finalRotation);
            }
        }
        return block;
    }

    public static boolean isActionActuallyFeasible(GridModel grid, BuildAction action) {
        Placement placement = calculatePlacement(grid, action);
        if (!placement.valid) return false;
        
        BuildingBlock block = createRealBlock(action, placement);
        if (block == null) return false;
        
        boolean isFurniture = action.actionType == BuildAction.ActionType.TC || 
                              action.actionType == BuildAction.ActionType.WORKBENCH || 
                              action.actionType == BuildAction.ActionType.LOOT_ROOM;
        if (isFurniture) {
            if (grid.canPlace(block)) {
                for (BuildingBlock b2 : grid.getAllBlocks()) {
                    if (com.rustbuilder.util.BuildingTypeUtils.isFurniture(b2.getType()) &&
                        b2.getZ() == block.getZ() &&
                        Math.abs(b2.getX() - block.getX()) < 1.0 &&
                        Math.abs(b2.getY() - block.getY()) < 1.0) {
                        return false;
                    }
                }
                return true;
            }
            return false;
        } else {
            return grid.canPlace(block);
        }
    }
}
