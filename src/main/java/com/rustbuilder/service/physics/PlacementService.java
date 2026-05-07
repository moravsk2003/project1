package com.rustbuilder.service.physics;

import com.rustbuilder.config.GameConstants;
import com.rustbuilder.core.action.BuildAction;
import com.rustbuilder.core.placement.PlacementError;
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
import com.rustbuilder.util.BlockFactory;
import com.rustbuilder.util.BuildingTypeUtils;
import com.rustbuilder.util.SocketCompatibilityUtils;
import java.util.List;

public class PlacementService {
    private static final int SPATIAL_TARGET_SEARCH_THRESHOLD = 96;
    private static final int AIM_GRID_SIZE = 4;
    private static final int AIM_SECTOR_COUNT = AIM_GRID_SIZE * AIM_GRID_SIZE;
    private static final ThreadLocal<DummyBlocks> DUMMY_BLOCKS = ThreadLocal.withInitial(DummyBlocks::new);

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

        // Tile-local aiming offset (aimSector is a 4x4 grid).
        int aimSector = Math.max(0, Math.min(AIM_SECTOR_COUNT - 1, action.aimSector));
        int aimGrid = AIM_GRID_SIZE;
        int sectorX = aimSector % aimGrid;
        int sectorY = aimSector / aimGrid;
        // Map [0..3] to [-1.5..1.5] then scale (step size = 30% of tile)
        double stepSize = tileSize * 0.3; // 60 * 0.3 = 18px
        double center = (aimGrid - 1) / 2.0;
        double offsetX = (sectorX - center) * stepSize;
        double offsetY = (sectorY - center) * stepSize;

        // Note: Building models are centered on top-left X/Y actually
        // gridX, gridY means the center of the grid cell
        double rawCenterX = startX + action.gridX * tileSize + offsetX;
        double rawCenterY = startY + action.gridY * tileSize + offsetY;
        
        // Base X,Y of a block if placed exactly on grid
        double exactX = rawCenterX - halfTile;
        double exactY = rawCenterY - halfTile;
        
        int z = (action.actionType == BuildAction.ActionType.FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) ? 0 : action.floor;

        BuildingBlock target = null;
        double minDistSq = Double.MAX_VALUE;
        List<BuildingBlock> allBlocks = grid.getAllBlocks();
        List<BuildingBlock> candidates = allBlocks.size() > SPATIAL_TARGET_SEARCH_THRESHOLD
            ? grid.getNearbyBlocks(rawCenterX, rawCenterY, z, tileSize * 2.0)
            : allBlocks;
        for (BuildingBlock b : candidates) {
            if (b.getZ() != z && b.getZ() != z - 1) continue; // Only same floor or floor below
            
            // Measure from center of block to rawCenter
            double dx = b.getX() + halfTile - rawCenterX;
            double dy = b.getY() + halfTile - rawCenterY;
            double dSq = dx * dx + dy * dy;
            if (dSq < minDistSq || shouldPreferCeilingWallTarget(action.actionType, dSq, minDistSq, b, target)) {
                minDistSq = dSq; 
                target = b; 
            }
        }

        double firstThreshold = tileSize * 1.5;
        boolean isFirst = target == null || minDistSq > firstThreshold * firstThreshold;

        // 1. Initial / Free placement
        if (isFirst) {
            if (action.actionType == BuildAction.ActionType.FOUNDATION) {
                return new Placement(exactX, exactY, 0, Orientation.NORTH, true, PlacementError.NONE);
            }
            if (action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) {
                return new Placement(exactX, exactY, horizontalRotationDegrees(action), Orientation.NORTH, true, PlacementError.NONE);
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
            if (target != null && isValidWallTarget(target, z)) {
                Orientation orient = getWallOrientation(target, action.orientation);
                return new Placement(target.getX(), target.getY(), target.getRotation(), orient, true);
            }
            return new Placement(0,0,0,null,false, target == null ? PlacementError.BAD_SOCKET_NO_TARGET : PlacementError.BAD_SOCKET_WRONG_TARGET_TYPE); // Invalid attachment target for wall
        }

        // 5. Connecting Foundations/Floors mathematically
        if (target != null && isValidBase(target.getType(), true)) {
            // Reuse a per-thread dummy in the center of the AI's requested cell
            BuildingBlock dummy = instantiateDummy(action.actionType, exactX, exactY, z);
            if (dummy == null) return new Placement(0,0,0,null,false);
            
            // Use the selected horizontal rotation, then snap that shape to the closest socket.
            double bestShiftX = 0;
            double bestShiftY = 0;
            double bestRot = 0;
            double globalMinDistSq = Double.MAX_VALUE;
            
            double testRot = target.getRotation() + horizontalRotationDegrees(action);
            {
                 dummy.setRotation((testRot + 360) % 360);
                 
                 for (Socket tSock : target.getSockets()) {
                     if (tSock.isCenter()) continue; // Skip center socket
                     for (Socket dSock : dummy.getSockets()) {
                         if (dSock.isCenter()) continue;
                         if (!SocketCompatibilityUtils.areEdgesParallel(tSock, dSock)) continue;
                         double dx = tSock.getX() - dSock.getX();
                         double dy = tSock.getY() - dSock.getY();
                         double distSq = dx * dx + dy * dy;
                         if (distSq < globalMinDistSq) {
                             globalMinDistSq = distSq;
                             bestShiftX = tSock.getX() - dSock.getX();
                             bestShiftY = tSock.getY() - dSock.getY();
                             bestRot = dummy.getRotation();
                         }
                     }
                 }
            }
            
            double socketThreshold = tileSize * 0.5;
            if (globalMinDistSq < socketThreshold * socketThreshold) {
                 double finalSnapX = exactX + bestShiftX;
                 double finalSnapY = exactY + bestShiftY;
                 
                 // Post-snap overlap guard: center-to-center distance must be >= 0.9 * tileSize
                 double cx1 = target.getX() + halfTile;
                 double cy1 = target.getY() + halfTile;
                 double cx2 = finalSnapX + halfTile;
                 double cy2 = finalSnapY + halfTile;
                 double centerDx = cx2 - cx1;
                 double centerDy = cy2 - cy1;
                 double centerDistSq = centerDx * centerDx + centerDy * centerDy;
                 
                 double centerThreshold = tileSize * 0.9;
                 boolean allowSameTileCeilingOnWall = isCeilingAction(action.actionType)
                         && target != null
                         && BuildingTypeUtils.isWall(target.getType())
                         && z == target.getZ() + 1;
                 if (centerDistSq >= centerThreshold * centerThreshold || allowSameTileCeilingOnWall) {
                     Placement p = new Placement(finalSnapX, finalSnapY, bestRot, Orientation.NORTH, true); p.minDist = Math.sqrt(minDistSq); p.socketDist = Math.sqrt(globalMinDistSq); return p;
                 } else {
                     Placement p = new Placement(0,0,0,null,false, PlacementError.BAD_SOCKET_CENTERDIST_REJECT); p.minDist = Math.sqrt(minDistSq); p.socketDist = Math.sqrt(globalMinDistSq); return p;
                 }
            } else {
                 Placement p = new Placement(0,0,0,null,false, PlacementError.BAD_SOCKET_NO_SOCKET_ALIGNMENT); p.minDist = Math.sqrt(minDistSq); p.socketDist = Math.sqrt(globalMinDistSq); return p;
            }
        }

        Placement p = new Placement(0,0,0,null,false, target == null ? PlacementError.BAD_SOCKET_NO_TARGET : PlacementError.BAD_SOCKET_WRONG_TARGET_TYPE); p.minDist = target == null ? -1.0 : Math.sqrt(minDistSq); return p;
    }
    
    private static boolean isWallLike(BuildAction.ActionType type) {
        return type == BuildAction.ActionType.WALL || type == BuildAction.ActionType.DOORWAY || type == BuildAction.ActionType.WINDOW_FRAME;
    }

    private static boolean shouldPreferCeilingWallTarget(BuildAction.ActionType actionType, double distanceSq,
            double bestDistanceSq, BuildingBlock candidate, BuildingBlock currentTarget) {
        if (!isCeilingAction(actionType) || candidate == null || !BuildingTypeUtils.isWall(candidate.getType())) {
            return false;
        }
        if (currentTarget != null && BuildingTypeUtils.isWall(currentTarget.getType())) {
            return false;
        }
        return Math.abs(distanceSq - bestDistanceSq) < 0.001;
    }

    private static boolean isCeilingAction(BuildAction.ActionType type) {
        return type == BuildAction.ActionType.FLOOR || type == BuildAction.ActionType.TRIANGLE_FLOOR;
    }
    
    private static boolean isValidBase(BuildingType type, boolean includeWall) {
        if (type == BuildingType.FOUNDATION || type == BuildingType.TRIANGLE_FOUNDATION || type == BuildingType.FLOOR || type == BuildingType.TRIANGLE_FLOOR) return true;
        if (includeWall && (type == BuildingType.WALL || type == BuildingType.DOORWAY || type == BuildingType.WINDOW_FRAME)) return true;
        return false;
    }

    private static double horizontalRotationDegrees(BuildAction action) {
        if (action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION ||
            action.actionType == BuildAction.ActionType.TRIANGLE_FLOOR) {
            int index = action.orientation % 6;
            if (index < 0) index += 6;
            return index * 60.0;
        }

        int index = action.orientation % 4;
        if (index < 0) index += 4;
        return index * 90.0;
    }

    private static boolean isValidWallTarget(BuildingBlock target, int placementFloor) {
        if (target == null) return false;
        if (BuildingTypeUtils.isHorizontalSurface(target.getType())) {
            return target.getZ() == placementFloor;
        }
        if (BuildingTypeUtils.isWall(target.getType())) {
            return placementFloor > 0 && target.getZ() == placementFloor - 1;
        }
        return false;
    }

    private static Orientation getWallOrientation(BuildingBlock target, int orientationSelection) {
        if (target instanceof Wall && BuildingTypeUtils.isWall(target.getType())) {
            return ((Wall) target).getOrientation();
        }
        return getOrientationFromAction(target.getType(), orientationSelection);
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
        BuildingBlock dummy = DUMMY_BLOCKS.get().get(type);
        if (dummy != null) {
            dummy.setTransform(x, y, z, 0);
        }
        return dummy;
    }

    private static class DummyBlocks {
        private final Foundation foundation = new Foundation(0, 0, 0);
        private final TriangleFoundation triangleFoundation = new TriangleFoundation(0, 0, 0, 0);
        private final Floor floor = new Floor(0, 0, 0, 0);
        private final TriangleFloor triangleFloor = new TriangleFloor(0, 0, 0, 0);

        private BuildingBlock get(BuildAction.ActionType type) {
            switch (type) {
                case FOUNDATION: return foundation;
                case TRIANGLE_FOUNDATION: return triangleFoundation;
                case FLOOR: return floor;
                case TRIANGLE_FLOOR: return triangleFloor;
                default: return null;
            }
        }
    }

    public static BuildingBlock createRealBlock(BuildAction action, Placement placement) {
        return createRealBlock(action, placement, BuildingTier.STONE, DoorType.SHEET_METAL);
    }

    public static BuildingBlock createRealBlock(BuildAction action, Placement placement, BuildingTier tier, DoorType doorType) {
        double finalX = placement.x;
        double finalY = placement.y;
        double finalRotation = placement.rotation;
        Orientation finalOrientation = placement.orientation;
        int z = (action.actionType == BuildAction.ActionType.FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) ? 0 : action.floor;
        if (z < 0) z = 0;
        
        BuildingBlock block = BlockFactory.create(
                action.actionType, finalX, finalY, z,
                finalRotation, finalOrientation, doorType);

        if (block != null) {
            block.setTier(tier != null ? tier : BuildingTier.STONE);
            if (!(block instanceof Wall)) {
                block.setRotation(finalRotation);
            }
        }
        return block;
    }

    public static boolean isActionActuallyFeasible(GridModel grid, BuildAction action) {
        Placement placement = calculatePlacement(grid, action);
        if (!placement.valid) return false;

        List<BuildingBlock> allBlocks = grid.getAllBlocks();
        if (isOccupiedExactSlot(grid, action, placement, allBlocks)) {
            return false;
        }
        
        BuildingBlock block = createRealBlock(action, placement);
        if (block == null) return false;
        
        boolean isFurniture = action.actionType == BuildAction.ActionType.TC || 
                              action.actionType == BuildAction.ActionType.WORKBENCH || 
                              action.actionType == BuildAction.ActionType.LOOT_ROOM;
        if (isFurniture) {
            if (grid.canPlace(block)) {
                List<BuildingBlock> furnitureCandidates = allBlocks.size() > SPATIAL_TARGET_SEARCH_THRESHOLD
                    ? grid.getNearbyBlocks(block.getX(), block.getY(), block.getZ(), 1.0)
                    : allBlocks;
                for (BuildingBlock b2 : furnitureCandidates) {
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

    private static boolean isOccupiedExactSlot(GridModel grid, BuildAction action, Placement placement, List<BuildingBlock> allBlocks) {
        int z = (action.actionType == BuildAction.ActionType.FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) ? 0 : action.floor;
        boolean horizontal = action.actionType == BuildAction.ActionType.FOUNDATION
            || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION
            || action.actionType == BuildAction.ActionType.FLOOR
            || action.actionType == BuildAction.ActionType.TRIANGLE_FLOOR;
        boolean furniture = action.actionType == BuildAction.ActionType.TC
            || action.actionType == BuildAction.ActionType.WORKBENCH
            || action.actionType == BuildAction.ActionType.LOOT_ROOM;

        if (!horizontal && !furniture) {
            return false;
        }

        List<BuildingBlock> candidates = allBlocks.size() > SPATIAL_TARGET_SEARCH_THRESHOLD
            ? grid.getNearbyBlocks(placement.x, placement.y, z, 1.0)
            : allBlocks;
        for (BuildingBlock existing : candidates) {
            if (existing.getZ() != z) {
                continue;
            }
            if (Math.abs(existing.getX() - placement.x) >= 1.0 || Math.abs(existing.getY() - placement.y) >= 1.0) {
                continue;
            }
            if (horizontal && BuildingTypeUtils.isHorizontalSurface(existing.getType())) {
                return true;
            }
            if (furniture && BuildingTypeUtils.isFurniture(existing.getType())) {
                return true;
            }
        }

        return false;
    }
}
