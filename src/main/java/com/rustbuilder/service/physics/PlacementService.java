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
import com.rustbuilder.util.BlockFactory;
import com.rustbuilder.util.BuildingTypeUtils;
import java.util.List;

public class PlacementService {
    private static final int SPATIAL_TARGET_SEARCH_THRESHOLD = 96;
    private static final int AIM_GRID_SIZE = 4;
    private static final int AIM_SECTOR_COUNT = AIM_GRID_SIZE * AIM_GRID_SIZE;

    public static PlacementResult calculatePlacement(GridModel grid, BuildAction action) {
        double tileSize = GameConstants.TILE_SIZE;
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

        int z = (action.actionType == BuildAction.ActionType.FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) ? 0 : action.floor;

        SocketPlacementResolver.Result resolved = SocketPlacementResolver.resolve(
                grid, rawCenterX, rawCenterY, toolIdForAction(action.actionType), z, false,
                (block, socket) -> socketPriorityForAction(action, block, socket));

        if (!resolved.valid) {
            return new PlacementResult.Invalid(
                    placementErrorFor(action, resolved),
                    centerDistance(resolved.block, rawCenterX, rawCenterY),
                    resolved.socketDistanceSq >= 0 ? Math.sqrt(resolved.socketDistanceSq) : -1.0);
        }

        double rotation = resolved.rotation;

        if (resolved.block == null && action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) {
            rotation = horizontalRotationDegrees(action);
        }

        return new PlacementResult.Valid(
                resolved.x,
                resolved.y,
                rotation,
                resolved.orientation,
                centerDistance(resolved.block, rawCenterX, rawCenterY),
                resolved.socketDistanceSq >= 0 ? Math.sqrt(resolved.socketDistanceSq) : -1.0);
    }

    private static String toolIdForAction(BuildAction.ActionType type) {
        if (type == BuildAction.ActionType.TRIANGLE_FOUNDATION) {
            return "TRIANGLE";
        }
        return type == null ? null : type.name();
    }

    private static PlacementError placementErrorFor(BuildAction action, SocketPlacementResolver.Result resolved) {
        if (resolved == null || resolved.block == null) {
            if (action.actionType == BuildAction.ActionType.FOUNDATION ||
                    action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) {
                return PlacementError.BAD_SOCKET_NO_TARGET;
            }
            return PlacementError.BAD_SOCKET_IS_FIRST;
        }
        return PlacementError.BAD_SOCKET_WRONG_TARGET_TYPE;
    }

    private static double centerDistance(BuildingBlock block, double rawCenterX, double rawCenterY) {
        if (block == null) {
            return -1.0;
        }
        double dx = block.getX() + GameConstants.HALF_TILE - rawCenterX;
        double dy = block.getY() + GameConstants.HALF_TILE - rawCenterY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    private static int socketPriorityForAction(BuildAction action, BuildingBlock block, Socket socket) {
        if (action == null || block == null || socket == null || !isWallAction(action.actionType)) {
            return 0;
        }
        int desiredSide = desiredWallSocketSide(action, block);
        return socket.getSide() == desiredSide ? 1 : 0;
    }

    private static boolean isWallAction(BuildAction.ActionType type) {
        return type == BuildAction.ActionType.WALL
                || type == BuildAction.ActionType.DOORWAY
                || type == BuildAction.ActionType.WINDOW_FRAME;
    }

    private static int desiredWallSocketSide(BuildAction action, BuildingBlock target) {
        if (BuildingTypeUtils.isWall(target.getType())) {
            return Socket.CENTER_SIDE;
        }
        if (target.getType() == BuildingType.TRIANGLE_FOUNDATION ||
                target.getType() == BuildingType.TRIANGLE_FLOOR) {
            int index = action.orientation % 3;
            if (index < 0) {
                index += 3;
            }
            switch (index) {
                case 1:
                    return 4;
                case 2:
                    return 5;
                case 0:
                default:
                    return 6;
            }
        }
        int side = action.orientation % 4;
        return side < 0 ? side + 4 : side;
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

    public static BuildingBlock createRealBlock(BuildAction action, PlacementResult.Valid placement) {
        return createRealBlock(action, placement, BuildingTier.STONE, DoorType.SHEET_METAL);
    }

    public static BuildingBlock createRealBlock(BuildAction action, PlacementResult.Valid placement, BuildingTier tier, DoorType doorType) {
        double finalX = placement.x();
        double finalY = placement.y();
        double finalRotation = placement.rotation();
        Orientation finalOrientation = placement.orientation();
        int z = (action.actionType == BuildAction.ActionType.FOUNDATION || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION) ? 0 : action.floor;
        if (z < 0) z = 0;
        
        BuildingBlock block = BlockFactory.create(
                action.actionType, finalX, finalY, z,
                finalRotation, finalOrientation, doorType);

        if (block != null) {
            block.setTier(tier != null ? tier : BuildingTier.STONE);
            block.setRotation(finalRotation);
        }
        return block;
    }

    public static boolean isActionActuallyFeasible(GridModel grid, BuildAction action) {
        PlacementResult placement = calculatePlacement(grid, action);
        if (!(placement instanceof PlacementResult.Valid validPlacement)) return false;

        List<BuildingBlock> allBlocks = grid.getAllBlocks();
        if (isOccupiedExactSlot(grid, action, validPlacement, allBlocks)) {
            return false;
        }
        
        BuildingBlock block = createRealBlock(action, validPlacement);
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

    private static boolean isOccupiedExactSlot(GridModel grid, BuildAction action, PlacementResult.Valid placement, List<BuildingBlock> allBlocks) {
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
            ? grid.getNearbyBlocks(placement.x(), placement.y(), z, 1.0)
            : allBlocks;
        for (BuildingBlock existing : candidates) {
            if (existing.getZ() != z) {
                continue;
            }
            if (Math.abs(existing.getX() - placement.x()) >= 1.0 || Math.abs(existing.getY() - placement.y()) >= 1.0) {
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
