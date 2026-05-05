package com.rustbuilder.service.physics;

import java.util.List;

import com.rustbuilder.config.GameConstants;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.core.Socket;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.util.SocketGeometryUtils;

public class SnappingService {

    private final GridModel gridModel;

    public SnappingService(GridModel gridModel) {
        this.gridModel = gridModel;
    }

    public static class SnapResult {
        public double x;
        public double y;
        public double rotation;
        public Orientation orientation;
        public boolean valid;

        public SnapResult(double x, double y, double rotation, Orientation orientation, boolean valid) {
            this.x = x;
            this.y = y;
            this.rotation = rotation;
            this.orientation = orientation;
            this.valid = valid;
        }
    }

    public SnapResult calculateSnap(double mouseX, double mouseY, String selectedTool, int currentFloor) {
        double ghostX = mouseX - GameConstants.HALF_TILE;
        double ghostY = mouseY - GameConstants.HALF_TILE;
        double ghostRotation = 0;
        Orientation ghostOrientation = Orientation.NORTH;
        boolean isUnsnappedValid = "FOUNDATION".equals(selectedTool) || "TRIANGLE".equals(selectedTool);
        boolean currentValid = isUnsnappedValid;

        Socket closestSocket = null;
        BuildingBlock closestBlock = null;
        double minDistSq = GameConstants.SNAP_RADIUS * GameConstants.SNAP_RADIUS;

        double searchRadius = GameConstants.TILE_SIZE * 2.5;
        List<BuildingBlock> localBlocks = gridModel.getNearbyBlocks(mouseX, mouseY, currentFloor, searchRadius);
        for (BuildingBlock block : localBlocks) {
            if (block.getZ() > currentFloor || block.getZ() < currentFloor - 1)
                continue;

            if (block.getZ() == currentFloor - 1 && !isWallType(block.getType())) {
                continue;
            }

            for (Socket socket : block.getSockets()) {
                if (!isSocketEligibleForTool(block, socket, selectedTool)) {
                    continue;
                }

                double dx = socket.getX() - mouseX;
                double dy = socket.getY() - mouseY;
                double distSq = dx * dx + dy * dy;
                if (distSq < minDistSq) {
                    minDistSq = distSq;
                    closestSocket = socket;
                    closestBlock = block;
                } else if (Math.abs(distSq - minDistSq) < 0.001) {
                    if (closestBlock != null && isWallType(block.getType()) && !isWallType(closestBlock.getType())) {
                        minDistSq = distSq;
                        closestSocket = socket;
                        closestBlock = block;
                    }
                }
            }
        }
        
        if (closestSocket != null && closestBlock != null) {
            currentValid = true;
            double sx = closestSocket.getX();
            double sy = closestSocket.getY();
            int side = closestSocket.getSide();
            double baseRotation = closestBlock.getRotation();

            boolean isGhostSquare = "FOUNDATION".equals(selectedTool) || "FLOOR".equals(selectedTool);
            boolean isGhostTriangle = "TRIANGLE".equals(selectedTool) || "TRIANGLE_FLOOR".equals(selectedTool);

            if (side == 10) {
                ghostX = closestBlock.getX();
                ghostY = closestBlock.getY();
                
                if (closestBlock instanceof Wall) {
                    ghostRotation = closestBlock.getRotation();
                    ghostOrientation = ((Wall) closestBlock).getOrientation();
                    
                    if (isGhostSquare) {
                        // For floors on walls, align with wall's orientation
                        double wallNormal = 0;
                        switch (ghostOrientation) {
                            case NORTH: wallNormal = -90; break;
                            case EAST: wallNormal = 0; break;
                            case SOUTH: wallNormal = 90; break;
                            case WEST: wallNormal = 180; break;
                            default: break;
                        }
                        ghostRotation = (wallNormal + 270 + 360) % 360;
                    }
                } else {
                    ghostRotation = closestBlock.getRotation();
                }
            } else if (isGhostSquare || isGhostTriangle) {
                // Determine normal vector of the socket
                double normalAngle = 0;
                if (side == 0) normalAngle = baseRotation - 90;
                else if (side == 1) normalAngle = baseRotation;
                else if (side == 2) normalAngle = baseRotation + 90;
                else if (side == 3) normalAngle = baseRotation + 180;
                else if (side == 4) normalAngle = baseRotation + 210;
                else if (side == 5) normalAngle = baseRotation + 330;
                else if (side == 6) normalAngle = baseRotation + 90;
                
                // Align the ghost block so its connecting socket directly opposes the normal
                if (isGhostSquare) {
                    // Align the ghost block so its Side 0 (initially North) faces the center of the target
                    ghostRotation = (normalAngle + 270 + 360) % 360; 
                    List<Socket> mockSockets = SocketGeometryUtils.getSquareSockets(0, 0, ghostRotation);
                    Socket matchSocket = mockSockets.get(0); // Use Side 0 for consistent snapping
                    ghostX = sx - matchSocket.getX();
                    ghostY = sy - matchSocket.getY();
                } else if (isGhostTriangle) {
                    // Triangle Side 6 (Base) faces its rotation + 90
                    // We want its Base to face opposite normal: R + 90 = normalAngle + 180 => R = normalAngle + 90
                    ghostRotation = normalAngle + 90; 
                    List<Socket> mockSockets = SocketGeometryUtils.getTriangleSockets(0, 0, ghostRotation);
                    Socket matchSocket = mockSockets.stream().filter(s -> s.getSide() == 6).findFirst().orElse(null);
                    if (matchSocket != null) {
                        ghostX = sx - matchSocket.getX();
                        ghostY = sy - matchSocket.getY();
                    }
                }
            }
            
            if (isWallType(selectedTool)) {
                if (closestBlock.getType() == BuildingType.FOUNDATION ||
                        closestBlock.getType() == BuildingType.FLOOR) {
                    ghostX = closestBlock.getX();
                    ghostY = closestBlock.getY();
                    ghostRotation = closestBlock.getRotation();
                    ghostOrientation = getOrientationFromSide(closestSocket.getSide());

                } else if (closestBlock.getType() == BuildingType.TRIANGLE_FOUNDATION ||
                           closestBlock.getType() == BuildingType.TRIANGLE_FLOOR) {
                    ghostX = closestBlock.getX();
                    ghostY = closestBlock.getY();
                    ghostRotation = closestBlock.getRotation();
                    ghostOrientation = getOrientationFromSide(closestSocket.getSide());
                }
            } else if ("TC".equals(selectedTool) || "WORKBENCH".equals(selectedTool) || "LOOT_ROOM".equals(selectedTool)) {
                 if (closestBlock.getType() == BuildingType.FOUNDATION ||
                        closestBlock.getType() == BuildingType.TRIANGLE_FOUNDATION ||
                        closestBlock.getType() == BuildingType.FLOOR ||
                        closestBlock.getType() == BuildingType.TRIANGLE_FLOOR) {
                     ghostX = closestBlock.getX();
                     ghostY = closestBlock.getY();
                     ghostRotation = closestBlock.getRotation();
                     currentValid = true;
                } else {
                    currentValid = false;
                }
            } else if ("DOOR".equals(selectedTool)) {
                if (closestBlock.getType() == BuildingType.DOORWAY && closestBlock instanceof Wall) {
                    Wall doorway = (Wall) closestBlock;
                    ghostX = doorway.getX();
                    ghostY = doorway.getY();
                    ghostRotation = doorway.getRotation();
                    ghostOrientation = doorway.getOrientation();
                    
                    boolean doorExists = false;
                    for (BuildingBlock b : localBlocks) {
                        if (b.getType() == BuildingType.DOOR && b.getZ() == closestBlock.getZ()
                            && Math.abs(b.getX() - ghostX) < 0.1 && Math.abs(b.getY() - ghostY) < 0.1) {
                            doorExists = true;
                            break;
                        }
                    }
                    currentValid = !doorExists;
                } else {
                    currentValid = false;
                }
            }
        }

        if (currentValid && ("FLOOR".equals(selectedTool) || "TRIANGLE_FLOOR".equals(selectedTool))) {
            boolean hasWall = false;
            for (BuildingBlock b : gridModel.getAllBlocks()) {
                if (isWallType(b.getType())) {
                    hasWall = true;
                    break;
                }
            }
            if (!hasWall) currentValid = false;
        }

        return new SnapResult(ghostX, ghostY, ghostRotation, ghostOrientation, currentValid);
    }

    private boolean isWallType(String tool) {
        return com.rustbuilder.util.BuildingTypeUtils.isWallTool(tool);
    }

    private boolean isWallType(BuildingType type) {
        return com.rustbuilder.util.BuildingTypeUtils.isWall(type);
    }

    private boolean isHorizontalSurface(BuildingType type) {
        return com.rustbuilder.util.BuildingTypeUtils.isHorizontalSurface(type);
    }

    private boolean isSocketEligibleForTool(BuildingBlock block, Socket socket, String selectedTool) {
        BuildingType blockType = block.getType();
        int side = socket.getSide();

        if ("DOOR".equals(selectedTool)) {
            return blockType == BuildingType.DOORWAY && side == 10;
        }

        if ("TC".equals(selectedTool) || "WORKBENCH".equals(selectedTool) || "LOOT_ROOM".equals(selectedTool)) {
            return isHorizontalSurface(blockType) && side == 10;
        }

        if (isWallType(selectedTool)) {
            if (isWallType(blockType)) {
                return side == 10;
            }
            if (isHorizontalSurface(blockType)) {
                return side != 10;
            }
            return false;
        }

        if ("FOUNDATION".equals(selectedTool) || "TRIANGLE".equals(selectedTool)) {
            return isHorizontalSurface(blockType) && side != 10;
        }

        if ("FLOOR".equals(selectedTool) || "TRIANGLE_FLOOR".equals(selectedTool)) {
            if (isWallType(blockType)) {
                return true;
            }
            if (isHorizontalSurface(blockType)) {
                return side != 10;
            }
            return false;
        }

        return side != 10;
    }

    private Orientation getOrientationFromSide(int side) {
        switch (side) {
            case 0: return Orientation.NORTH;
            case 1: return Orientation.EAST;
            case 2: return Orientation.SOUTH;
            case 3: return Orientation.WEST;
            case 4: return Orientation.TRIANGLE_LEFT;
            case 5: return Orientation.TRIANGLE_RIGHT;
            case 6: return Orientation.TRIANGLE_BASE;
            default: return Orientation.NORTH;
        }
    }
    
}
