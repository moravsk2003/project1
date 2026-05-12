package com.rustbuilder.service.physics;

import com.rustbuilder.config.GameConstants;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingType;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.core.Socket;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.util.BlockFactory;
import com.rustbuilder.util.BuildingTypeUtils;
import com.rustbuilder.util.SocketGeometryUtils;
import java.util.ArrayList;
import java.util.List;

/**
 * Shared socket-to-placement resolver for both manual UI snapping and generated
 * AI/EA placement. The caller provides an aim point; this class chooses the
 * nearest eligible socket and converts it into final block coordinates.
 */
final class SocketPlacementResolver {

    interface SocketPreference {
        int priority(BuildingBlock block, Socket socket);
    }

    private static final SocketPreference NO_PREFERENCE = (block, socket) -> 0;

    static final class Result {
        final double x;
        final double y;
        final double rotation;
        final Orientation orientation;
        final boolean valid;
        final Socket socket;
        final BuildingBlock block;
        final double socketDistanceSq;

        Result(double x, double y, double rotation, Orientation orientation, boolean valid,
                Socket socket, BuildingBlock block, double socketDistanceSq) {
            this.x = x;
            this.y = y;
            this.rotation = rotation;
            this.orientation = orientation;
            this.valid = valid;
            this.socket = socket;
            this.block = block;
            this.socketDistanceSq = socketDistanceSq;
        }
    }

    private static final class Candidate {
        final Socket socket;
        final BuildingBlock block;
        final double distSq;

        Candidate(Socket socket, BuildingBlock block, double distSq) {
            this.socket = socket;
            this.block = block;
            this.distSq = distSq;
        }
    }

    private SocketPlacementResolver() {
    }

    static Result resolve(GridModel gridModel, double aimX, double aimY, String selectedTool, int currentFloor,
            boolean preferPlaceable) {
        return resolve(gridModel, aimX, aimY, selectedTool, currentFloor, preferPlaceable, NO_PREFERENCE);
    }

    static Result resolve(GridModel gridModel, double aimX, double aimY, String selectedTool, int currentFloor,
            boolean preferPlaceable, SocketPreference socketPreference) {
        double ghostX = aimX - GameConstants.HALF_TILE;
        double ghostY = aimY - GameConstants.HALF_TILE;
        double ghostRotation = 0;
        Orientation ghostOrientation = Orientation.NORTH;
        boolean currentValid = isFreeFoundationTool(selectedTool);
        SocketPreference preference = socketPreference != null ? socketPreference : NO_PREFERENCE;

        double snapRadiusSq = GameConstants.SNAP_RADIUS * GameConstants.SNAP_RADIUS;
        ArrayList<Candidate> candidates = new ArrayList<>();

        double searchRadius = GameConstants.TILE_SIZE * 2.5;
        List<BuildingBlock> localBlocks = gridModel.getNearbyBlocks(aimX, aimY, currentFloor, searchRadius);
        for (BuildingBlock block : localBlocks) {
            if (block.getZ() > currentFloor || block.getZ() < currentFloor - 1) {
                continue;
            }

            if (block.getZ() == currentFloor - 1 && !isWallType(block.getType())) {
                continue;
            }

            for (Socket socket : block.getSockets()) {
                if (!isSocketEligibleForTool(block, socket, selectedTool, currentFloor)) {
                    continue;
                }

                double dx = socket.getX() - aimX;
                double dy = socket.getY() - aimY;
                double distSq = dx * dx + dy * dy;
                if (distSq < snapRadiusSq) {
                    candidates.add(new Candidate(socket, block, distSq));
                }
            }
        }

        candidates.sort((a, b) -> {
            int distCompare = Double.compare(a.distSq, b.distSq);
            if (distCompare != 0) {
                return distCompare;
            }
            int preferenceCompare = Integer.compare(
                    preference.priority(a.block, a.socket),
                    preference.priority(b.block, b.socket));
            if (preferenceCompare != 0) {
                return preferenceCompare;
            }
            if (isWallType(a.block.getType()) && !isWallType(b.block.getType())) {
                return -1;
            }
            if (!isWallType(a.block.getType()) && isWallType(b.block.getType())) {
                return 1;
            }
            return 0;
        });

        Result closestResult = null;
        for (Candidate candidate : candidates) {
            Result result = resolveCandidate(
                    gridModel, aimX, aimY, selectedTool, currentFloor, localBlocks, candidate.socket, candidate.block,
                    candidate.distSq);
            if (closestResult == null) {
                closestResult = result;
            }
            if (preferPlaceable) {
                if (isPlaceableSnap(gridModel, result, selectedTool, currentFloor)) {
                    return result;
                }
            } else if (result.valid) {
                return result;
            }
        }

        if (closestResult != null) {
            return closestResult;
        }

        return new Result(ghostX, ghostY, ghostRotation, ghostOrientation, currentValid, null, null, -1.0);
    }

    private static Result resolveCandidate(GridModel gridModel, double aimX, double aimY, String selectedTool, int currentFloor,
            List<BuildingBlock> localBlocks, Socket closestSocket, BuildingBlock closestBlock, double socketDistanceSq) {
        boolean currentValid = true;
        double ghostX = aimX - GameConstants.HALF_TILE;
        double ghostY = aimY - GameConstants.HALF_TILE;
        double ghostRotation = 0;
        Orientation ghostOrientation = Orientation.NORTH;

        double sx = closestSocket.getX();
        double sy = closestSocket.getY();
        int side = closestSocket.getSide();
        double baseRotation = closestBlock.getRotation();

        boolean isGhostSquare = "FOUNDATION".equals(selectedTool) || "FLOOR".equals(selectedTool);
        boolean isGhostTriangle = isTriangleFoundationTool(selectedTool) || "TRIANGLE_FLOOR".equals(selectedTool);

        if (side == Socket.CENTER_SIDE) {
            ghostX = closestBlock.getX();
            ghostY = closestBlock.getY();

            if (closestBlock instanceof Wall) {
                ghostRotation = closestBlock.getRotation();
                ghostOrientation = ((Wall) closestBlock).getOrientation();

                if (isGhostSquare) {
                    double wallNormal = 0;
                    switch (ghostOrientation) {
                        case NORTH:
                            wallNormal = -90;
                            break;
                        case EAST:
                            wallNormal = 0;
                            break;
                        case SOUTH:
                            wallNormal = 90;
                            break;
                        case WEST:
                            wallNormal = 180;
                            break;
                        default:
                            break;
                    }
                    ghostRotation = (wallNormal + 270 + 360) % 360;
                }
            } else {
                ghostRotation = closestBlock.getRotation();
            }
        } else if (isGhostSquare || isGhostTriangle) {
            double normalAngle = 0;
            if (side == 0) {
                normalAngle = baseRotation - 90;
            } else if (side == 1) {
                normalAngle = baseRotation;
            } else if (side == 2) {
                normalAngle = baseRotation + 90;
            } else if (side == 3) {
                normalAngle = baseRotation + 180;
            } else if (side == 4) {
                normalAngle = baseRotation + 210;
            } else if (side == 5) {
                normalAngle = baseRotation + 330;
            } else if (side == 6) {
                normalAngle = baseRotation + 90;
            }

            if (isGhostSquare) {
                ghostRotation = (normalAngle + 270 + 360) % 360;
                List<Socket> mockSockets = SocketGeometryUtils.getSquareSockets(0, 0, ghostRotation);
                int matchSide = 0;
                if (isWallType(closestBlock.getType())) {
                    double normalRad = Math.toRadians(normalAngle);
                    double normalX = Math.cos(normalRad);
                    double normalY = Math.sin(normalRad);
                    double aimSide = (aimX - sx) * normalX + (aimY - sy) * normalY;
                    matchSide = aimSide < 0 ? 2 : 0;
                }
                Socket matchSocket = findSocketBySide(mockSockets, matchSide);
                ghostX = sx - matchSocket.getX();
                ghostY = sy - matchSocket.getY();
            } else if (isGhostTriangle) {
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

        if (currentValid && ("FLOOR".equals(selectedTool) || "TRIANGLE_FLOOR".equals(selectedTool))) {
            boolean hasWall = false;
            for (BuildingBlock b : gridModel.getAllBlocks()) {
                if (isWallType(b.getType())) {
                    hasWall = true;
                    break;
                }
            }
            if (!hasWall) {
                currentValid = false;
            }
        }

        return new Result(ghostX, ghostY, ghostRotation, ghostOrientation, currentValid,
                closestSocket, closestBlock, socketDistanceSq);
    }

    private static Socket findSocketBySide(List<Socket> sockets, int side) {
        for (Socket socket : sockets) {
            if (socket.getSide() == side) {
                return socket;
            }
        }
        return sockets.get(0);
    }

    private static boolean isPlaceableSnap(GridModel gridModel, Result result, String selectedTool, int currentFloor) {
        if (result == null || !result.valid) {
            return false;
        }

        BuildingBlock block = createBlockForValidation(
                result.x, result.y, currentFloor, result.rotation, result.orientation, selectedTool);
        return block != null && gridModel.canPlace(block);
    }

    private static BuildingBlock createBlockForValidation(double x, double y, int z, double rotation,
            Orientation orientation, String tool) {
        BuildingType type = BuildingTypeUtils.fromToolId(tool);
        if (z > 0 && BuildingTypeUtils.isFoundation(type)) {
            return null;
        }

        BuildingBlock block = BlockFactory.create(type, x, y, z, rotation, orientation, DoorType.SHEET_METAL);

        if (block != null && (isWallType(tool) || "DOOR".equals(tool))) {
            block.setRotation(rotation);
        }
        return block;
    }

    private static boolean isSocketEligibleForTool(BuildingBlock block, Socket socket, String selectedTool, int currentFloor) {
        BuildingType blockType = block.getType();
        int side = socket.getSide();

        if ("DOOR".equals(selectedTool)) {
            return blockType == BuildingType.DOORWAY && side == Socket.CENTER_SIDE;
        }

        if ("TC".equals(selectedTool) || "WORKBENCH".equals(selectedTool) || "LOOT_ROOM".equals(selectedTool)) {
            return isHorizontalSurface(blockType) && side == Socket.CENTER_SIDE;
        }

        if (isWallType(selectedTool)) {
            if (isWallType(blockType)) {
                return side == Socket.CENTER_SIDE && currentFloor > 0 && block.getZ() == currentFloor - 1;
            }
            if (isHorizontalSurface(blockType)) {
                return side != Socket.CENTER_SIDE;
            }
            return false;
        }

        if ("FOUNDATION".equals(selectedTool) || isTriangleFoundationTool(selectedTool)) {
            return isHorizontalSurface(blockType) && side != Socket.CENTER_SIDE;
        }

        if ("FLOOR".equals(selectedTool) || "TRIANGLE_FLOOR".equals(selectedTool)) {
            if (isWallType(blockType)) {
                return side != Socket.CENTER_SIDE;
            }
            if (isHorizontalSurface(blockType)) {
                return side != Socket.CENTER_SIDE;
            }
            return false;
        }

        return side != Socket.CENTER_SIDE;
    }

    private static Orientation getOrientationFromSide(int side) {
        switch (side) {
            case 0:
                return Orientation.NORTH;
            case 1:
                return Orientation.EAST;
            case 2:
                return Orientation.SOUTH;
            case 3:
                return Orientation.WEST;
            case 4:
                return Orientation.TRIANGLE_LEFT;
            case 5:
                return Orientation.TRIANGLE_RIGHT;
            case 6:
                return Orientation.TRIANGLE_BASE;
            default:
                return Orientation.NORTH;
        }
    }

    private static boolean isFreeFoundationTool(String tool) {
        return "FOUNDATION".equals(tool) || isTriangleFoundationTool(tool);
    }

    private static boolean isTriangleFoundationTool(String tool) {
        return "TRIANGLE".equals(tool) || "TRIANGLE_FOUNDATION".equals(tool);
    }

    private static boolean isWallType(String tool) {
        return BuildingTypeUtils.isWallTool(tool);
    }

    private static boolean isWallType(BuildingType type) {
        return BuildingTypeUtils.isWall(type);
    }

    private static boolean isHorizontalSurface(BuildingType type) {
        return BuildingTypeUtils.isHorizontalSurface(type);
    }
}
