package com.rustbuilder.model.structure;

import com.rustbuilder.model.core.*;
import java.util.ArrayList;
import java.util.List;

public class Wall extends BuildingBlock {

    private Orientation orientation;
    private DoorType doorType = DoorType.SHEET_METAL; // default door type for doorways

    public Wall(double x, double y, int z, Orientation orientation) {
        super(BuildingType.WALL, x, y, z);
        this.orientation = orientation;
    }

    public Orientation getOrientation() {
        return orientation;
    }

    public void setOrientation(Orientation orientation) {
        this.orientation = orientation;
        invalidateGeometryCache();
    }

    public DoorType getDoorType() {
        return doorType;
    }

    public void setDoorType(DoorType doorType) {
        this.doorType = doorType;
    }

    // Removed override getUpkeepCost and getHealth

    @Override
    protected BuildingBlock copyBlock() {
        Wall copy = new Wall(getX(), getY(), getZ(), orientation);
        copy.setType(getType());
        copy.setDoorType(doorType);
        return copy;
    }

    private double[] calculatePoints(boolean isCollision) {
        double size = com.rustbuilder.config.GameConstants.TILE_SIZE;
        double thickness = com.rustbuilder.config.GameConstants.WALL_THICKNESS;
        double half = com.rustbuilder.config.GameConstants.HALF_TILE;
        double offset = com.rustbuilder.config.GameConstants.TRIANGLE_OFFSET;

        double localCx = 0, localCy = 0, localW = 0, localH = 0, localRot = 0;

        switch (orientation) {
            case NORTH:
                localCx = half; localCy = 0; localW = size; localH = thickness; break;
            case SOUTH:
            case TRIANGLE_BASE:
                localCx = half; localCy = size; localW = size; localH = thickness; break;
            case WEST:
                localCx = 0; localCy = half; localW = thickness; localH = size; break;
            case EAST:
                localCx = size; localCy = half; localW = thickness; localH = size; break;
            case TRIANGLE_LEFT:
                localCx = half / 2; localCy = half + offset / 2; localW = size; localH = thickness; localRot = -60; break;
            case TRIANGLE_RIGHT:
                localCx = half * 1.5; localCy = half + offset / 2; localW = size; localH = thickness; localRot = 60; break;
        }

        if (isCollision) {
            double shorten = thickness;
            if (orientation == Orientation.TRIANGLE_LEFT || 
                orientation == Orientation.TRIANGLE_RIGHT || 
                orientation == Orientation.TRIANGLE_BASE) {
                shorten = thickness * 2.0;
            } else {
                 shorten = thickness * 1.1; // Slightly more than thickness for squares to be safe
            }
            if (localW > localH) localW -= shorten; else localH -= shorten;
        } else {
            double scale = 0.98;
            localW *= scale;
            localH *= scale;
        }

        double hw = localW / 2;
        double hh = localH / 2;
        double[] corners = { -hw, -hh, hw, -hh, hw, hh, -hw, hh };
        double[] points = new double[8];

        for (int i = 0; i < 4; i++) {
            double px = corners[i * 2];
            double py = corners[i * 2 + 1];

            if (localRot != 0) {
                double[] rotated = com.rustbuilder.util.TransformUtils.rotatePoint(px, py, localRot);
                px = rotated[0]; py = rotated[1];
            }

            double relX = px + localCx - half;
            double relY = py + localCy - half;
            double[] finalRotated = com.rustbuilder.util.TransformUtils.rotatePoint(relX, relY, getRotation());
            
            points[i * 2] = finalRotated[0] + half + getX();
            points[i * 2 + 1] = finalRotated[1] + half + getY();
        }
        return points;
    }

    @Override
    protected double[] computePolygonPoints() {
        return calculatePoints(false);
    }

    @Override
    protected double[] computeCollisionPoints() {
        return calculatePoints(true);
    }

    @Override
    protected List<Socket> computeSockets() {
        List<Socket> sockets = new ArrayList<>();
        double size = com.rustbuilder.config.GameConstants.TILE_SIZE;
        double cx = getX() + size / 2;
        double cy = getY() + size / 2;
        
        // Add a center socket for stacking walls or placing floors
        // The side depends on orientation
        // Add a center socket for stacking walls or placing floors
        // The side depends on orientation
        
        // We provide a socket ON the wall itself to make snapping easier from both sides
        // Side 10 indicates "Center/Top" - MainApp will snap to the block origin regardless of this socket's exact pos
        // We place it at the CENTER of the tile to distinguish it from the Edge Sockets (0-3)
        double sx = cx;
        double sy = cy;
        
        // Previously we moved this to the edge, but that conflicts with the specific Edge Sockets
        // Now Side 10 is strictly Center (Inside), and Edge Sockets are for Outside.

        sockets.add(new Socket(sx, sy, getRotation(), Socket.CENTER_SIDE));

        // Add edge sockets for "Outside" placement (Ceilings/Walls)
        // These sockets are located at the "outer" edge of the wall
        double half = com.rustbuilder.config.GameConstants.HALF_TILE;
        double offset = com.rustbuilder.config.GameConstants.TRIANGLE_OFFSET;
        
        double ex = cx;
        double ey = cy;
        int edgeSide = -1;

        switch (orientation) {
            case NORTH: 
                ex = cx; ey = cy - half; edgeSide = 0; 
                break;
            case SOUTH: 
                ex = cx; ey = cy + half; edgeSide = 2; 
                break;
            case WEST: 
                ex = cx - half; ey = cy; edgeSide = 3; 
                break;
            case EAST: 
                ex = cx + half; ey = cy; edgeSide = 1; 
                break;
            case TRIANGLE_BASE:
                // Base is at the bottom (South)
                ex = cx; ey = cy + half; edgeSide = 6;
                break;
            case TRIANGLE_LEFT:
                // Left slope midpoint
                // Offset relative to center: -half/2, offset/2
                // Need to rotate? No, these are fixed offsets for the triangle shape
                // TriangleFloor uses: { -half/2, offset/2, 4 } relative to center
                // But wait, TriangleFloor center is different?
                // TriangleFloor center is geometric center.
                // Wall center is Tile Center.
                // TriangleFloor fits in the tile.
                // So relative to Tile Center, the offsets should be the same.
                ex = cx - half / 2; 
                ey = cy + offset / 2; 
                edgeSide = 4;
                break;
            case TRIANGLE_RIGHT:
                // Right slope midpoint
                ex = cx + half / 2; 
                ey = cy + offset / 2; 
                edgeSide = 5;
                break;
        }

        if (edgeSide != -1) {
            // Rotate the edge socket position if the wall itself is rotated (though usually walls are 0 rot)
            if (getRotation() != 0) {
                double[] rotated = com.rustbuilder.util.TransformUtils.rotatePoint(ex - cx, ey - cy, getRotation());
                ex = cx + rotated[0];
                ey = cy + rotated[1];
            }
            sockets.add(new Socket(ex, ey, getRotation(), edgeSide));
        }
        
        return sockets;
    }
}
