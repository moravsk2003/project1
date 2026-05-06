package com.rustbuilder.core.action;

import java.io.Serializable;
import java.util.Objects;

/**
 * Domain-level command describing one requested building action.
 *
 * <p>The command is shared by manual placement, EA, and RL flows. Generation
 * strategies may create or mutate these commands, but the command itself is not
 * owned by any AI implementation.
 */
public class BuildAction implements Serializable {
    private static final long serialVersionUID = 1L;

    public enum ActionType {
        FOUNDATION,
        WALL,
        FLOOR,
        DOORWAY,
        WINDOW_FRAME,
        TRIANGLE_FOUNDATION,
        TRIANGLE_FLOOR,
        TC,
        WORKBENCH,
        LOOT_ROOM
    }

    public ActionType actionType;
    public int gridX;
    public int gridY;
    public int floor;
    public int orientation; // 0=N, 1=E, 2=S, 3=W
    public int tier;        // 0=Twig, 1=Wood, 2=Stone, 3=Metal, 4=HQM
    public int doorType;    // 0=Sheet, 1=Garage, 2=Armored
    public int aimSector;   // 0-15 grid for tile-local offset

    public BuildAction(ActionType actionType, int gridX, int gridY, int floor,
                       int orientation, int tier, int doorType) {
        this(actionType, gridX, gridY, floor, orientation, tier, doorType, 5);
    }

    public BuildAction(ActionType actionType, int gridX, int gridY, int floor,
                       int orientation, int tier, int doorType, int aimSector) {
        this.actionType = actionType;
        this.gridX = gridX;
        this.gridY = gridY;
        this.floor = floor;
        this.orientation = orientation;
        this.tier = tier;
        this.doorType = doorType;
        this.aimSector = aimSector;
    }

    @Override
    public int hashCode() {
        return Objects.hash(actionType, gridX, gridY, floor, orientation, tier, doorType, aimSector);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        BuildAction other = (BuildAction) obj;
        return actionType == other.actionType
                && gridX == other.gridX
                && gridY == other.gridY
                && floor == other.floor
                && orientation == other.orientation
                && tier == other.tier
                && doorType == other.doorType
                && aimSector == other.aimSector;
    }
}
