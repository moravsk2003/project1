package com.rustbuilder.service.physics;

import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.Orientation;

public class SnappingService implements SnapResolver {

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

    @Override
    public SnapResult calculateSnap(double mouseX, double mouseY, String selectedTool, int currentFloor) {
        SocketPlacementResolver.Result result = SocketPlacementResolver.resolve(
                gridModel, mouseX, mouseY, selectedTool, currentFloor, true);
        return new SnapResult(result.x, result.y, result.rotation, result.orientation, result.valid);
    }
}
