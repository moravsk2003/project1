package com.rustbuilder.service.physics;

public interface SnapResolver {
    SnappingService.SnapResult calculateSnap(double mouseX, double mouseY, String selectedTool, int currentFloor);
}
