package com.rustbuilder.model.core;

import java.util.List;
import java.util.UUID;
import java.util.Collections;
import java.util.concurrent.ConcurrentHashMap;

public abstract class BuildingBlock {
    private final String id;
    private BuildingType type;
    private BuildingTier tier;
    private double x;
    private double y;
    private int z; // Floor
    private double rotation; // Degrees
    protected java.util.Map<ResourceType, Integer> buildCost;

    private static final java.util.Map<String, java.util.Map<ResourceType, Integer>> COST_CACHE = new ConcurrentHashMap<>();

    private double[] cachedPolygonPoints = null;
    private double[] cachedCollisionPoints = null;
    private List<Socket> cachedSockets = null;
    // Split-cache: x[]/y[] derived from cachedPolygonPoints — avoids per-frame allocation in draw path
    private double[] cachedXPoints = null;
    private double[] cachedYPoints = null;

    public BuildingBlock(BuildingType type, double x, double y, int z) {
        this.id = UUID.randomUUID().toString();
        this.type = type;
        this.tier = BuildingTier.TWIG; // Default
        this.x = x;
        this.y = y;
        this.z = z;
        this.rotation = 0;
        this.stability = 0.0;
        updateCost(); 
    }

    private double stability;

    public double getStability() {
        return stability;
    }

    public void setStability(double stability) {
        this.stability = stability;
    }

    /**
     * Prototype-style clone. Subclasses only provide the constructor-specific
     * copy; shared block state is copied here in one place.
     */
    @Override
    public final BuildingBlock clone() {
        BuildingBlock clone = copyBlock();
        clone.setType(getType());
        clone.setTier(getTier());
        clone.setTransform(getX(), getY(), getZ(), getRotation());
        clone.setStability(getStability());
        return clone;
    }

    protected abstract BuildingBlock copyBlock();

    public String getId() {
        return id;
    }

    public BuildingType getType() {
        return type;
    }

    public void invalidateGeometryCache() {
        cachedPolygonPoints = null;
        cachedCollisionPoints = null;
        cachedSockets = null;
        cachedXPoints = null;
        cachedYPoints = null;
    }

    public void setType(BuildingType type) {
        this.type = type;
        updateCost(); 
        invalidateGeometryCache();
    }

    public BuildingTier getTier() {
        return tier;
    }

    public void setTier(BuildingTier tier) {
        this.tier = tier;
        updateCost();
    }

    protected void updateCost() {
        String cacheKey = type.name() + "_" + tier.name();
        if (COST_CACHE.containsKey(cacheKey)) {
            this.buildCost = COST_CACHE.get(cacheKey);
            return;
        }

        java.util.Map<ResourceType, Integer> cost = new java.util.HashMap<>();
        int amount = 0;
        ResourceType resType = ResourceType.WOOD;

        if (type == BuildingType.TC) {
             cost.put(ResourceType.WOOD, 1000);
             this.buildCost = Collections.unmodifiableMap(cost);
             return;
        }

        int fullCost = 0;
        switch (tier) {
            case TWIG:
                fullCost = 50;
                resType = ResourceType.WOOD;
                break;
            case WOOD:
                fullCost = 200;
                resType = ResourceType.WOOD;
                break;
            case STONE:
                fullCost = 300;
                resType = ResourceType.STONE;
                break;
            case METAL:
                fullCost = 200;
                resType = ResourceType.METAL;
                break;
            case HQM:
                fullCost = 25;
                resType = ResourceType.HQM;
                break;
        }

        double multiplier = 1.0;
        if (type == BuildingType.FLOOR || type == BuildingType.TRIANGLE_FOUNDATION) {
             multiplier = 0.5;
        } else if (type == BuildingType.TRIANGLE_FLOOR) {
             multiplier = 0.25; 
        }

        amount = (int) Math.ceil(fullCost * multiplier);
        cost.put(resType, amount);
        
        COST_CACHE.put(cacheKey, Collections.unmodifiableMap(cost));
        this.buildCost = COST_CACHE.get(cacheKey);
    }
    
    public java.util.Map<ResourceType, Integer> getBuildCost() {
        return buildCost;
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
        invalidateGeometryCache();
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
        invalidateGeometryCache();
    }

    public int getZ() {
        return z;
    }

    public void setZ(int z) {
        this.z = z;
        invalidateGeometryCache();
    }

    public double getRotation() {
        return rotation;
    }

    public void setRotation(double rotation) {
        this.rotation = rotation;
        invalidateGeometryCache();
    }

    public void setTransform(double x, double y, int z, double rotation) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.rotation = rotation;
        invalidateGeometryCache();
    }

    public double getUpkeepCost() {
        double base = 0;
        switch (getTier()) {
            case TWIG: base = 0; break;
            case WOOD: base = 10; break;
            case STONE: base = 20; break;
            case METAL: base = 30; break;
            case HQM: base = 50; break;
        }
        if (type == BuildingType.TRIANGLE_FOUNDATION || type == BuildingType.TRIANGLE_FLOOR) {
            return base * 0.5;
        }
        return base;
    }

    public double getHealth() {
        switch (getTier()) {
            case TWIG: return 10;
            case WOOD: return 250;
            case STONE: return 500;
            case METAL: return 1000;
            case HQM: return 2000;
            default: return 0;
        }
    }

    public final List<Socket> getSockets() {
        if (cachedSockets == null) {
            cachedSockets = computeSockets();
        }
        return cachedSockets;
    }

    protected abstract List<Socket> computeSockets();

    public final double[] getPolygonPoints() {
        if (cachedPolygonPoints == null) {
            cachedPolygonPoints = computePolygonPoints();
        }
        return cachedPolygonPoints;
    }

    protected abstract double[] computePolygonPoints();

    public final double[] getCollisionPoints() {
        if (cachedCollisionPoints == null) {
            cachedCollisionPoints = computeCollisionPoints();
        }
        return cachedCollisionPoints;
    }

    protected double[] computeCollisionPoints() {
        return getPolygonPoints();
    }

    /**
     * Returns the cached x-coordinates of the polygon vertices.
     * Derived lazily from {@link #getPolygonPoints()} to avoid per-frame array allocation
     * in the hot render path. Invalidated by {@link #invalidateGeometryCache()}.
     */
    public final double[] getSplitPolygonX() {
        if (cachedXPoints == null) {
            buildSplitCache();
        }
        return cachedXPoints;
    }

    /**
     * Returns the cached y-coordinates of the polygon vertices.
     * @see #getSplitPolygonX()
     */
    public final double[] getSplitPolygonY() {
        if (cachedYPoints == null) {
            buildSplitCache();
        }
        return cachedYPoints;
    }

    private void buildSplitCache() {
        double[] pts = getPolygonPoints();
        int n = pts.length / 2;
        cachedXPoints = new double[n];
        cachedYPoints = new double[n];
        for (int i = 0; i < n; i++) {
            cachedXPoints[i] = pts[i * 2];
            cachedYPoints[i] = pts[i * 2 + 1];
        }
    }
}
