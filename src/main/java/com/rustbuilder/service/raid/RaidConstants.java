package com.rustbuilder.service.raid;

import com.rustbuilder.model.core.BuildingTier;

/**
 * Shared raid cost constants used by graph construction and evaluation.
 */
public final class RaidConstants {

    public static final int ROCKET_SULFUR_COST = 1400;
    public static final double ROCKET_DIRECT_DAMAGE = 350.0;
    public static final double ROCKET_SPLASH_DAMAGE = 137.0;
    public static final int MAX_SPLASH_TARGETS = 4;

    private RaidConstants() {}

    public static int getWallSulfurCost(BuildingTier tier) {
        switch (tier) {
            case TWIG:  return 0;
            case WOOD:  return 250;
            case STONE: return 4400;
            case METAL: return 8800;
            case HQM:   return 17600;
            default:    return 0;
        }
    }

    public static int getCeilingSulfurCostFromBelow(BuildingTier tier) {
        int baseCost = getWallSulfurCost(tier);
        if (baseCost <= 0) {
            return 0;
        }
        return (int) Math.ceil(baseCost / getCeilingWeakSideMultiplier(tier));
    }

    public static double getCeilingWeakSideMultiplier(BuildingTier tier) {
        switch (tier) {
            case WOOD:  return 5.0;
            case STONE: return 3.0;
            case METAL: return 2.0;
            case HQM:   return 1.0;
            case TWIG:
            default:    return 1.0;
        }
    }

    public static int splashSulfurCostPerWall(BuildingTier tier, int adjacentWalls) {
        if (adjacentWalls < 2 || adjacentWalls > MAX_SPLASH_TARGETS) {
            return getWallSulfurCost(tier);
        }
        double hp = getWallHP(tier);
        if (hp <= 0) {
            return 0;
        }

        int rocketsNeeded = (int) Math.ceil(hp / ROCKET_SPLASH_DAMAGE);
        int totalSulfur = rocketsNeeded * ROCKET_SULFUR_COST;
        int perWallSulfur = totalSulfur / adjacentWalls;

        int individualCost = getWallSulfurCost(tier);
        return Math.min(perWallSulfur, individualCost);
    }

    public static double getWallHP(BuildingTier tier) {
        switch (tier) {
            case TWIG:  return 10;
            case WOOD:  return 250;
            case STONE: return 500;
            case METAL: return 1000;
            case HQM:   return 2000;
            default:    return 0;
        }
    }
}
