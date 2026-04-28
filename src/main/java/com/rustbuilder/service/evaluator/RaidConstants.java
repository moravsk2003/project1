package com.rustbuilder.service.evaluator;

import com.rustbuilder.model.core.BuildingTier;

/**
 * Raid cost constants from Rust.
 * Sulfur cost = cheapest efficient method to destroy each structure per tier.
 *
 * Splash damage: A rocket (1400 sulfur) hits up to 4 adjacent walls.
 * Each wall takes ~137 splash damage per rocket.
 */
public class RaidConstants {

    // === Sulfur cost to destroy a WALL, DOOR FRAME, WINDOW FRAME, FLOOR, or FOUNDATION ===
    public static int getWallSulfurCost(BuildingTier tier) {
        switch (tier) {
            case TWIG:  return 0;      // Melee
            case WOOD:  return 250;    // Flame arrows / incendiary
            case STONE: return 4400;   // 2× C4
            case METAL: return 8800;   // 4× C4
            case HQM:   return 17600;  // 8× C4
            default:    return 0;
        }
    }

    // === Rocket constants for splash damage ===
    public static final int ROCKET_SULFUR_COST = 1400;
    public static final double ROCKET_DIRECT_DAMAGE = 350.0;   // Direct hit on a structure
    public static final double ROCKET_SPLASH_DAMAGE = 137.0;   // Splash on adjacent walls
    public static final int MAX_SPLASH_TARGETS = 4;            // Max walls hit by one rocket

    /**
     * Cost to destroy N adjacent walls with rockets using splash damage.
     * Each rocket does ROCKET_SPLASH_DAMAGE to each of N walls simultaneously.
     * Total rockets needed = ceil(wallHP / ROCKET_SPLASH_DAMAGE).
     * Total sulfur = rockets * ROCKET_SULFUR_COST.
     * Effective per-wall cost = total / N.
     */
    public static int splashSulfurCostPerWall(BuildingTier tier, int adjacentWalls) {
        if (adjacentWalls < 2 || adjacentWalls > MAX_SPLASH_TARGETS) {
            return getWallSulfurCost(tier); // No splash advantage
        }
        double hp = getWallHP(tier);
        if (hp <= 0) return 0;

        int rocketsNeeded = (int) Math.ceil(hp / ROCKET_SPLASH_DAMAGE);
        int totalSulfur = rocketsNeeded * ROCKET_SULFUR_COST;
        int perWallSulfur = totalSulfur / adjacentWalls;

        // Only use splash if it's actually cheaper per wall
        int individualCost = getWallSulfurCost(tier);
        return Math.min(perWallSulfur, individualCost);
    }

    /**
     * Wall/structure HP per tier (used for splash calculation).
     */
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
