package com.rustbuilder.ai.ea.domain;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.rustbuilder.core.action.BuildAction;
import com.rustbuilder.model.GridModel;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.BuildingTier;
import com.rustbuilder.model.core.DoorType;
import com.rustbuilder.model.core.Orientation;
import com.rustbuilder.model.structure.Door;
import com.rustbuilder.model.structure.Wall;
import com.rustbuilder.service.physics.PlacementResult;
import com.rustbuilder.service.physics.PlacementService;
import com.rustbuilder.util.BlockFactory;

/**
 * Represents a house design as a list of domain-level building actions.
 */
public class BaseGenome implements Serializable {
    private static final long serialVersionUID = 1L;

    public static final int MAX_ACTIONS = 60;
    public static final int MIN_ACTIONS = 10;
    public static final int MAX_GRID = 8;
    public static final int MAX_FLOORS = 4;
    private static final Random RNG = new Random();

    private List<BuildAction> actions;
    private double fitness = 0.0;

    public BaseGenome() {
        this.actions = new ArrayList<>();
    }

    public BaseGenome(List<BuildAction> actions) {
        this.actions = new ArrayList<>(actions);
    }

    public double getFitness() {
        return fitness;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    public BaseGenome copy() {
        BaseGenome genome = new BaseGenome(this.actions);
        genome.setFitness(this.fitness);
        return genome;
    }

    /**
     * Create a random genome with a random number of actions.
     */
    public static BaseGenome randomGenome() {
        BaseGenome genome = new BaseGenome();
        int count = MIN_ACTIONS + RNG.nextInt(MAX_ACTIONS - MIN_ACTIONS + 1);
        for (int i = 0; i < count; i++) {
            genome.actions.add(randomAction());
        }
        return genome;
    }

    /**
     * Like {@link #randomGenome()} but guarantees at least one TC action placed
     * near the grid centre. Without a TC the logistics score is always 0, which
     * makes the fitness landscape totally flat and stops evolution in its tracks.
     */
    public static BaseGenome randomGenomeWithTC() {
        BaseGenome genome = randomGenome();
        boolean hasTC = genome.actions.stream()
                .anyMatch(a -> a.actionType == BuildAction.ActionType.TC);
        if (!hasTC) {
            BuildAction tcAction = new BuildAction(
                BuildAction.ActionType.TC,
                3 + RNG.nextInt(2), 3 + RNG.nextInt(2), 0, 0, 2, 0
            );

            if (genome.actions.size() < MAX_ACTIONS) {
                genome.actions.add(tcAction);
            } else {
                genome.actions.set(RNG.nextInt(genome.actions.size()), tcAction);
            }
        }
        return genome;
    }

    /**
     * Single-point crossover between two parents.
     */
    public static BaseGenome crossover(BaseGenome parent1, BaseGenome parent2) {
        List<BuildAction> a1 = parent1.actions;
        List<BuildAction> a2 = parent2.actions;

        int cut1 = 1 + RNG.nextInt(Math.max(1, a1.size() - 1));
        int cut2 = RNG.nextInt(Math.max(1, a2.size()));

        List<BuildAction> child = new ArrayList<>();
        for (int i = 0; i < cut1 && i < a1.size(); i++) {
            child.add(a1.get(i));
        }
        for (int i = cut2; i < a2.size(); i++) {
            child.add(a2.get(i));
        }

        if (child.size() > MAX_ACTIONS) {
            child = new ArrayList<>(child.subList(0, MAX_ACTIONS));
        }
        while (child.size() < MIN_ACTIONS) {
            child.add(randomAction());
        }

        return new BaseGenome(child);
    }

    /**
     * Mutate the genome by modifying, adding, or removing random actions.
     */
    public void mutate(double mutationRate) {
        for (int i = 0; i < actions.size(); i++) {
            if (RNG.nextDouble() < mutationRate) {
                actions.set(i, mutatedAction(actions.get(i)));
            }
        }
        if (RNG.nextDouble() < mutationRate && actions.size() < MAX_ACTIONS) {
            actions.add(RNG.nextInt(actions.size() + 1), randomAction());
        }
        if (RNG.nextDouble() < mutationRate && actions.size() > MIN_ACTIONS) {
            actions.remove(RNG.nextInt(actions.size()));
        }

        boolean hasTC = actions.stream().anyMatch(a -> a.actionType == BuildAction.ActionType.TC);
        if (!hasTC && actions.size() < MAX_ACTIONS && RNG.nextDouble() < 0.30) {
            BuildAction tcAction = new BuildAction(
                BuildAction.ActionType.TC,
                3 + RNG.nextInt(2), 3 + RNG.nextInt(2), 0, 0, 2, 0
            );
            actions.add(tcAction);
        }
    }

    /**
     * Decode this genome into building blocks on the given GridModel.
     * Returns true if at least some blocks were placed.
     */
    public boolean decode(GridModel gridModel) {
        gridModel.clear();

        int placed = 0;

        // Phase order ensures that when a wall/floor tries to snap, foundations
        // already exist in the grid. Furniture is placed last so it lands on a
        // built floor rather than floating in the air.
        for (int phase = 0; phase < 3; phase++) {
            for (BuildAction action : actions) {
                boolean isFoundation = action.actionType == BuildAction.ActionType.FOUNDATION
                                    || action.actionType == BuildAction.ActionType.TRIANGLE_FOUNDATION;
                boolean isFurniture = action.actionType == BuildAction.ActionType.TC
                                    || action.actionType == BuildAction.ActionType.WORKBENCH
                                    || action.actionType == BuildAction.ActionType.LOOT_ROOM;
                boolean isStructural = !isFoundation && !isFurniture;

                if (phase == 0 && !isFoundation) continue;
                if (phase == 1 && !isStructural) continue;
                if (phase == 2 && !isFurniture) continue;

                PlacementResult placement = PlacementService.calculatePlacement(gridModel, action);
                if (!(placement instanceof PlacementResult.Valid validPlacement)) {
                    continue;
                }

                double finalX = validPlacement.x();
                double finalY = validPlacement.y();
                double finalRotation = validPlacement.rotation();
                Orientation finalOrientation = validPlacement.orientation();
                int z = isFoundation ? 0 : action.floor;

                BuildingTier tier = BlockFactory.tierFromInt(action.tier);
                DoorType doorType = BlockFactory.doorTypeFromInt(action.doorType);
                BuildingBlock block = BlockFactory.create(
                        action.actionType, finalX, finalY, z,
                        finalRotation, finalOrientation, doorType);

                if (action.actionType == BuildAction.ActionType.DOORWAY && block != null) {
                    Door door = new Door(finalX, finalY, z, finalOrientation, doorType);
                    door.setTier(tier);
                    if (gridModel.canPlace(door)) {
                        gridModel.addBlockSilent(door);
                    }
                }

                if (block != null) {
                    if (!(block instanceof Wall)) {
                        block.setRotation(finalRotation);
                    }
                    block.setTier(tier);

                    if (isFurniture || isFoundation) {
                        if (gridModel.canPlace(block)) {
                            gridModel.addBlockSilent(block);
                            placed++;
                        }
                    } else if (!gridModel.hasCollision(block) && gridModel.addBlockSilent(block)) {
                        placed++;
                    }
                }
            }
        }

        gridModel.finalizeLoad();
        return placed > 0;
    }

    private static BuildAction randomAction() {
        BuildAction.ActionType type = getRandomActionType();

        int gx = (int) Math.round(3.5 + RNG.nextGaussian() * 2.0);
        int gy = (int) Math.round(3.5 + RNG.nextGaussian() * 2.0);
        gx = Math.max(0, Math.min(MAX_GRID - 1, gx));
        gy = Math.max(0, Math.min(MAX_GRID - 1, gy));

        return new BuildAction(
            type,
            gx, gy,
            RNG.nextInt(MAX_FLOORS),
            RNG.nextInt(4),
            2 + RNG.nextInt(3),
            RNG.nextInt(3),
            5
        );
    }

    private static BuildAction mutatedAction(BuildAction action) {
        int field = RNG.nextInt(7);

        BuildAction.ActionType newType = action.actionType;
        int newX = action.gridX;
        int newY = action.gridY;
        int newFloor = action.floor;
        int newOrient = action.orientation;
        int newTier = action.tier;
        int newDoor = action.doorType;

        switch (field) {
            case 0: newType = getRandomActionType(); break;
            case 1: newX = Math.max(0, Math.min(MAX_GRID - 1, action.gridX + (int)Math.round(RNG.nextGaussian()))); break;
            case 2: newY = Math.max(0, Math.min(MAX_GRID - 1, action.gridY + (int)Math.round(RNG.nextGaussian()))); break;
            case 3: newFloor = Math.max(0, Math.min(MAX_FLOORS - 1, action.floor + (RNG.nextBoolean() ? 1 : -1))); break;
            case 4: newOrient = (action.orientation + (RNG.nextBoolean() ? 1 : -1) + 4) % 4; break;
            case 5: newTier = Math.max(2, Math.min(4, action.tier + (RNG.nextBoolean() ? 1 : -1))); break;
            case 6: newDoor = (action.doorType + 1) % 3; break;
        }

        return new BuildAction(newType, newX, newY, newFloor, newOrient, newTier, newDoor, action.aimSector);
    }

    private static BuildAction.ActionType getRandomActionType() {
        double r = RNG.nextDouble();
        if (r < 0.20) return BuildAction.ActionType.FOUNDATION;
        if (r < 0.37) return BuildAction.ActionType.WALL;
        if (r < 0.52) return BuildAction.ActionType.FLOOR;
        if (r < 0.59) return BuildAction.ActionType.TRIANGLE_FOUNDATION;
        if (r < 0.66) return BuildAction.ActionType.TRIANGLE_FLOOR;
        if (r < 0.78) return BuildAction.ActionType.DOORWAY;
        if (r < 0.83) return BuildAction.ActionType.WINDOW_FRAME;
        if (r < 0.93) return BuildAction.ActionType.TC;
        if (r < 0.96) return BuildAction.ActionType.LOOT_ROOM;
        return BuildAction.ActionType.WORKBENCH;
    }

    public List<BuildAction> getActions() {
        return actions;
    }

    @Override
    public int hashCode() {
        return java.util.Objects.hash(actions);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        BaseGenome other = (BaseGenome) obj;
        return java.util.Objects.equals(actions, other.actions);
    }
}
