package com.rustbuilder.service.physics;

import com.rustbuilder.core.placement.PlacementError;
import com.rustbuilder.model.core.Orientation;
import java.util.Objects;

public sealed interface PlacementResult permits PlacementResult.Valid, PlacementResult.Invalid {

    double nearestDistance();

    double socketDistance();

    PlacementError error();

    default boolean isValid() {
        return this instanceof Valid;
    }

    record Valid(
            double x,
            double y,
            double rotation,
            Orientation orientation,
            double nearestDistance,
            double socketDistance
    ) implements PlacementResult {
        public Valid {
            Objects.requireNonNull(orientation, "orientation");
        }

        @Override
        public PlacementError error() {
            return PlacementError.NONE;
        }
    }

    record Invalid(
            PlacementError error,
            double nearestDistance,
            double socketDistance
    ) implements PlacementResult {
        public Invalid {
            if (error == null || error == PlacementError.NONE) {
                error = PlacementError.UNKNOWN;
            }
        }
    }
}
