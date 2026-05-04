package com.rustbuilder.ai.rl.env.state;

import com.rustbuilder.model.GridModel;

public interface StateRepresentationEncoder {
    EncodedState encode(GridModel gridModel, int selectedTypeOrdinal);
}
