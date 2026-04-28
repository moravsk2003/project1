package com.rustbuilder.ai.rl.multidiscrete;

import java.util.Objects;

/**
 * [RL REDESIGN]
 * A policy implementation that relies on an external provider 
 * (e.g., scripted vector, neural network output) for its 
 * multi-discrete phase decisions.
 */
public class ProvidedPhaseMultiDiscretePolicy implements MultiDiscretePhasePolicy {

    private final MultiDiscretePhaseDecisionProvider provider;

    public ProvidedPhaseMultiDiscretePolicy(MultiDiscretePhaseDecisionProvider provider) {
        this.provider = Objects.requireNonNull(provider, "Provider cannot be null for ProvidedPhaseMultiDiscretePolicy");
    }

    @Override
    public MultiDiscreteAction chooseAction(MultiDiscretePhaseContext context, MultiDiscreteStateObserver observer) {
        MultiDiscretePhaseDecision d = provider.provideDecision(context);
        
        if (observer != null) {
            observer.observePhase(context, "TYPE", 0, d.getTypeIndex());
            observer.observePhase(context, "FLOOR", 1, d.getFloorIndex());
            observer.observePhase(context, "TILE", 2, d.getTileIndex());
            observer.observePhase(context, "ROTATION", 3, d.getRotationIndex());
            observer.observePhase(context, "AIM_SECTOR", 4, d.getAimSector());
        }
        
        MultiDiscreteAction action = new MultiDiscreteAction(
                d.getTypeIndex(), 
                d.getFloorIndex(), 
                d.getTileIndex(), 
                d.getRotationIndex(), 
                d.getAimSector()
        );
        
        if (observer != null) {
            observer.onActionAssembled(context, action);
        }
        
        return action;
    }
}
