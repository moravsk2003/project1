package com.rustbuilder.ai.rl.env.state;

import org.nd4j.linalg.api.ndarray.INDArray;
import com.rustbuilder.ai.rl.env.spec.StateEncodingSpec;

public class EncodedState implements AutoCloseable {
    private INDArray voxelTensor;
    private INDArray globalVector;
    private Object objectTable;
    private Object graphState;
    private final StateEncodingSpec spec;
    private final StateEncodingDiagnostics diagnostics;
    
    private boolean closed = false;

    public EncodedState(INDArray voxelTensor, StateEncodingSpec spec) {
        this.voxelTensor = voxelTensor;
        this.spec = spec;
        this.diagnostics = new StateEncodingDiagnostics();
    }
    
    public EncodedState(INDArray voxelTensor, INDArray globalVector, Object objectTable, Object graphState, StateEncodingSpec spec, StateEncodingDiagnostics diagnostics) {
        this.voxelTensor = voxelTensor;
        this.globalVector = globalVector;
        this.objectTable = objectTable;
        this.graphState = graphState;
        this.spec = spec;
        this.diagnostics = diagnostics != null ? diagnostics : new StateEncodingDiagnostics();
    }

    public INDArray voxelTensor() {
        return voxelTensor;
    }
    
    public INDArray getGlobalVector() { return globalVector; }
    public void setGlobalVector(INDArray globalVector) { this.globalVector = globalVector; }
    
    public Object getObjectTable() { return objectTable; }
    public void setObjectTable(Object objectTable) { this.objectTable = objectTable; }
    
    public Object getGraphState() { return graphState; }
    public void setGraphState(Object graphState) { this.graphState = graphState; }
    
    public StateEncodingSpec getSpec() { return spec; }
    public StateEncodingDiagnostics getDiagnostics() { return diagnostics; }

    public EncodedState dup() {
        INDArray dupTensor = voxelTensor != null ? voxelTensor.dup() : null;
        INDArray dupGlobal = globalVector != null ? globalVector.dup() : null;
        EncodedState copy = new EncodedState(dupTensor, spec);
        copy.setGlobalVector(dupGlobal);
        copy.setObjectTable(objectTable);
        copy.setGraphState(graphState);
        return copy;
    }

    @Override
    public void close() {
        if (!closed) {
            if (voxelTensor != null && !voxelTensor.wasClosed()) {
                voxelTensor.close();
            }
            if (globalVector != null && !globalVector.wasClosed()) {
                globalVector.close();
            }
            closed = true;
        }
    }
    
    public boolean wasClosed() {
        return closed;
    }
}
