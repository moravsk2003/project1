package com.rustbuilder.service.graph;

import java.util.Objects;

public class NodeKey {
    public final int x;
    public final int y;
    public final int z;
    public final String type;

    public NodeKey(double x, double y, int z, String type) {
        this.x = (int) Math.round(x);
        this.y = (int) Math.round(y);
        this.z = z;
        this.type = type;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        NodeKey nodeKey = (NodeKey) o;
        return x == nodeKey.x &&
               y == nodeKey.y &&
               z == nodeKey.z &&
               Objects.equals(type, nodeKey.type);
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y, z, type);
    }

    @Override
    public String toString() {
        return x + "_" + y + "_" + z + "_" + type;
    }
}
