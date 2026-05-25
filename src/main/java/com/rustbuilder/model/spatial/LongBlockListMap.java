package com.rustbuilder.model.spatial;

import com.rustbuilder.model.core.BuildingBlock;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class LongBlockListMap {
    private static final int DEFAULT_CAPACITY = 32;
    private static final float MAX_LOAD = 0.65f;

    private long[] keys;
    private List<BuildingBlock>[] values;
    private byte[] states; // 0 = empty, 1 = occupied, 2 = deleted
    private int size;
    private int tombstones;
    private int resizeThreshold;

    public LongBlockListMap() {
        allocate(DEFAULT_CAPACITY);
    }

    public List<BuildingBlock> get(long key) {
        int mask = keys.length - 1;
        int index = mix(key) & mask;
        while (true) {
            byte state = states[index];
            if (state == 0) {
                return null;
            }
            if (state == 1 && keys[index] == key) {
                return values[index];
            }
            index = (index + 1) & mask;
        }
    }

    public List<BuildingBlock> getOrCreate(long key) {
        if (size + tombstones + 1 > resizeThreshold) {
            rehash(keys.length * 2);
        }

        int mask = keys.length - 1;
        int index = mix(key) & mask;
        int firstDeleted = -1;
        while (true) {
            byte state = states[index];
            if (state == 0) {
                int target = firstDeleted >= 0 ? firstDeleted : index;
                if (firstDeleted >= 0) {
                    tombstones--;
                }
                keys[target] = key;
                states[target] = 1;
                values[target] = new ArrayList<>();
                size++;
                return values[target];
            }
            if (state == 1 && keys[index] == key) {
                return values[index];
            }
            if (state == 2 && firstDeleted < 0) {
                firstDeleted = index;
            }
            index = (index + 1) & mask;
        }
    }

    public void remove(long key) {
        int mask = keys.length - 1;
        int index = mix(key) & mask;
        while (true) {
            byte state = states[index];
            if (state == 0) {
                return;
            }
            if (state == 1 && keys[index] == key) {
                states[index] = 2;
                values[index] = null;
                size--;
                tombstones++;
                if (tombstones > size && keys.length > DEFAULT_CAPACITY) {
                    rehash(keys.length);
                }
                return;
            }
            index = (index + 1) & mask;
        }
    }

    public void clear() {
        Arrays.fill(states, (byte) 0);
        Arrays.fill(values, null);
        size = 0;
        tombstones = 0;
    }

    private void rehash(int capacity) {
        long[] oldKeys = keys;
        List<BuildingBlock>[] oldValues = values;
        byte[] oldStates = states;
        allocate(capacity);

        for (int i = 0; i < oldKeys.length; i++) {
            if (oldStates[i] == 1) {
                putRehashed(oldKeys[i], oldValues[i]);
            }
        }
    }

    private void putRehashed(long key, List<BuildingBlock> value) {
        int mask = keys.length - 1;
        int index = mix(key) & mask;
        while (states[index] == 1) {
            index = (index + 1) & mask;
        }
        keys[index] = key;
        values[index] = value;
        states[index] = 1;
        size++;
    }

    @SuppressWarnings("unchecked")
    private void allocate(int requestedCapacity) {
        int capacity = 1;
        while (capacity < requestedCapacity) {
            capacity <<= 1;
        }
        keys = new long[capacity];
        values = (List<BuildingBlock>[]) new List[capacity];
        states = new byte[capacity];
        size = 0;
        tombstones = 0;
        resizeThreshold = Math.max(1, (int) (capacity * MAX_LOAD));
    }

    private static int mix(long value) {
        value ^= value >>> 33;
        value *= 0xff51afd7ed558ccdL;
        value ^= value >>> 33;
        value *= 0xc4ceb9fe1a85ec53L;
        value ^= value >>> 33;
        return (int) value;
    }
}
