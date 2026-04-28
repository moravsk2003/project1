package com.rustbuilder.service.economy;

import java.util.HashMap;
import java.util.List; // Assuming we might need constants
import java.util.Map;
import com.rustbuilder.model.core.BuildingBlock;
import com.rustbuilder.model.core.ResourceType;

public class UpkeepService {

    public static Map<ResourceType, Integer> calculateUpkeep(List<BuildingBlock> blocks) {
        Map<ResourceType, Integer> totalUpkeep = new HashMap<>();
        Map<ResourceType, Integer> totalCost = new HashMap<>();
        
        for (BuildingBlock b : blocks) {
            b.getBuildCost().forEach((res, amt) -> totalCost.merge(res, amt, Integer::sum));
        }
        
        int n = blocks.size();
        if (n == 0) return totalUpkeep;
        
        double factorSum = 0;
        for (int i = 1; i <= n; i++) {
            if (i <= 15) factorSum += 0.10;
            else if (i <= 50) factorSum += 0.15;
            else if (i <= 125) factorSum += 0.20;
            else factorSum += 0.333333333;
        }
        
        double averageMultiplier = factorSum / n;
        
        for (Map.Entry<ResourceType, Integer> entry : totalCost.entrySet()) {
            double upkeep = entry.getValue() * averageMultiplier;
            int val = (int) Math.round(upkeep);
            if (val < 1 && entry.getValue() > 0) val = 1;
            totalUpkeep.put(entry.getKey(), val);
        }
        
        return totalUpkeep;
    }
}
