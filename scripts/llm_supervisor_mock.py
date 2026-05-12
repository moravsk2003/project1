#!/usr/bin/env python3
import json
import sys


def main():
    observation = json.load(sys.stdin)
    invalid_rate = float(observation.get("invalidActionRate", 0.0))
    epsilon = float(observation.get("epsilon", 1.0))
    best_blocks = int(observation.get("bestBaseBlocks", 0))

    if invalid_rate > 0.35:
        decision = {
            "action": "SET_EPSILON",
            "epsilon": min(1.0, epsilon + 0.10),
            "reason": "Mock policy raised exploration because invalid action rate is high.",
        }
    elif best_blocks > 20 and epsilon > 0.15:
        decision = {
            "action": "SET_EPSILON",
            "epsilon": max(0.15, epsilon - 0.05),
            "reason": "Mock policy reduced exploration after finding a non-trivial base.",
        }
    else:
        decision = {
            "action": "KEEP_GOING",
            "reason": "Mock policy sees no safe change.",
        }

    print(json.dumps(decision, separators=(",", ":")))


if __name__ == "__main__":
    main()
