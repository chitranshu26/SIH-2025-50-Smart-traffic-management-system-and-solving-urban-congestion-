# scheduler/rule_scheduler.py

MIN_GREEN = 10
MAX_GREEN = 60
CYCLE_LENGTH = 120  # total cycle time (seconds)


def proportional_allocate(counts, min_green=MIN_GREEN, max_green=MAX_GREEN, cycle=CYCLE_LENGTH):
    total = sum(counts.values()) + 1e-6
    raw = {k: (counts[k] / total) * cycle for k in counts}

    # Clip to min/max green
    clipped = {k: max(min_green, min(max_green, int(raw[k]))) for k in raw}

    # Adjust to make sum = cycle length
    s = sum(clipped.values())
    if s != cycle:
        factor = cycle / s
        clipped = {k: max(min_green, min(max_green, int(v * factor))) for k, v in clipped.items()}

    return clipped


if __name__ == "__main__":
    counts = {"lane1": 5, "lane2": 12, "lane3": 3, "lane4": 1}
    print(proportional_allocate(counts))
