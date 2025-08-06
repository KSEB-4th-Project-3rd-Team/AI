import pandas as pd
import random
import csv

def parse_rack_coords(filepath):
    rack_locs = {}
    rack_letters = [chr(ord('A') + i) for i in range(20)]    # A~T (20줄)
    rack_numbers = [f"{j:03d}" for j in range(1, 13)]        # 001~012
    valid_rack_ids = {f"{row}{col}" for row in rack_letters for col in rack_numbers}

    with open(filepath, 'r', encoding='utf-8') as f:
        for y, row in enumerate(csv.reader(f)):
            for x, c in enumerate(row):
                c = c.strip()
                if c in valid_rack_ids:
                    rack_locs[c] = (y, x)
    return rack_locs

def parse_zones(filepath):
    symbol_locs = {'I': [], 'O': []}
    with open(filepath, 'r', encoding='utf-8') as f:
        for y, row in enumerate(csv.reader(f)):
            for x, c in enumerate(row):
                c = c.strip()
                if c in symbol_locs:
                    symbol_locs[c].append((y, x))
    return symbol_locs

rack_coords = parse_rack_coords("data/map.csv")
zones = parse_zones("data/map.csv")
i_zone = zones['I']
o_zone = zones['O']

df = pd.read_csv("data/assign_task2.csv")

for idx, row in df.iterrows():
    rack_id = row['rack_id']
    t = row['type']

    if rack_id not in rack_coords:
        print(f"[경고] {rack_id}가 맵에 없음!")
        continue

    rack_y, rack_x = rack_coords[rack_id]

    if t == 'INBOUND':
        # 입고: 입고존(랜덤/라운드로빈) → 해당 랙
        start_y, start_x = random.choice(i_zone)
        goal_y, goal_x = rack_y, rack_x
    elif t == 'OUTBOUND':
        # 출고: 해당 랙 → 출고존
        start_y, start_x = rack_y, rack_x
        goal_y, goal_x = random.choice(o_zone)
    else:
        continue

    df.at[idx, 'start_y'] = start_y
    df.at[idx, 'start_x'] = start_x
    df.at[idx, 'goal_y'] = goal_y
    df.at[idx, 'goal_x'] = goal_x

    print(f"[DEBUG] idx={idx} {t} rack_id={rack_id} start=({start_y},{start_x}) goal=({goal_y},{goal_x})")

df.to_csv("data/assign_task2.csv", index=False)
print("→ 랙 기반 좌표 변환 완료")
