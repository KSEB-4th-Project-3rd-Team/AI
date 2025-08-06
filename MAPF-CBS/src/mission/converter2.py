import pandas as pd
import random
import csv

def parse_zones(filepath):
    symbol_locs = {'I': [], 'O': []}
    with open(filepath, 'r', encoding='utf-8') as f:
        for y, row in enumerate(csv.reader(f, delimiter='\t')):
            for x, c in enumerate(row):
                c = c.strip()
                if c in symbol_locs:
                    symbol_locs[c].append((y, x))
    return symbol_locs

# 1. 메인 랙 좌표 자동화(20줄, 24개/줄, 2칸씩)
rack_letters = [chr(ord('A') + i) for i in range(48)]     # A ~ AV
rack_numbers = [f"{j:03d}" for j in range(1, 25)]          # 001~024

main_rack_coords = {}
with open("data/map.txt", "r", encoding="utf-8") as f:
    map_lines = [line.rstrip('\n').split('\t') for line in f]
    for row_i, y in enumerate(range(4, 52)):  # 4~51행 (20줄)
        row = map_lines[y]
        i, rack_col = 0, 0
        while i < len(row) - 3:
            if row[i] == row[i+1] == row[i+2] == row[i+3] == '@':
                # 왼쪽 랙
                rack_id_left = f"{rack_letters[row_i]}{rack_numbers[rack_col]}"
                main_rack_coords[rack_id_left] = (y, i)
                rack_col += 1
                # 오른쪽 랙
                rack_id_right = f"{rack_letters[row_i]}{rack_numbers[rack_col]}"
                main_rack_coords[rack_id_right] = (y, i+2)
                rack_col += 1
                i += 4
            else:
                i += 1


# 2. 작은 랙 (58~71행)
small_rack_coords = {}
with open("data/map.txt", "r", encoding="utf-8") as f:
    map_lines = [line.rstrip('\n').split('\t') for line in f]
    for row_i, y in enumerate(range(58, 72)):
        row = map_lines[y]
        i, rack_col = 0, 0
        while i < len(row) - 3:
            if row[i] == row[i+1] == row[i+2] == row[i+3] == '@':
                rack_id = f"S{(row_i*8)+rack_col+1:03d}"
                small_rack_coords[rack_id] = (y, i)
                rack_col += 1
                i += 4
            else:
                i += 1

# 3. 관리소(64~79행)
mgmt_coords = {}
with open("data/map.txt", "r", encoding="utf-8") as f:
    map_lines = [line.rstrip('\n').split('\t') for line in f]
    mgmt_id = 1
    for y in range(64, 79):  # 64~78 (79 미포함)
        row = map_lines[y]
        for x in range(len(row) - 3):
            if row[x] == row[x+1] == row[x+2] == row[x+3] == '@':
                for dx in range(4):
                    mgmt_coords[f"MGMT{mgmt_id:03d}"] = (y, x + dx)
                    mgmt_id += 1

# 4. 입/출고존
zones = parse_zones("data/map.txt")
in_coords = {f"IN{idx+1:03d}": xy for idx, xy in enumerate(zones['I'])}
out_coords = {f"OUT{idx+1:03d}": xy for idx, xy in enumerate(zones['O'])}

# 5. 모든 rack id + 좌표 통합
all_coords = {}
all_coords.update(main_rack_coords)
all_coords.update(small_rack_coords)
all_coords.update(mgmt_coords)
all_coords.update(in_coords)
all_coords.update(out_coords)

# 6. 미션 좌표 자동 할당
df = pd.read_csv("data/assign_task2.csv")
for idx, row in df.iterrows():
    rack_id = row['rack_id']
    t = row['type']
    if rack_id not in all_coords:
        print(f"[경고] {rack_id}가 모든 구역에서 매칭되지 않음!")
        continue
    rack_y, rack_x = all_coords[rack_id]
    if t == 'INBOUND':
        start_y, start_x = random.choice(list(in_coords.values()))
        goal_y, goal_x = rack_y, rack_x
    elif t == 'OUTBOUND':
        start_y, start_x = rack_y, rack_x
        goal_y, goal_x = random.choice(list(out_coords.values()))
    else:
        continue
    df.at[idx, 'start_y'] = start_y
    df.at[idx, 'start_x'] = start_x
    df.at[idx, 'goal_y'] = goal_y
    df.at[idx, 'goal_x'] = goal_x
    print(f"[DEBUG] idx={idx} {t} rack_id={rack_id} start=({start_y},{start_x}) goal=({goal_y},{goal_x})")
df.to_csv("data/assign_task2.csv", index=False)
print("→ 모든 구역 기반 좌표 변환 완료")

# 7. map 확인 (debug용)
# with open("data/map.txt", "r", encoding="utf-8") as f:
#     for y, line in enumerate(f):
#         print(f"{y:02d}: {line.rstrip()}")
