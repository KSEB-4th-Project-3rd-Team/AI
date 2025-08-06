import pandas as pd
import random
import csv

# --- map.csv에서 zone별 좌표 추출 ---
def parse_zones(filepath):
    """
    map.csv에서 'I'(입고 zone), 'O'(출고 zone), '.', '@' 좌표 집합 추출
    return: dict{'I': [...], 'O': [...], '.': [...], '@': [...]}
    """
    import csv
    symbol_locs = {'I': [], 'O': [], '@': [], '.' : []}
    with open(filepath, 'r', encoding='utf-8') as f:
        for y, row in enumerate(csv.reader(f)):
            for x, c in enumerate(row):
                c = c.strip()
                if c in symbol_locs:
                    symbol_locs[c].append((y, x))
    return symbol_locs

# 1. zone 추출
zones = parse_zones("data/map.csv")
i_zone = zones['I']
o_zone = zones['O']
free_zone = zones['.']

# 2. 미션별 좌표 변환 (여기서 타입별 로직 모두 적용)
df = pd.read_csv("data/missions.csv")
i_idx, o_idx = 0, 0  # 라운드로빈 인덱스(원하면 zone별 별도 관리 가능)
for idx, row in df.iterrows():
    t = row['type']
    # INBOUND, PICK: start in I zone, goal in 랙/빈칸
    if t in ('INBOUND', 'PICK'):
        # start: I zone 내 임의 좌표
        start_y, start_x = random.choice(i_zone)  # 또는 i_zone[i_idx % len(i_zone)]
        # goal: 임의의 랙 or free_zone
        goal_y, goal_x = random.choice(free_zone)
    # OUTBOUND, DROP: start in 랙/빈칸, goal in O zone
    elif t in ('OUTBOUND', 'DROP'):
        start_y, start_x = random.choice(free_zone)
        goal_y, goal_x = random.choice(o_zone)    # 또는 o_zone[o_idx % len(o_zone)]
    # MOVE: 임의 빈칸끼리
    elif t == 'MOVE':
        points = random.sample(free_zone, 2)
        start_y, start_x = points[0]
        goal_y, goal_x = points[1]
    # 기타: I zone → O zone
    else:
        start_y, start_x = random.choice(i_zone)
        goal_y, goal_x = random.choice(o_zone)

    df.at[idx, 'start_y'] = start_y
    df.at[idx, 'start_x'] = start_x
    df.at[idx, 'goal_y'] = goal_y
    df.at[idx, 'goal_x'] = goal_x

    # 라운드로빈 인덱스 증가 (원하면)
    # i_idx += 1; o_idx += 1

    print(f"[DEBUG] idx={idx} type={t} start=({start_y},{start_x}) goal=({goal_y},{goal_x})")

df.to_csv("data/assign_task.csv", index=False)
print("좌표 집합 기반 변환 완료 → data/assign_task.csv")
# ------------------------------

# map.csv를 2차원 배열로 읽어와서 좌표 유효성 체크
def load_map_array(filepath):
    arr = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            arr.append([c.strip() for c in row])
    return arr

map_arr = load_map_array('data/map.csv')
height, width = len(map_arr), len(map_arr[0])
print(f"map shape: {height}x{width}")

# 좌표 범위 + cell 값 검증
def is_valid_coord(y, x, arr, allowed_symbols=('.', 'I', 'O')):
    h, w = len(arr), len(arr[0])
    if not (0 <= y < h and 0 <= x < w):
        return False
    if arr[y][x] not in allowed_symbols:
        return False
    return True
# assign_task.csv 저장 전, 좌표 전수검증
for idx, row in df.iterrows():
    sy, sx = int(row['start_y']), int(row['start_x'])
    gy, gx = int(row['goal_y']), int(row['goal_x'])

    if not is_valid_coord(sy, sx, map_arr):
        print(f"[경고] idx={idx} start ({sy},{sx}) 가 유효하지 않은 칸! map={map_arr[sy][sx]}")
    if not is_valid_coord(gy, gx, map_arr):
        print(f"[경고] idx={idx} goal ({gy},{gx}) 가 유효하지 않은 칸! map={map_arr[gy][gx]}")

arr = []
with open("data/map.csv", "r", encoding="utf-8") as f:
    for row in csv.reader(f):
        arr.append([c.strip() for c in row])
for y, row in enumerate(arr):
    print(f"{y:02d}: {''.join(row)}")