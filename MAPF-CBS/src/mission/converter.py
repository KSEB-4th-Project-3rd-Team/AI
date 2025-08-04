"""
Task DB의 rack_id, map.csv의 I/O를 활용해
start_y, start_x, goal_y, goal_x 좌표를 자동으로 채우는 변환 스크립트
(입고/출고/피킹 등 type별 분기도 지원)
"""
import csv
import pandas as pd

# ---- 1. 랙 id → (y, x) 좌표 매핑테이블 생성 (A~T, 001~012) ----
rack_letters = [chr(ord('A') + i) for i in range(20)]    # 'A'~'T'
rack_numbers = [f"{j:03d}" for j in range(1, 13)]        # '001'~'012'

rack_coords = {f"{row}{col}": (y, x)
               for x, row in enumerate(rack_letters)
               for y, col in enumerate(rack_numbers)
}

# ---- 2. map.csv에서 I(출발), O(도착) 좌표 추출 ----
def get_IO_from_map(filepath):
    start, goal = None, None
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            for j, c in enumerate(row):
                c = c.strip()
                if c == 'I':
                    start = (i, j)
                elif c == 'O':
                    goal = (i, j)
    return start, goal

start, goal = get_IO_from_map('data/map.csv')
print("Start:", start, "Goal:", goal)

# ---- 3. task.csv 불러오기 ----
df = pd.read_csv('data/missions.csv')

# ---- 4. type별 좌표 자동 변환 (INBOUND, OUTBOUND, PICK, MOVE) ----
for idx, row in df.iterrows():
    t = row['type']
    rack_y, rack_x = rack_coords.get(row['rack_id'], (None, None))
    if t in ('INBOUND', 'PICK'):
        # 입고/피킹: 출발(I), 도착(랙)
        df.at[idx, 'start_y'], df.at[idx, 'start_x'] = start
        df.at[idx, 'goal_y'], df.at[idx, 'goal_x'] = rack_y, rack_x
    elif t in ('OUTBOUND', 'DROP'):
        # 출고/하차: 출발(랙), 도착(O)
        df.at[idx, 'start_y'], df.at[idx, 'start_x'] = rack_y, rack_x
        df.at[idx, 'goal_y'], df.at[idx, 'goal_x'] = goal
    elif t == 'MOVE':
        # MOVE: (랜덤 랙 -> 랜덤 랙) 예시
        import random
        other_rack = random.choice(list(rack_coords.keys()))
        goal_y, goal_x = rack_coords.get(other_rack, (None, None))
        df.at[idx, 'start_y'], df.at[idx, 'start_x'] = rack_y, rack_x
        df.at[idx, 'goal_y'], df.at[idx, 'goal_x'] = goal_y, goal_x
    else:
        # 예외: 모두 I->O로 처리 (테스트용)
        df.at[idx, 'start_y'], df.at[idx, 'start_x'] = start
        df.at[idx, 'goal_y'], df.at[idx, 'goal_x'] = goal

# ---- 5. 저장 ----
df.to_csv('data/assign_task.csv', index=False)
print("좌표 치환 완료! -> data/assign_task.csv")
