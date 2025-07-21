"""
MAPF용 변환 기준
1 : 랙 : 장애물, 이동불가 : @

0 : 통로 : 이동 가능 : .

-1 : 출고존/입고존 : 시작점, 목표점 : S / G
"""
import os
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "data", "WarehouseGrid.csv")
save_dir = os.path.join(BASE_DIR, "data")
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "map.txt")

data = pd.read_csv(csv_path, sep=',', header=None, encoding='utf-8')

# 맵 변환 함수
def cell_to_char(cell):
    if cell == 0 or cell == 2:  # 이동 가능(통로, 입출고 등)
        return '.'
    else:                       # 벽/장애물/특수존
        return '@'

# 변환 실행
char_map = data.applymap(cell_to_char).values

# 맵 크기
#char_map = data.applymap(cell_to_char).values
rows, cols = char_map.shape

# 저장
with open(save_path, 'w', encoding='utf-8') as f:
    f.write(f"{rows} {cols}\n")
    for row in char_map:
        f.write(''.join(row) + '\n')

print(f"변환 완료 : {save_path}")