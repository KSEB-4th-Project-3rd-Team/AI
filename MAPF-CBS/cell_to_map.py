"""
MAPF용 변환 기준
1 : 랙 : 장애물, 이동불가 : @

0 : 통로 : 이동 가능 : .

-1 : 출고존/입고존 : 시작점, 목표점 : S / G
"""
import numpy as np
import pandas as pd

csv_path = "WarehouseGrid.csv"
map_path = "warehouse_map.txt"

# CSV 불러오기 (구분자가 tab 또는 \t 혹은 \s+ 인 경우 sep 지정)
data = pd.read_csv(csv_path, sep='\t', header=None)

# 맵 변환 함수
def cell_to_char(cell):
    if cell == 0 or cell == 2:  # 이동 가능(통로, 입출고 등)
        return '.'
    else:                       # 벽/장애물/특수존
        return '@'

# 변환 실행
char_map = data.applymap(cell_to_char).values

# 맵 크기
rows, cols = char_map.shape

# 맵 파일 저장 (MAPF 오픈소스 포맷)
with open(map_path, 'w') as f:
    f.write(f"{rows} {cols}\n")
    for row in char_map:
        f.write(''.join(row) + '\n')

print(f"변환 완료 {map_path} 저장")
