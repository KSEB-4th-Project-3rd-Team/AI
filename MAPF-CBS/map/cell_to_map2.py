"""
MAPF용 변환 기준
1 : 랙 : 장애물, 이동불가 : @

0 : 통로 : 이동 가능 : .

-1 : 출고존/입고존 : 시작점, 목표점 : S / G
"""
import os
import numpy as np

BASE_dir = os.path.dirname(os.path.abspath(__file__))           
SAVE_dir = os.path.join(BASE_dir, "map")                        
os.makedirs(SAVE_dir, exist_ok=True)
SAVE_path = os.path.join(SAVE_dir, "map_102x80.txt")  


rows, cols = 80, 102
warehouse = np.full((rows, cols), '.', dtype=str)

total_racks = 16
rack_width = 2
rack_gap = 5
rack_height = 50
rack_start_row = 10

rack_zone_width = total_racks * rack_width + (total_racks-1) * rack_gap
rack_start_col = (cols - rack_zone_width) // 2

for i in range(total_racks):
    col = rack_start_col + i * (rack_width + rack_gap)
    warehouse[rack_start_row:rack_start_row + rack_height, col] = '@'

# 벽
warehouse[0, :] = '@'
warehouse[-1, :] = '@'
warehouse[:, 0] = '@'
warehouse[:, -1] = '@'

# 결과 프린트 (상위 25줄만)
for row in warehouse[:25]:
    print(''.join(row))
