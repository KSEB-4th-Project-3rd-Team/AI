import os
import numpy as np

BASE_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_dir = os.path.join(BASE_dir, "map")
os.makedirs(SAVE_dir, exist_ok=True)
SAVE_path = os.path.join(SAVE_dir, "realsizeCalmap.txt")


# 실제 창고 크기 (mm)
real_width = 510_000
real_height = 400_000

# 그리드 수
cols, rows = 102, 80
cell_w = real_width / cols   # 1셀 가로(mm)
cell_h = real_height / rows  # 1셀 세로(mm)

print(f"셀 크기: 가로 {cell_w:.1f} mm, 세로 {cell_h:.1f} mm")
print(f"셀 크기: 가로 {cell_w/1000:.2f} m, 세로 {cell_h/1000:.2f} m\n")
# 셀 크기: 가로 5000.0 mm, 세로 5000.0 mm
# 셀 크기: 가로 5.00 m, 세로 5.00 m
# 셀 하나당 5m*5m 로 비율축소  -> 유니티에서 셀 하나당 0.5로 계산

"""
실제 창고 크기(510m x 400m) : cols=102, rows=80 : 셀 크기=5m

"""

# 실제 mm좌표 → 맵 좌표 변환 함수
def mm_to_cell(x_mm, y_mm):
    col = int(round(x_mm / cell_w))
    row = int(round(y_mm / cell_h))
    return row, col

rows, cols = 80, 102
warehouse = np.full((rows, cols), '.', dtype=str)

# 랙 
total_racks = 16
rack_width = 2
rack_gap = 5
rack_height = 54     # 예: row=4~59까지
rack_start_row = 4
rack_zone_width = total_racks * rack_width + (total_racks-1) * rack_gap
rack_start_col = 4

for i in range(total_racks):
    col = rack_start_col + i * (rack_width + rack_gap)
    warehouse[rack_start_row:rack_start_row + rack_height, col:col + rack_width] = '@'

# 관리소
returns_rows = 10
returns_cols = 18
returns_row_end = rows - 1         # 맨 하단 벽 위 한줄까지
returns_row_start = returns_row_end - returns_rows
returns_col_start = 1
returns_col_end = returns_col_start + returns_cols

warehouse[returns_row_start:returns_row_end, returns_col_start:returns_col_end] = 'R'

print(f"관리소 영역: 행 {returns_row_start}~{returns_row_end-1}, 열 {returns_col_start}~{returns_col_end-1}")
# 관리소 영역: 행 69~78, 열 1~18

# 인바운드 - 관리소 오른쪽 하단, 가로 5개씩 2줄, 띄엄띄엄
# 인바운드 박스 파라미터
inbound_box_w = 4 # 가로
inbound_box_h = 10 # 길이
inbound_cols = 8  # 줄 개수

inbound_row = rows - inbound_box_h - 7  # 관리소랑 안 겹치게 하단에서부터 올라감
inbound_col_start = returns_col_end + 8  # 관리소 오른쪽 띄우고 시작

for col_offset in range(inbound_cols):
    box_col = inbound_col_start + col_offset * (inbound_box_w + 5)  # gap=3
    warehouse[inbound_row:inbound_row+inbound_box_h, box_col:box_col+inbound_box_w] = 'I'

# 벽(경계)
warehouse[0, :] = '@'
warehouse[-1, :] = '@'
warehouse[:, 0] = '@'
warehouse[:, -1] = '@'

# 저장
BASE_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_dir = os.path.join(BASE_dir, "map")
os.makedirs(SAVE_dir, exist_ok=True)
SAVE_path = os.path.join(SAVE_dir, "map_102x80.txt")
with open(SAVE_path, 'w', encoding='utf-8') as f:
    f.write(f"{rows} {cols}\n")
    for row in warehouse:
        f.write(''.join(row) + '\n')

print(f"\n맵 텍스트 저장 완료: {SAVE_path}")