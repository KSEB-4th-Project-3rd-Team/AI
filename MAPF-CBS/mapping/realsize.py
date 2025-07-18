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

# 실제 mm좌표 → 맵 좌표 변환 함수
def mm_to_cell(x_mm, y_mm):
    col = int(round(x_mm / cell_w))
    row = int(round(y_mm / cell_h))
    return row, col

# 텍스트맵 배열 초기화 (통로=.)
warehouse = np.full((rows, cols), '.', dtype=str)

# 랙 파라미터
total_racks = 16
rack_width = 2         # 랙 가로(셀)
rack_gap = 5           # 랙간 통로(셀)
rack_height = 50       # 랙 세로(셀)
rack_start_row = 10    # 랙 시작 행

rack_zone_width = total_racks * rack_width + (total_racks-1) * rack_gap
rack_start_col = (cols - rack_zone_width) // 2

# 랙 배치
for i in range(total_racks):
    col = rack_start_col + i * (rack_width + rack_gap)
    warehouse[rack_start_row:rack_start_row + rack_height, col:col + rack_width] = '@'

# 벽(경계)
warehouse[0, :] = '@'
warehouse[-1, :] = '@'
warehouse[:, 0] = '@'
warehouse[:, -1] = '@'

# 저장
with open(SAVE_path, 'w', encoding='utf-8') as f:
    f.write(f"{rows} {cols}\n")
    for row in warehouse:
        f.write(''.join(row) + '\n')

print(f"맵 텍스트 저장: {SAVE_path}")