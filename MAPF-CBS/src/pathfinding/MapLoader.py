import pandas as pd
import csv

def load_map(csv_path):
    """
    1차 실험 : 장애물: '@', 이동가능: '.', 출발: 'I', 도착: 'O'
    """
    warehouse_map = []
    start = None
    goal = None
    rack_list = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            map_row = []
            for j, c in enumerate(row):
                c = c.strip()
                if c == '@':
                    map_row.append(1)
                else:
                    map_row.append(0)
                if c == 'I':
                    start = (i, j)
                elif c == 'O':
                    goal = (i, j)
                elif c.isalpha() and c not in ('I', 'O'):
                    rack_list.append({'loc': (i, j), 'rack_id': c})
            warehouse_map.append(map_row)
    return warehouse_map, start, goal, rack_list

# --- 출력 확인용 ---
# my_map, start, goal, rack_list = load_map("C:/Users/sj123/WMSProject/AI/MAPF-CBS/data/map.csv")
# print("맵:", my_map)
# print("출발:", start)
# print("도착:", goal)
# print("랙:", rack_list)