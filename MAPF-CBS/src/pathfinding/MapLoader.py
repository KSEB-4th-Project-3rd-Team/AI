import pandas as pd

def load_map(csv_path):
    """
    CSV 파일을 2차원 0/1 리스트로 변환
    (0=free, 1=obstacle)
    """
    df = pd.read_csv("data/csv_path", header=None)
    warehouse_map = df.values.astype(int).tolist()
    return warehouse_map

def load_map_txt(txt_path):
    """
    심볼 기반 맵 파일을 2D 0/1 맵 + 출발/도착 좌표로 변환
    1차 실험 : 장애물: '@', 이동가능: '.', 출발: 'I', 도착: 'O'
    """
    warehouse_map = []
    start = None
    goal = None
    rack_list = []
    with open(txt_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            row = []
            for j, c in enumerate(line):
                if c == '@':
                    row.append(1)
                else:
                    row.append(0)
                if c == 'I':
                    start = (i, j)
                elif c == 'O':
                    goal = (i, j)
                elif c.isalpha() and c not in ('I', 'O'):
                    rack_list.append({'loc': (i, j), 'rack_id': c})
            warehouse_map.append(row)
    return warehouse_map, start, goal, rack_list
