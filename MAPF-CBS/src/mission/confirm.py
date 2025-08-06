# 맵 전체 읽어서 랙(@@@@)이 어디에 있는지 자동으로 출력
map_path = "data/map.csv"
rack_blocks = []

with open(map_path, "r", encoding="utf-8") as f:
    for y, line in enumerate(f):
        line = line.rstrip()
        x = 0
        while x < len(line):
            if line[x:x+4] == '@@@@':
                # 랙 블록의 왼쪽 위 꼭짓점 (y, x)
                rack_blocks.append((y, x))
                x += 4
            else:
                x += 1

# 랙 블록 위치 전수 출력
print(f"[INFO] 맵에서 추출된 랙(@@@@) 위치 개수: {len(rack_blocks)}")
for i, (y, x) in enumerate(rack_blocks):
    print(f"  [{i:03d}] 랙블록 (y={y}, x={x})")
