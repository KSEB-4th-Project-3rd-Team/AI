from pathfinding.MapLoader import load_map

if __name__ == "__main__":
    map_path = "data/map.csv"   # 실제 파일 위치에 맞게 수정
    
    my_map, start, goal, rack_list = load_map_txt(map_path)
    print("맵:")
    for row in my_map:
        print(row)
    print("출발 위치:", start)
    print("도착 위치:", goal)
    print("랙(리스트):", rack_list)