'''
맵/DB/Task 로더, 에이전트 초기상태 및 시작/목표 지정 시나리오 함수 
'''

def single_agent_scenario(map_file="data/map.txt"):
    """
    맵 파일에서 출발/도착 위치 자동 추출(1대용)
    """
    from map_loader import load_map_txt
    my_map, start, goal, rack_list = load_map_txt(map_file)
    # 중간 랙 미사용 시 rack_list는 []
    return {
        "my_map": my_map,
        "agents": [
            {
                "start": start,
                "goal": goal,
                "racks": rack_list  
            }
        ]
    }


def multi_agent_scenario(map_file="data/map.txt"):
    """
    여긴 출발/목표/중간 랙이 모두 맵에 정의되어 있다고 가정
    (이름 or 위치를 수동지정 or 맵 파싱 결과 사용)
    """
    from map_loader import load_map_txt
    my_map, start, goal, rack_list = load_map_txt(map_file)

    # 여러 agent 위치가 맵에 여러 개 있을 경우, 자동 추출 예시:
    starts, goals = [], []
    rack_coords = []
    for rack in rack_list:
        # rack['rack_id']에 따라 중간 목표 할당 가능
        rack_coords.append(rack['loc'])

    # 맵에서 여러 I, O가 있을 때 자동으로 찾아주는 파서로 확장 가능
    # 여기서는 일단 단일 출발/도착
    starts.append(start)
    goals.append(goal)

    return {
        "my_map": my_map,
        "agents": [
            {
                "start": s,
                "goal": g,
                "racks": rack_coords   # 필요시
            }
            for s, g in zip(starts, goals)
        ]
    }

def get_scenario(map_file="data/map.txt", multi_agent=False):
    """
    맵 파일에서 에이전트 출발/도착/중간 경유지 정보 자동 파싱 및 구조화
    multi_agent: 다중 에이전트용
    """
    # 내부적으로 위 함수 중 선택
    if not multi_agent:
        return get_single_agent_scenario(map_file)
    else:
        return get_multi_agent_scenario(map_file)

# scenario = get_scenario("data/map.txt", multi_agent=False)
# my_map = scenario['my_map']
# agents = scenario['agents']  # [ {start:(y,x), goal:(y,x), racks:[(y,x), ...]} ]
