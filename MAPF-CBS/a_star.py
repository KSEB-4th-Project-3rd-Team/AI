import heapq
from itertools import product
import copy
import collections

# 좌표와 방향을 받아 이동한 위치 반환 (0:상, 1:우, 2:하, 3:좌, 4:제자리)
def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]

# 전체 경로의 총 비용 계산 (길이의 합)
def get_sum_of_cost(paths):
    total = 0
    for path in paths:
        total += len(path) - 1
        if len(path) > 1:
            assert path[-1] != path[-2]
    return total

# 다익스트라로 목표지점에서부터 모든 위치까지 최소 비용 계산
def compute_heuristics(my_map, goal):
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while open_list:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            # 맵 경계 체크
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) or \
               child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
                continue
            # 장애물 체크
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                if closed_list[child_loc]['cost'] > child_cost:
                    closed_list[child_loc] = child
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # 휴리스틱 값 테이블 생성 (각 위치별 최소 비용)
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values

# 제약 리스트(Constraint)에서 시간별로 정리된 테이블 생성
def build_constraint_table(constraints, meta_agent):
    constraint_table = collections.defaultdict(list)
    if not constraints:
        return constraint_table
    for constraint in constraints:
        timestep = constraint['timestep']
        for agent in meta_agent:
            # 해당 에이전트의 제약이면 그대로 추가
            if (constraint['agent'] == agent):
                constraint_table[timestep].append(constraint)
            # 다른 에이전트의 양성 제약이면 음성 제약으로 변환해 추가
            elif constraint['positive']:
                neg_constraint = copy.deepcopy(constraint)
                neg_constraint['agent'] = agent
                neg_constraint['meta_agent'] = meta_agent
                if len(constraint['loc']) == 2:
                    prev_loc = constraint['loc'][1]
                    curr_loc = constraint['loc'][0]
                    neg_constraint['loc'] = [prev_loc, curr_loc]
                neg_constraint['positive'] = False
                constraint_table[timestep].append(neg_constraint)
    return constraint_table

# 경로에서 주어진 시간에 해당하는 위치 반환
def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # 목표지점에서 대기

# goal_node까지 거슬러 올라가 경로를 복원 (meta agent용)
def get_path(goal_node, meta_agent):
    path = [[] for _ in range(len(meta_agent))]
    curr = goal_node
    while curr is not None:
        for i in range(len(meta_agent)):
            path[i].append(curr['loc'][i])
        curr = curr['parent']
    for i in range(len(meta_agent)):
        path[i].reverse()
        # 경로 끝부분 중복 제거
        while len(path[i]) > 1 and path[i][-1] == path[i][-2]:
            path[i].pop()
    return path

# 해당 시간에 주어진 위치/이동이 외부 제약에 걸리는지 체크
def is_constrained(curr_loc, next_loc, timestep, constraint_table, agent):
    if timestep not in constraint_table:
        return False
    for constraint in constraint_table[timestep]:
        if agent == constraint['agent'] and not constraint['positive']:
            # 위치 제약
            if len(constraint['loc']) == 1:
                if next_loc == constraint['loc'][0]:
                    return True
            # 이동 제약(엣지)
            else:
                if constraint['loc'] == [curr_loc, next_loc]:
                    return True
    return False

# 양성(필수) 제약을 어기는지 체크
def violates_pos_constraint(curr_loc, next_loc, timestep, constraint_table, agent, meta_agent):
    if timestep not in constraint_table:
        return False
    for constraint in constraint_table[timestep]:
        if agent == constraint['agent'] and constraint['positive']:
            if len(constraint['loc']) == 1:
                if next_loc != constraint['loc'][0]:
                    return True
            else:
                if constraint['loc'] != [curr_loc, next_loc]:
                    return True
    return False

# 미래에 해당 위치에 외부 제약이 걸릴 예정인지 체크
def future_constraint_exists(agent, meta_agent, agent_loc, timestep, constraint_table):
    for t in constraint_table:
        if t > timestep:
            for constraint in constraint_table[t]:
                if constraint['loc'][-1] == agent_loc:
                    if agent == constraint['agent'] and not constraint['positive']:
                        return True
                    if agent != constraint['agent'] and constraint['positive']:
                        return True
    return False

# open_list에 노드 추가 (우선순위큐)
def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))

# open_list에서 최상위 노드 꺼내기
def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr

# 노드 비교 (f 값 기준)
def compare_nodes(n1, n2):
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

# 다중 에이전트 A* (CBS/MA-CBS에서 사용)
def a_star(my_map, start_locs, goal_loc, h_values, meta_agent, constraints):
    """
    my_map      - 장애물(1)/이동가능(0) 맵 (2차원 배열)
    start_locs  - 각 에이전트 시작 위치 (리스트)
    goal_loc    - 각 에이전트 목표 위치 (리스트)
    h_values    - 각 에이전트별 휴리스틱 값 (dict)
    meta_agent  - 현재 탐색하는 에이전트 집합
    constraints - 제약조건 리스트 (딕셔너리들)
    """

    open_list = []
    closed_list = dict()
    h_value = 0

    # meta_agent가 단일 값이면 리스트로 변환
    if not isinstance(meta_agent, list):
        meta_agent = [meta_agent]
        for c in constraints:
            c['meta_agent'] = {c['agent']}
    ma_length = len(meta_agent)

    table = build_constraint_table(constraints, meta_agent)

    # h_value(휴리스틱) 계산 (multi-agent면 합산)
    for agent in meta_agent:
        h_value += h_values[agent][start_locs[agent]]

    root = {
        'loc': [start_locs[a] for a in meta_agent],
        'g_val': 0,
        'h_val': h_value,
        'parent': None,
        'timestep': 0,
        'reached_goal': [False for _ in range(len(meta_agent))]
    }

    push_node(open_list, root)
    closed_list[(tuple(root['loc']), root['timestep'])] = root

    while open_list:
        curr = pop_node(open_list)

        # 각 에이전트가 목표 위치에 도달했는지 체크
        for a in range(ma_length):
            if curr['loc'][a] == goal_loc[meta_agent[a]]:
                # 미래에 해당 위치에 제약이 있는지도 확인
                if not future_constraint_exists(meta_agent[a], meta_agent, curr['loc'][a], curr['timestep'], table):
                    curr['reached_goal'][a] = True

        # 모든 에이전트가 목표에 도달하면 경로 반환
        if all(curr['reached_goal']):
            return get_path(curr, meta_agent)

        # 도달 못한 에이전트만 남겨서 진행
        seeking_ma = copy.deepcopy(meta_agent)
        num_a_path_complete = 0
        for i, a in enumerate(meta_agent):
            if curr['reached_goal'][i]:
                seeking_ma.remove(a)
                num_a_path_complete += 1

        s_ma_length = len(seeking_ma)
        assert s_ma_length == ma_length - num_a_path_complete

        # 남은 에이전트의 이동방향 조합 생성
        ma_dirs_list = [list(range(5)) for _ in range(s_ma_length)]
        ma_dirs = product(*ma_dirs_list)

        for dirs in ma_dirs:
            invalid_move = False
            child_loc = copy.deepcopy(curr['loc'])
            # 각 에이전트 이동 적용 및 내부충돌 체크
            for a in range(ma_length):
                if curr['reached_goal'][a]:
                    continue
                else:
                    agent = meta_agent[a]
                    i_dir = seeking_ma.index(agent)
                    aloc = move(curr['loc'][a], dirs[i_dir])
                    # vertex 충돌(동일 위치)
                    if aloc in child_loc:
                        invalid_move = True
                        break
                    child_loc[a] = aloc

            if invalid_move:
                continue

            # edge 충돌(자리 맞바꿈)
            for ai in range(ma_length):
                for aj in range(ma_length):
                    if ai != aj:
                        if child_loc[ai] == curr['loc'][aj] and child_loc[aj] == curr['loc'][ai]:
                            invalid_move = True
            if invalid_move:
                continue

            # 맵 경계, 장애물, 외부 제약 체크
            for i in range(len(child_loc)):
                loc = child_loc[i]
                if loc[0] < 0 or loc[0] >= len(my_map) or loc[1] < 0 or loc[1] >= len(my_map[0]):
                    invalid_move = True
                    break
                if my_map[loc[0]][loc[1]]:
                    invalid_move = True
                    break
                if is_constrained(curr['loc'][i], loc, curr['timestep'] + 1, table, meta_agent[i]):
                    invalid_move = True
                    break
                if violates_pos_constraint(curr['loc'][i], loc, curr['timestep'] + 1, table, meta_agent[i], meta_agent):
                    invalid_move = True
                    break
            if invalid_move:
                continue

            # 다음 상태의 h_value 계산 (multi-agent 합산)
            h_value = 0
            for i in range(ma_length):
                h_value += h_values[meta_agent[i]][child_loc[i]]

            child = {
                'loc': child_loc,
                'g_val': curr['g_val'] + s_ma_length,  # 이동한 에이전트 수 만큼 비용 증가
                'h_val': h_value,
                'parent': curr,
                'timestep': curr['timestep'] + 1,
                'reached_goal': [False for _ in range(len(meta_agent))]
            }

            key = (tuple(child['loc']), child['timestep'])
            if key in closed_list:
                if compare_nodes(child, closed_list[key]):
                    closed_list[key] = child
                    push_node(open_list, child)
            else:
                closed_list[key] = child
                push_node(open_list, child)
    print('해결 불가')
    return None