'''
다익스트라 기반 휴리스틱 : 
실제 맵상의 모든 장애물, 경로를 고려
목표에서 모든 위치까지 '실제 최단 거리'를 미리 계산(A*에서 휴리스틱 테이블로 활용)
맨해튼 기반 휴리스틱 : 
대부분의 A*경로탐색에서 기본값으로 사용 -> 맵이 단순하거나 적을 때
단점: 장애물로 인해 실제 경로가 훨씬 더 길 수도 있는데 이걸 반영하지 못함
=> 1차 실험 -> 맨해튼
최적화/성능 검증 후 다익스트라 기반 휴리스틱 추가
'''
import heapq
from itertools import product
import copy
import collections
import time as timer
import numpy as np
from .constraints import (
    is_constrained,
    violates_pos_constraint,
    future_constraint_exists,
    build_constraint_table,
)
# -------------- 1. 위치 이동-------------------
# 1. 위치 이동 함수 (상하좌우+정지)
def move(loc, dir):
    """좌표(loc)에서 방향(dir: 0~4)으로 이동한 좌표 반환"""
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]

# -------------- 2. 휴리스틱 함수-------------------

# 단일 에이전트 맨해튼 휴리스틱
def manhattan_heuristics(my_map, goal):
    """
    각 위치에서 goal까지의 맨해튼 거리를 휴리스틱 값으로 갖는 딕셔너리 반환
    """
    h_values = {}
    for i in range(len(my_map)):
        for j in range(len(my_map[0])):
            h = abs(goal[0] - i) + abs(goal[1] - j)
            h_values[(i, j)] = h
    return h_values

# 멀티 에이전트 맨해튼 휴리스틱
def multi_manhattan_heuristics(my_map, goals):
    """
    다중 agent의 각 goal에 대해 맨해튼 휴리스틱 테이블 생성
    - goals: {agent_id: (y, x), ...}
    - return: {agent_id: { (y, x): h, ... }, ...}
    """
    h_values = {}
    for agent, goal in goals.items():
        h_values[agent] = {}
        for i in range(len(my_map)):
            for j in range(len(my_map[0])):
                h = abs(goal[0] - i) + abs(goal[1] - j)
                h_values[agent][(i, j)] = h
    return h_values

# 다익스트라 휴리스틱
def Dijkstra_heuristics(my_map, goal):
    '''
    다익스트라로 목표지점에서부터 모든 위치까지 최소 비용 계산
    '''
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
    
# -------------- 3. 제약조건 관련 -------------------

# 시간별로 정리된 constraint 테이블 생성
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

# -------------- 4. 경로/위치 추적 및 복원 -------------------

# 경로에서 time에 따르는 위치 반환 
def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # 목표지점에서 대기

# goal_node까지 거슬러 올라가 경로복원(meta agent용 : 실글일때는 길이 1리스트/ 멀티일때는 길이 2이상 리스트로)
def get_path(goal_node, meta_agent):
    ''' 
    탐색 트리의 goal_node에서 root까지 역추적하여, 각 agent의 경로 반환
    - path: [ [loc0, loc1, ...], ... ]  (agent별)
    '''
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

# -------------- 5. 제약조건 위반 -------------------
# constraints.py
# from pathfinding.constraints import is_constrained, violates_pos_constraint, future_constraint_exists
# -------------- 6. open list(우선순위큐) 관련 함수 -------------------

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

# ----------------- 7. 경로탐색 함수 ----------------

#  단일 agent A*
def a_star_single(my_map, start, goal, h_values, agent, constraints):
    result = a_star_multi(my_map, [start], [goal], {agent: h_values}, [agent], constraints)
    if result and isinstance(result[0], list):
        return result[0]
    else:
        return result
    '''
    a_star_single 함수에서 a_star_multi를 호출할 때,
    h_values와 start_locs 리스트의 인덱스가 에이전트 ID와 일치하지 않으면 IndexError
    '''
# 다중 에이전트 A* (a_star_single → a_star_multi 호출)
'''
싱글이 멀티를 호출함
-> a_star_multi는 'N개의 에이전트'에 대해 경로탐색
-> a_star_single은 '단 1개의 에이전트'에 대해서만 경로탐색
a_star_multi를 'N=1'인 케이스로 바로 사용 가능
=> multi 함수로 구현 / wrapping 함수는 에이전트 1대 경로탐색만 혹시, 필요한 경우를 위해
'''
def a_star_multi(my_map, start_locs, goal_loc, h_values, meta_agent, constraints):
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

# -------------- 8. 경로 총 비용 계산 -------------------

# 전체 경로의 총 비용 계산 (길이의 합)
def get_sum_of_cost(paths):
    total = 0
    for path in paths:
        total += len(path) - 1
        if len(path) > 1:
            assert path[-1] != path[-2]
    return total
