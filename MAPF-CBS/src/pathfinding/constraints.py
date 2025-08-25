# import copy
# import collections
# from pathfinding.Astar import a_star_multi, manhattan_heuristics
# from constraints import (
#     build_constraint_table, 
#     is_constrained, 
#     violates_pos_constraint, 
#     future_constraint_exists, 
#     make_constraints
# )

# 맵, 에이전트 위치 지정
# my_map = ...  # 2D array
# start_locs = [(2,2), (4,5)]
# goal_locs = [(10,10), (7,3)]
# meta_agent = [0, 1]

# 다중 휴리스틱 세팅 
# goals_dict = {0: goal_locs[0], 1: goal_locs[1]}
# h_values = multi_manhattan_heuristics(my_map, goals_dict)

# 제약조건 리스트 구조를 에이전트/좌표/시간/positive로 맞춰둠
# constraints = [
#     {"agent": 0, "loc": [(y, x)], "timestep": t, "positive": False},
#     # 필요시 엣지 제약도 추가 가능
#     {"agent": 1, "loc": [(y1, x1), (y2, x2)], "timestep": t, "positive": False}
# ]

# # 경로 탐색
# paths = a_star_multi(my_map, start_locs, goal_locs, h_values, meta_agent, constraints)
# print(paths)

# --- 제약조건 생성기 함수 ---
def make_constraints(meta_agents, options=None):
    constraints = []
    # 필요하면 각 에이전트, 시간, 위치별로 생성
    # (초기에는 return constraints)
    return constraints

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

# 필수 제약을 어기는지 체크
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

# ---  제약조건 테이블 변환 ---
def build_constraint_table(constraints, meta_agent):
    """
    constraints 리스트를 시간별 defaultdict(list)로 변환
    """
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


