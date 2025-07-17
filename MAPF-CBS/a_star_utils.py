import time as timer
import heapq
from itertools import product
import numpy as np
import copy

def move(loc, dir):
    # 상하좌우 및 제자리 이동 (0, 1, 2, 3, 4)
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]

def get_sum_of_cost(paths):
    # 전체 경로의 총 cost(이동 칸 수)를 계산
    rst = 0
    for path in paths:
        rst += len(path) - 1
        if(len(path) > 1):
            assert path[-1] != path[-2]
    return rst

def compute_heuristics(my_map, goal):
    # 목표지점(goal)에서 각 위치까지의 최단거리(휴리스틱) 계산 (다익스트라)
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):  # 상하좌우
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))
    # 휴리스틱 테이블 생성
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values

def get_location(path, time):
    # 특정 time에 위치 반환 (goal에 도달하면 대기)
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]

class A_Star(object):
    def __init__(self, my_map, starts, goals, heuristics, agents, constraints):
        """
        my_map   - 맵 배열(장애물 위치)
        starts   - 각 agent의 시작 위치 리스트
        goals    - 각 agent의 목표 위치 리스트
        heuristics - 각 agent별 휴리스틱 값
        agents   - 경로 찾는 대상 agent 또는 meta-agent
        constraints - 제약조건 리스트 (CBS에서 생성)
        """
        self.my_map = my_map
        self.num_generated = 0
        self.num_expanded = 0
        self.CPU_time = 0
        self.open_list = []
        self.closed_list = dict()
        self.constraints = constraints # 제약조건
        self.agents = agents
        if not isinstance(agents, list):
            self.agents = [agents]
            for c in self.constraints:
                c['meta_agent'] = {c['agent']}
        self.starts = [starts[a] for a in self.agents]
        self.heuristics = [heuristics[a] for a in self.agents]
        self.goals = [goals[a] for a in self.agents]
        self.c_table = [] # constraint table
        self.max_constraints = np.zeros((len(self.agents),), dtype=int)

    def push_node(self, node):
        # 노드를 open_list(heap)에 삽입 (우선순위: f=g+h, h, 위치, 생성순)
        f_value = node['g_val'] + node['h_val']
        heapq.heappush(self.open_list, (f_value, node['h_val'], node['loc'], self.num_generated, node))
        self.num_generated += 1

    def pop_node(self):
        # open_list에서 우선순위 노드 pop
        _,_,_, id, curr = heapq.heappop(self.open_list)
        self.num_expanded += 1
        return curr

    def build_constraint_table(self, agent):
        # 특정 agent에 대한 constraint 테이블 생성
        constraint_table = dict()
        if not self.constraints:
            return constraint_table
        for constraint in self.constraints:
            timestep = constraint['timestep']
            t_constraint = []
            if timestep in constraint_table:
                t_constraint = constraint_table[timestep]
            # positive, negative constraint 구분 및 적용
            if constraint['positive'] and constraint['agent'] == agent:
                t_constraint.append(constraint)
                constraint_table[timestep] = t_constraint
            elif not constraint['positive'] and constraint['agent'] == agent:
                t_constraint.append(constraint)
                constraint_table[timestep] = t_constraint
            elif constraint['positive']:
                neg_constraint = copy.deepcopy(constraint)
                neg_constraint['agent'] = agent
                # edge collision이면 방향 반대로
                if len(constraint['loc']) == 2:
                    prev_loc = constraint['loc'][1]
                    curr_loc = constraint['loc'][0]
                    neg_constraint['loc'] = [prev_loc, curr_loc]
                neg_constraint['positive'] = False
                t_constraint.append(neg_constraint)
                constraint_table[timestep] = t_constraint
        return constraint_table

    def constraint_violated(self, curr_loc, next_loc, timestep, c_table_agent, agent):
        # 현재/다음 위치, 타임스텝에서 제약조건 위반 여부 체크
        if timestep not in c_table_agent:
            return None
        for constraint in c_table_agent[timestep]:
            if agent == constraint['agent']:
                if len(constraint['loc']) == 1:
                    # vertex(위치) constraint
                    if constraint['positive'] and next_loc != constraint['loc'][0]:
                        return constraint
                    elif not constraint['positive'] and next_loc == constraint['loc'][0]:
                        return constraint
                else:
                    # edge(이동경로) constraint
                    if constraint['positive'] and constraint['loc'] != [curr_loc, next_loc]:
                        return constraint
                    if not constraint['positive'] and constraint['loc'] == [curr_loc, next_loc]:
                        return constraint
        return None

    def future_constraint_violated(self, curr_loc, timestep, max_timestep, c_table_agent, agent):
        # goal에 도착한 뒤 이후 timestep에도 제약 위반이 있는지 체크
        for t in range(timestep+1, max_timestep+1):
            if t not in c_table_agent:
                continue
            for constraint in c_table_agent[t]:
                if agent == constraint['agent']:
                    if len(constraint['loc']) == 1:
                        if constraint['positive'] and curr_loc != constraint['loc'][0]:
                            return True
                        elif not constraint['positive'] and curr_loc == constraint['loc'][0]:
                            return True
        return False

    def generate_child_nodes(self, curr):
        # 현재 상태에서 다음 가능한 모든 행동(자식 노드) 생성
        children = []
        ma_dirs = product(list(range(5)), repeat=len(self.agents)) # 각 agent의 이동방향 조합
        for dirs in ma_dirs:
            invalid_move = False
            child_loc = []
            for i, a in enumerate(self.agents):           
                aloc = move(curr['loc'][i], dirs[i])
                # 위치 충돌 체크 (internal conflict)
                if aloc in child_loc:
                    invalid_move = True
                    break
                child_loc.append(aloc)
            if invalid_move:
                continue
            # edge collision(자리 맞바꿈) 체크
            for i, a in enumerate(self.agents):   
                for j, a in enumerate(self.agents):   
                    if i != j:
                        if child_loc[i] == curr['loc'][j] and child_loc[j] == curr['loc'][i]:
                            invalid_move = True
            if invalid_move:
                continue
            # 맵 범위, 장애물, 외부 제약조건 체크
            for i, a in enumerate(self.agents):  
                next_loc= child_loc[i]
                if next_loc[0]<0 or next_loc[0]>=len(self.my_map) or next_loc[1]<0 or next_loc[1]>=len(self.my_map[0]):
                    invalid_move = True
                elif self.my_map[next_loc[0]][next_loc[1]]:
                    invalid_move = True
                elif self.constraint_violated(curr['loc'][i],next_loc,curr['timestep']+1,self.c_table[i], self.agents[i]):
                    invalid_move = True
                if invalid_move:
                    break
            if invalid_move:
                continue
            # 휴리스틱/코스트 계산
            h_value = sum([self.heuristics[i][child_loc[i]] for i in range(len(self.agents))])
            num_moves = curr['reached_goal'].count(False)
            g_value = curr['g_val'] + num_moves
            reached_goal = [False for i in range(len(self.agents))]
            for i, a in enumerate(self.agents):
                if not reached_goal[i] and child_loc[i] == self.goals[i]:
                    if curr['timestep']+1 <= self.max_constraints[i]:
                        if not self.future_constraint_violated(child_loc[i], curr['timestep']+1, self.max_constraints[i], self.c_table[i], self.agents[i]):
                            reached_goal[i] = True
                    else:
                        reached_goal[i] = True
            child = {'loc': child_loc,
                    'g_val': g_value,
                    'h_val': h_value,
                    'parent': curr,
                    'timestep': curr['timestep']+1,
                    'reached_goal': copy.deepcopy(reached_goal)
                    } 
            children.append(child)
        return children

    def compare_nodes(self, n1, n2):
        # 두 노드의 f=g+h 값을 비교 (더 작으면 우선)
        assert isinstance(n1['g_val'] + n1['h_val'], int)
        assert isinstance(n2['g_val'] + n2['h_val'], int)
        return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

    def find_paths(self):
        # 실제 경로 탐색 실행부 (A* search)
        self.start_time = timer.time()
        print("> build constraint table")
        for i, a in enumerate(self.agents):
            table_i = self.build_constraint_table(a)
            print(table_i)
            self.c_table.append(table_i)
            if table_i.keys():
                self.max_constraints[i] = max(table_i.keys())
        h_value = sum([self.heuristics[i][self.starts[i]] for i in range(len(self.agents))])
        root = {'loc': [self.starts[j] for j in range(len(self.agents))],
                'g_val': 0, 
                'h_val': h_value, 
                'parent': None,
                'timestep': 0,
                'reached_goal': [False for i in range(len(self.agents))]
                }
        for i, a in enumerate(self.agents):
            if root['loc'][i] == self.goals[i]:
                if root['timestep'] <= self.max_constraints[i]:
                    if not self.future_constraint_violated(root['loc'][i], root['timestep'], self.max_constraints[i], self.c_table[i], self.agents[i]):
                        root['reached_goal'][i] = True
                        self.max_constraints[i] = 0
        self.push_node(root)
        self.closed_list[(tuple(root['loc']),root['timestep'])] = [root]
        while len(self.open_list) > 0:
            curr = self.pop_node()
            solution_found = all(curr['reached_goal'][i] for i in range(len(self.agents)))
            if solution_found:
                return get_path(curr, self.agents)
            children = self.generate_child_nodes(curr)
            for child in children:
                f_value = child['g_val'] + child['h_val']
                if (tuple(child['loc']), child['timestep']) in self.closed_list:
                    existing = self.closed_list[(tuple(child['loc']), child['timestep'])]
                    if (child['g_val'] + child['h_val'] < existing['g_val'] + existing['h_val']) and (child['g_val'] < existing['g_val']) and child['reached_goal'].count(False) <= existing['reached_goal'].count(False):
                        print("child is better than existing in closed list")
                        self.closed_list[(tuple(child['loc']), child['timestep'])] = child
                        self.push_node(child)
                else:
                    self.closed_list[(tuple(child['loc']), child['timestep'])] = child
                    self.push_node(child)
        print('no solution')
        return None
