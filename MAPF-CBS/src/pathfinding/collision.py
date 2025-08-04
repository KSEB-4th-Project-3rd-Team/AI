# --- CBS에서 경로간 충돌 판별 ---
def detect_collision(path1, path2):
    '''
    두 agent의 경로(좌표 시퀀스)를 받아, 가장 먼저 발생하는 충돌(동일 위치/동일 엣지 swap)을 탐지해
    [충돌 위치], timestep으로 반환
    vertex collision: 같은 위치
    edge collision: 두 agent가 서로 자리 바꿈(스왑)
    경로 길이 다를 수 있음, get_location이 길이 넘어가면 마지막 좌표 반복 반환 구조 활용)
    '''
    t_range = max(len(path1),len(path2))
    for t in range(t_range):
        loc_c1 =get_location(path1,t)
        loc_c2 = get_location(path2,t)
        loc1 = get_location(path1,t+1)
        loc2 = get_location(path2,t+1)
        # vertex collision
        if loc1 == loc2:
            return [loc1],t
        # edge collision
        if[loc_c1,loc1] ==[loc2,loc_c2]:
            return [loc2,loc_c2],t
        
    return None

# --- paths에 충돌이 남아있는지 확인 ---
def detect_collisions(paths):
    '''
    여러 agent의 path 리스트를 받아, 모든 쌍에 대해 최초 충돌(위치+시간)전부 탐지
    딕셔너리 list로 반환: {a1, a2, loc, timestep}
    a1: agent1 id
    a2: agent2 id
    loc: 충돌 위치([y,x] or [edge])
    timestep: 충돌 발생 시간
    '''
    collisions =[]
    for i in range(len(paths)-1):
        for j in range(i+1,len(paths)):
            if detect_collision(paths[i],paths[j]) !=None:
                position,t = detect_collision(paths[i],paths[j])
                collisions.append({'a1':i,
                                'a2':j,
                                'loc':position,
                                'timestep':t+1})
    return collisions

# --- CBS의 기본 constraint 분기 (각 충돌마다, agent1/agent2 각각에게 해당 위치/엣지 금지 constraint 추가) ---
def standard_splitting(collision):
    '''
    (vertex) 충돌 시 양쪽 agent에 각각 vertex constraint 부여
    (edge) 충돌 시 양쪽 모두 엣지 constraint 부여
    constraint는 dict로: {"agent": a, "loc": loc, "timestep": t, "positive": False}
    '''
    constraints = []
    if len(collision['loc'])==1:
        constraints.append({'agent':collision['a1'],
                            'loc':collision['loc'],
                            'timestep':collision['timestep'],
                            'positive':False
                            })
        constraints.append({'agent':collision['a2'],
                            'loc':collision['loc'],
                            'timestep':collision['timestep'],
                            'positive':False
                            })
    else:
        constraints.append({'agent':collision['a1'],
                            'loc':[collision['loc'][0],collision['loc'][1]],
                            'timestep':collision['timestep'],
                            'positive':False
                            })
        constraints.append({'agent':collision['a2'],
                            'loc':[collision['loc'][1],collision['loc'][0]],
                            'timestep':collision['timestep'],
                            'positive':False
                            })
    return constraints

# --- 고수준 CBS 알고리즘 -- 
class CBSSolver(object):
    '''
    MAPF(CBS)의 메인루프
    agent별 경로 찾기, 충돌 감지, 제약 branch 생성, 재탐색 등 전체 흐름
    find_solution에서 splitting 함수(standard/disjoint/type-priority 등) 선택해서 분기
    open_list에 노드(push/pop), 각 노드는 paths/constraints/collisions를 가짐
    '''
    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node


    def find_solution(self, disjoint):
        """ Finds paths for all agents from their start locations to their goal locations
        disjoint         - use disjoint splitting or not
        """
        self.start_time = timer.time()
        
        if disjoint:
            splitter = disjoint_splitting
        else:
            splitter = standard_splitting

        print("USING: ", splitter)

        AStar = A_Star
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}

        for i in range(self.num_of_agents):  # Find initial path for each agent
            astar = AStar(self.my_map, self.starts, self.goals, self.heuristics,i, root['constraints'])
            path = astar.find_paths()

            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path[0])

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)
        
        while len(self.open_list) > 0:

            p = self.pop_node()
            if p['collisions'] == []:
                self.print_results(p)
                for pa in p['paths']:
                    print(pa)
                return p['paths'], self.num_of_generated, self.num_of_expanded # number of nodes generated/expanded for comparing implementations
            collision = p['collisions'].pop(0)
            # constraints = standard_splitting(collision)
            # constraints = disjoint_splitting(collision)
            constraints = splitter(collision)

            for constraint in constraints:
                q = {'cost':0,
                    'constraints': [constraint],
                    'paths':[],
                    'collisions':[]
                }
                for c in p['constraints']:
                    if c not in q['constraints']:
                        q['constraints'].append(c)
                for pa in p['paths']:
                    q['paths'].append(pa)
                
                ai = constraint['agent']
                astar = AStar(self.my_map,self.starts, self.goals,self.heuristics,ai,q['constraints'])
                path = astar.find_paths()

                if path is not None:
                    q['paths'][ai]= path[0]
                    # task 4
                    continue_flag = False
                    if constraint['positive']:
                        vol = paths_violate_constraint(constraint,q['paths'])
                        for v in vol:
                            astar_v = AStar(self.my_map,self.starts, self.goals,self.heuristics,v,q['constraints'])
                            path_v = astar_v.find_paths()
                            if path_v  is None:
                                continue_flag =True
                            else:
                                q['paths'][v] = path_v[0]
                        if continue_flag:
                            continue
                    q['collisions'] = detect_collisions(q['paths'])
                    q['cost'] = get_sum_of_cost(q['paths'])
                    self.push_node(q)     
        return None

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))

        print("Solution:")
        for i in range(len(node['paths'])):
            print("agent", i, ": ", node['paths'][i])
            
#--------------------------------------------------------------------------------------------------
# --- positive/negative constraint 구조 실험 ---
def disjoint_splitting(collision):
    '''
    양쪽 중 한 agent만(랜덤) 해라/하지마(positive/negative) constraint 부여
    양성 제약: 반드시 해당 위치/엣지 통과/도착
    음성 제약: 해당 위치/엣지 불가
    '''
    constraints = []
    agent = random.randint(0,1)
    a = 'a'+str(agent +1)
    if len(collision['loc'])==1:
        constraints.append({'agent':collision[a],
                            'loc':collision['loc'],
                            'timestep':collision['timestep'],
                            'positive':True
                            })
        constraints.append({'agent':collision[a],
                            'loc':collision['loc'],
                            'timestep':collision['timestep'],
                            'positive':False
                            })
    else:
        if agent ==0:
            constraints.append({'agent':collision[a],
                                'loc':[collision['loc'][0],collision['loc'][1]],
                                'timestep':collision['timestep'],
                                'positive':True
                                })
            constraints.append({'agent':collision[a],
                                'loc':[collision['loc'][0],collision['loc'][1]],
                                'timestep':collision['timestep'],
                                'positive':False
                                })
        else:
            constraints.append({'agent':collision[a],
                                'loc':[collision['loc'][1],collision['loc'][0]],
                                'timestep':collision['timestep'],
                                'positive':True
                                })
            constraints.append({'agent':collision[a],
                                'loc':[collision['loc'][1],collision['loc'][0]],
                                'timestep':collision['timestep'],
                                'positive':False
                                })
    return constraints

# --- CBS 하위 A*에서 positive constraint 만족하지 않으면 경로 재탐색 ---
def paths_violate_constraint(constraint, paths):
    '''
    주어진 positive constraint(=특정 agent만 가능한 상황)에서,
    나머지 agent 경로들이 위반하는지(즉, positive constraint를 뚫는지) 체크
    위반 agent의 인덱스 리스트 반환
    '''
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst

def resolve_conflict_by_type(agent1, agent2, agent_types, config):
    '''
    CBS 갈등시 "누가 constraint를 받을지" 결정하는 함수
    OUTBOUND(0)이 INBOUND(1)보다 우선이면,
    충돌 시 INBOUND agent에게 constraint(멈춤)를 부여
    '''
    type_pri1 = config.TYPE_PRIORITY.get(agent_types[agent1], 99)
    type_pri2 = config.TYPE_PRIORITY.get(agent_types[agent2], 99)
    if type_pri1 < type_pri2:
        return agent2  # agent2가 constraint(멈춤)
    elif type_pri2 < type_pri1:
        return agent1
    else:
        return min(agent1, agent2)  # 동등하면 id 작은 쪽에 constraint

# --- 표준/랜덤 대신, type/priority에 따라 constraint를 한 agent에게만 부여 ----
def type_priority_splitting(collision, agent_types, config):
    # 충돌한 두 agent 중 누가 constraint를 받을지 결정
    to_constrain = resolve_conflict_by_type(collision['a1'], collision['a2'], agent_types, config)
    constraints = []
    if len(collision['loc']) == 1:
        constraints.append({'agent': to_constrain, 'loc': collision['loc'],
                            'timestep': collision['timestep'], 'positive': False})
    else:
        constraints.append({'agent': to_constrain, 'loc': [collision['loc'][0], collision['loc'][1]],
                            'timestep': collision['timestep'], 'positive': False})
    return constraints
