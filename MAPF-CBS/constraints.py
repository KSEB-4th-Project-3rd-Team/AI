# def is_constrained, build_constraint_table, constraint_violated, future_constraint_violated,
# check_constraints
# 피킹존 5초 이상 머무르기 금지' 같은 경우는 조건을 추가해야 함
# constraint list 여러번 호출함 constraint table을 defaultdict로 바꾸기
import copy

def build_constraint_table(self, agent):
    constraint_table = dict()

    # 제약조건이 없으면 빈 테이블 반환
    if not self.constraints:
        return constraint_table

    for constraint in self.constraints:
        timestep = constraint['timestep']

        t_constraint = []
        if timestep in constraint_table:
            t_constraint = constraint_table[timestep]

        # 현재 agent에 대한 positive 제약조건
        if constraint['positive'] and constraint['agent'] == agent:
            t_constraint.append(constraint)
            constraint_table[timestep] = t_constraint
        # 현재 agent에 대한 negative(외부) 제약조건
        elif not constraint['positive'] and constraint['agent'] == agent:
            t_constraint.append(constraint)
            constraint_table[timestep] = t_constraint
        # 타 agent의 positive 제약조건 -> 현 agent에게 negative 제약조건으로 추가
        elif constraint['positive']:
            neg_constraint = copy.deepcopy(constraint)
            neg_constraint['agent'] = agent
            # 이동 제약인 경우, 방향 전환
            if len(constraint['loc']) == 2:
                prev_loc = constraint['loc'][1]
                curr_loc = constraint['loc'][0]
                neg_constraint['loc'] = [prev_loc, curr_loc]
            neg_constraint['positive'] = False
            t_constraint.append(neg_constraint)
            constraint_table[timestep] = t_constraint

    return constraint_table


def constraint_violated(self, curr_loc, next_loc, timestep, c_table_agent, agent):
    # 해당 timestep에 제약이 없으면 통과
    if timestep not in c_table_agent:
        return None

    for constraint in c_table_agent[timestep]:
        if agent == constraint['agent']:
            # 정점(위치) 제약
            if len(constraint['loc']) == 1:
                # positive: 반드시 특정 위치로 가야함
                if constraint['positive'] and next_loc != constraint['loc'][0]:
                    return constraint
                # negative: 특정 위치로 가면 안 됨
                elif not constraint['positive'] and next_loc == constraint['loc'][0]:
                    return constraint
            # 이동 제약
            else:
                # positive: 반드시 특정 경로로 이동해야 함
                if constraint['positive'] and constraint['loc'] != [curr_loc, next_loc]:
                    return constraint
                # negative: 특정 경로로 이동하면 안 됨
                if not constraint['positive'] and constraint['loc'] == [curr_loc, next_loc]:
                    return constraint

    return None

def future_constraint_violated(self, curr_loc, timestep, max_timestep, c_table_agent, agent):
    for t in range(timestep+1, max_timestep+1):
        if t not in c_table_agent:
            continue
        for constraint in c_table_agent[t]:
            if agent == constraint['agent']:
                # 정점(위치) 제약
                if len(constraint['loc']) == 1:
                    # positive: 반드시 특정 위치여야 함
                    if constraint['positive'] and curr_loc != constraint['loc'][0]:
                        return True
                    # negative: 해당 위치에 있으면 안 됨
                    elif not constraint['positive'] and curr_loc == constraint['loc'][0]:
                        return True
    return False
