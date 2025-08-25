# cbs_batch_run.py
import time
import ast
import numpy as np
import pandas as pd
from .collision import detect_collisions, standard_splitting  # 필요시 다른 splitting도 추가
from .astar import a_star_single, Dijkstra_heuristics, multi_manhattan_heuristics

# CBSSolver.py
import time
from collision import detect_collisions, standard_splitting
from pathfinding.Astar import a_star_single, Dijkstra_heuristics, multi_manhattan_heuristics

class CBSSolver:
    def __init__(self, my_map, starts, goals, heuristic="dijkstra"):
        self.my_map = my_map
        self.starts = list(starts)
        self.goals  = list(goals)
        self.n = len(starts)
        self.heuristic = heuristic

    def _heuristic_table(self, agent_idx, goal):
        if self.heuristic == "dijkstra":
            return Dijkstra_heuristics(self.my_map, goal)
        else:
            return multi_manhattan_heuristics(self.my_map, {agent_idx: goal})[agent_idx]

    def _low_level(self, agent_idx, start, goal, constraints):
        h = self._heuristic_table(agent_idx, goal)
        return a_star_single(self.my_map, start, goal, h, agent_idx, constraints)

    def find_solution(self, disjoint=False):
        # 1) 초기 경로(제약 없음)
        constraints = []
        paths = []
        for i in range(self.n):
            p = self._low_level(i, self.starts[i], self.goals[i], constraints)
            if p is None: return None, 0, 0
            paths.append(p)

        # 2) 충돌이 없어질 때까지 split
        generated = expanded = 0
        open_nodes = [ {"paths": paths, "constraints": constraints} ]

        while open_nodes:
            node = open_nodes.pop()   # LIFO (스택처럼)
            expanded += 1
            cols = detect_collisions(node["paths"])
            if not cols:
                return node["paths"], generated, expanded

            col = cols[0]
            branches = standard_splitting(col)

            for c in branches:
                new_constraints = node["constraints"] + [c]
                new_paths = node["paths"][:]
                ai = c["agent"]

                p = self._low_level(ai, self.starts[ai], self.goals[ai], new_constraints)
                if p is None:
                    continue
                new_paths[ai] = p
                open_nodes.append({"paths": new_paths, "constraints": new_constraints})
                generated += 1

        return None, generated, expanded





def _astar_shortest_len(my_map, start, goal, heuristic="dijkstra"):
    # 최단거리(정확) 계산 : baseline
    if heuristic == "dijkstra":
        h = Dijkstra_heuristics(my_map, goal)
    else:  # "manhattan"
        h = multi_manhattan_heuristics(my_map, {0: goal})[0]
    sp = a_star_single(my_map, start, goal, h, 0, constraints=[])
    return len(sp) if sp else None

def _to_tuple(y, x): return (int(y), int(x))

def run_cbs_batches(my_map, dispatch_df, batch_size=4, run_id="cbs_run_1"):
    """
    dispatch_df: build_dispatch_queue() 결과
    batch_size : 동시에 움직일 에이전트 수(= CBS 적용 대상 수)
    반환: per-mission KPI DataFrame
    """
    rows = []
    for s in range(0, len(dispatch_df), batch_size):
        batch = dispatch_df.iloc[s:s+batch_size]
        if batch.empty: break

        starts = [ _to_tuple(r.start_y, r.start_x) for r in batch.itertuples(index=False) ]
        goals  = [ _to_tuple(r.goal_y, r.goal_x) for r in batch.itertuples(index=False) ]

        # ★ CBS 실행
        t0 = time.time()
        solver = CBSSolver(my_map, starts, goals)
        paths, n_gen, n_exp = solver.find_solution(disjoint=False)  # 표준 split
        elapsed = time.time() - t0

        # 실패시 스킵/로그
        if paths is None:
            for i, r in enumerate(batch.itertuples(index=False)):
                rows.append({
                    "run_id": run_id, "algo": "CBS",
                    "order_item_id": r.order_item_id, "type": r.type,
                    "start": str(starts[i]), "goal": str(goals[i]),
                    "path_length": None, "shortest_len": None,
                    "path_efficiency_pct": None, "elapsed_time": elapsed,
                    "replan_latency_ms": round(elapsed*1000,3),
                    "collisions": None, "success": False, "episode_len": None
                })
            continue

        # KPI 집계 (배치 내 각 에이전트별)
        for i, r in enumerate(batch.itertuples(index=False)):
            path = paths[i]
            pl = len(path) if path else None
            sl = _astar_shortest_len(my_map, starts[i], goals[i], heuristic="dijkstra")
            eff = (sl/pl*100) if (sl and pl and pl>0) else None
            rows.append({
                "run_id": run_id, "algo": "CBS",
                "order_item_id": r.order_item_id, "type": r.type,
                "start": str(starts[i]), "goal": str(goals[i]),
                "path_length": pl, "shortest_len": sl,
                "path_efficiency_pct": round(min(100.0, eff),2) if eff else None,
                "elapsed_time": elapsed,  # 배치 전체 탐색 시간(재계산 시간 p50/p95 산출용 원시값)
                "replan_latency_ms": round(elapsed*1000,3),
                "collisions": 0,         # CBSSolver가 충돌 없는 해를 내므로 0 (실패시 None)
                "success": bool(pl and pl>0),
                "episode_len": pl        # 경로길이 proxy
            })
    return pd.DataFrame(rows)


# run_id, algo, order_item_id, type, start, goal,
# path_length, shortest_len, path_efficiency_pct,
# elapsed_time, replan_latency_ms,
# collisions, success, episode_len
