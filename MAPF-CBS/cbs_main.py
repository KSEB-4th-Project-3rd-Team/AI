import sys
import os
sys.path.append(os.path.abspath("src"))
from config import config
os.makedirs(config.RESULTS_DIR, exist_ok=True)
import random
import numpy as np
import time
import pandas as pd
from pathfinding.MapLoader import load_map
from pathfinding.Astar import (
    multi_manhattan_heuristics,
    Dijkstra_heuristics,
    a_star_single,
    get_sum_of_cost
)
from pathfinding.visualize import (
    plot_map_with_paths,
    animate_paths
)

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

def run_agent_sequential_tasks(agent_id=1, max_tasks=5, loop=0):
    my_map, _, _, rack_list = load_map(config.MAP_PATH)
    racks = [rack["loc"] for rack in rack_list] if rack_list else None

    tasks = config.get_tasks_for_agent(agent_id=agent_id, max_tasks=max_tasks)
    if tasks.empty:
        print(f"[경고] agent {agent_id}의 미션이 없습니다.")
        return

    paths = []
    stats = []
    curr_pos = (int(tasks.iloc[0]['start_y']), int(tasks.iloc[0]['start_x']))
    for idx, row in tasks.iterrows():
        goal = (int(row['goal_y']), int(row['goal_x']))
        if config.HEURISTIC_TYPE == "manhattan":
            h_values = multi_manhattan_heuristics(my_map, {0: goal})
        elif config.HEURISTIC_TYPE == "dijkstra":
            h_values = {0: Dijkstra_heuristics(my_map, goal)}
        else:
            raise ValueError(f"Unknown heuristic type: {config.HEURISTIC_TYPE}")
        t0 = time.time()
        path = a_star_single(my_map, curr_pos, goal, h_values[0], 0, constraints=[])
        elapsed = time.time() - t0

        # path의 구조를 반드시 확인
        print(f">>> DEBUG: idx={idx} path type={type(path)} path[:3]={path[:3]}")
        paths.append(path)
        stats.append({
            "order_item_id": row['order_item_id'],
            "type": row['type'],
            "start": str(curr_pos),
            "goal": str(goal),
            "path_length": len(path) if path else None,
            "elapsed_time": elapsed,
            "full_path": str(path)
        })
        curr_pos = goal

    fig_path = config.get_result_path("fig", loop=loop)
    anim_path = config.get_result_path("anim", loop=loop)
    stat_path = config.get_result_path("stat", loop=loop)

    # === 전체 path 연결 (중복 없이 1차원으로 평탄화) ===
    full_path = []
    for i, p in enumerate(paths):
        # p가 반드시 1차원 튜플 리스트여야 함!
        if i == 0:
            full_path.extend(p)
        else:
            full_path.extend(p[1:])
    print(">>> DEBUG: paths type =", type(paths))
    print(">>> DEBUG: paths[0] type =", type(paths[0]), "sample =", paths[0][:3])
    print(">>> DEBUG: full_path type =", type(full_path))
    print(">>> DEBUG: full_path[:3] =", full_path[:3])

    if config.SAVE_FIG:
        plot_map_with_paths(
            my_map,
            [tuple(map(int, tasks.iloc[0][['start_y','start_x']]))],
            [tuple(map(int, tasks.iloc[-1][['goal_y','goal_x']]))],
            [full_path],      # 1차원 튜플리스트를 리스트로 한 번만 감싸서!
            racks=racks,
            agent_names=[f"AGV_{agent_id}"],
            title=f"Agent{agent_id} Sequential 5-Tasks",
            save_path=fig_path
        )
    if config.SAVE_ANIMATION:
        animate_paths(
            my_map, [full_path],
            racks=racks,
            interval=config.ANIMATION_INTERVAL,
            title=f"Agent{agent_id} Sequential 5-Tasks",
            save_path=anim_path
        )

    pd.DataFrame(stats).to_csv(stat_path, index=False)
    print(f"Agent{agent_id} 5개 미션 연속수행 실험 저장 완료: {stat_path}")

def main():
    run_agent_sequential_tasks(agent_id=1, max_tasks=5, loop=0)

if __name__ == "__main__":
    main()
