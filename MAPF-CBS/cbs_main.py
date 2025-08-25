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
    print(f"\n[DEBUG] run_agent_sequential_tasks() 시작, agent_id={agent_id}, max_tasks={max_tasks}, loop={loop}")

    my_map, _, _, rack_list = load_map(config.MAP_PATH)
    print("[DEBUG] load_map 완료, rack_list 길이:", len(rack_list) if rack_list else 0)
    racks = [rack["loc"] for rack in rack_list] if rack_list else None

    tasks = config.get_tasks_for_agent(agent_id=agent_id, max_tasks=max_tasks)
    print(f"[DEBUG] get_tasks_for_agent 완료, 추출된 tasks.shape: {tasks.shape}")
    if tasks.empty:
        print(f"[경고] agent {agent_id}의 미션이 없습니다.")
        return

    print("[DEBUG] 경로 탐색 시작")
    paths = []
    stats = []
    curr_pos = (int(tasks.iloc[0]['start_y']), int(tasks.iloc[0]['start_x']))
    for idx, row in tasks.iterrows():
        goal = (int(row['goal_y']), int(row['goal_x']))
        print(f"[DEBUG] ({idx}) A* 탐색 시작: curr_pos={curr_pos}, goal={goal}")
        if config.HEURISTIC_TYPE == "manhattan":
            h_values = multi_manhattan_heuristics(my_map, {0: goal})
        elif config.HEURISTIC_TYPE == "dijkstra":
            h_values = {0: Dijkstra_heuristics(my_map, goal)}
        else:
            raise ValueError(f"Unknown heuristic type: {config.HEURISTIC_TYPE}")
        t0 = time.time()
        path = a_star_single(my_map, curr_pos, goal, h_values[0], 0, constraints=[])
        elapsed = time.time() - t0

        print(f"[DEBUG] ({idx}) path 길이: {len(path) if path else 0}, path head: {path[:3] if path else None}, 시간: {elapsed:.4f}s")
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

    print("[DEBUG] 경로 평탄화(full_path 생성) 시작")
    fig_path = config.get_result_path("fig")
    anim_path = config.get_result_path("anim")
    stat_path = config.get_result_path("stat")
    print(f"[DEBUG] 저장 경로 - fig: {fig_path}, anim: {anim_path}, stat: {stat_path}")

    # === 전체 path 연결 (중복 없이 1차원으로 평탄화) ===
    full_path = []
    task_splits = [0]  # 태스크 경계 인덱스 기록
    curr_pos = (int(tasks.iloc[0]['start_y']), int(tasks.iloc[0]['start_x']))
    for idx, row in tasks.iterrows():
        task_start = (int(row['start_y']), int(row['start_x']))
        task_goal = (int(row['goal_y']), int(row['goal_x']))
        if curr_pos != task_start:
            path_to_start = a_star_single(my_map, curr_pos, task_start, h_values[0], 0, [])
            if path_to_start:
                if full_path and path_to_start[0] == full_path[-1]:
                    full_path += path_to_start[1:]
                else:
                    full_path += path_to_start
            curr_pos = task_start
        task_path = a_star_single(my_map, curr_pos, task_goal, h_values[0], 0, [])
        if task_path:
            if full_path and task_path[0] == full_path[-1]:
                full_path += task_path[1:]
            else:
                full_path += task_path
            curr_pos = task_goal
        # 경계 저장 (현재 full_path 길이 = 다음 태스크 시작 인덱스)
        task_splits.append(len(full_path))

    print(f"[DEBUG] full_path 길이: {len(full_path)}, head: {full_path[:5]}")
    # 태스크별 path가 여러 개라면, 항상 2차원 리스트 (list of list of (y,x))여야 함
    for idx, path in enumerate(paths):
        if not isinstance(path, list) or not isinstance(path[0], tuple):
            print(f"[경고] paths[{idx}] 구조 이상: {type(path)}, {type(path[0])}")

    if config.SAVE_FIG:
        print("[DEBUG] plot_map_with_paths() 호출")
        plot_map_with_paths(
            my_map,
            full_path,
            task_splits,
            racks=racks,
            title=f"Agent{agent_id} Sequential {max_tasks}-Tasks",
            save_path=fig_path
        )
        print("[DEBUG] plot_map_with_paths() 완료")

    if config.SAVE_ANIMATION:
        print("[DEBUG] animate_paths() 호출")
        animate_paths(
            my_map,
            full_path,
            task_splits,
            racks=racks,
            interval=config.ANIMATION_INTERVAL,
            title=f"Agent{agent_id} Sequential {max_tasks}-Tasks",
            save_path=anim_path
        )
        print("[DEBUG] animate_paths() 완료")

    print("[DEBUG] stat 파일 저장")
    pd.DataFrame(stats).to_csv(stat_path, index=False)
    print(f"Agent{agent_id} 5개 미션 연속수행 실험 저장 완료: {stat_path}")

def main():
    print("[DEBUG] main() 진입")
    run_agent_sequential_tasks(agent_id=1, max_tasks=5, loop=0)
    print("[DEBUG] main() 종료")

if __name__ == "__main__":
    main()