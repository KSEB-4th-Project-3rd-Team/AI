# cbs_batch_run.py
import time
import pandas as pd
from visualize import plot_multi_paths, animate_multi_paths
from config import config
from ..config import config
from ..src.scheduler import build_dispatch_queue_from_df
from ..pathfinding import (
    CBSSolver,
    a_star_single,
    Dijkstra_heuristics,
    multi_manhattan_heuristics,
    load_map,
    plot_multi_paths,
    animate_multi_paths,
)

def _astar_shortest_len(my_map, start, goal, mode="dijkstra"):
    if mode == "dijkstra":
        h = Dijkstra_heuristics(my_map, goal)
    else:
        h = multi_manhattan_heuristics(my_map, {0: goal})[0]
    p = a_star_single(my_map, start, goal, h, 0, constraints=[])
    return len(p) if p else None

def _to_tuple(y, x): return (int(y), int(x))

def run_cbs_batches(my_map, dispatch_df, batch_size=8, run_id="CBS_run", shortest_mode="dijkstra"):
    """
    dispatch_df: build_dispatch_queue() 결과
    batch_size : 동시에 움직일 에이전트 수(=8)
    반환: per-mission KPI DataFrame
    """
    rows = []
    saved_visual = False   # 1) 초기화

    for s in range(0, len(dispatch_df), batch_size):
        batch = dispatch_df.iloc[s:s+batch_size]
        if batch.empty:
            break

        starts = [_to_tuple(r.start_y, r.start_x) for r in batch.itertuples(index=False)]
        goals  = [_to_tuple(r.goal_y,  r.goal_x)  for r in batch.itertuples(index=False)]

        t0 = time.time()
        solver = CBSSolver(my_map, starts, goals, heuristic=config.HEURISTIC_TYPE)  # 내부에서 A* 호출
        paths, n_gen, n_exp = solver.find_solution(disjoint=False)
        elapsed = time.time() - t0

        # 배치 충돌 검증(정상해면 0)
        col_list = detect_collisions(paths) if paths else None
        col_cnt  = len(col_list) if col_list else None

        # 에이전트별 KPI 행 추가  2) append는 루프 안에서!
        path_lengths = []
        for i, r in enumerate(batch.itertuples(index=False)):
            p  = paths[i] if paths else None
            pl = len(p) if p else None
            sl = _astar_shortest_len(my_map, starts[i], goals[i], mode=shortest_mode)
            eff = (sl/pl*100) if (sl and pl and pl > 0) else None

            path_lengths.append(pl or 0)

            rows.append({
                "run_id": run_id, "algo": "CBS",
                "order_item_id": r.order_item_id, "type": r.type,
                "start": str(starts[i]), "goal": str(goals[i]),
                "path_length": pl, "shortest_len": sl,
                "path_efficiency_pct": round(min(100.0, eff), 2) if eff else None,
                "elapsed_time": elapsed,                   # 배치 탐색 총 시간(s)
                "replan_latency_ms": round(elapsed*1000, 3),
                "collisions_in_batch": col_cnt,            # 배치 충돌 수(검증용)
                "success": bool(pl and pl > 0),
                "episode_len": pl
            })

        # ---- 시각화: 첫 배치만 저장 (원하면 조건 제거하고 매 배치 저장) ----
        if not saved_visual and paths and (config.SAVE_FIG or config.SAVE_ANIMATION):
            fig_path  = config.get_result_path("fig",  max_tasks=len(dispatch_df), extra=f"cbs_batch{s//batch_size}")
            anim_path = config.get_result_path("anim", max_tasks=len(dispatch_df), extra=f"cbs_batch{s//batch_size}")

            if config.SAVE_FIG:
                plot_multi_paths(
                    my_map, paths,
                    title=f"CBS Batch {s//batch_size}",
                    save_path=fig_path,
                    agent_marker_size=config.AGENT_MARKER_SIZE,
                    agent_line_width=config.AGENT_LINE_WIDTH,
                    rack_alpha=config.RACK_ALPHA,
                    path_lengths=path_lengths                 # 거리 표기용 전달
                )
            if config.SAVE_ANIMATION:
                animate_multi_paths(
                    my_map, paths,
                    interval=config.ANIMATION_INTERVAL,
                    title=f"CBS Batch {s//batch_size}",
                    save_path=anim_path,
                    agent_marker_size=config.AGENT_MARKER_SIZE,
                    agent_line_width=config.AGENT_LINE_WIDTH,
                    path_lengths=path_lengths                 # 거리 표기용 전달
                )
            saved_visual = True

    return pd.DataFrame(rows)
