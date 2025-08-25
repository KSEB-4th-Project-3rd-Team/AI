# metrics_cbs_8agents.py
import os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "pathfinding"))

from ..config import config
from ..scheduler import build_dispatch_queue_from_df
from ..pathfinding import (
    CBSSolver,
    a_star_single,
    Dijkstra_heuristics,
    multi_manhattan_heuristics,
)
def summarize_kpi(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    out["missions"]          = len(df)
    out["success_%"]         = round(df["success"].mean()*100, 2)
    out["path_eff_p50_%"]    = round(df["path_efficiency_pct"].median(), 2)
    out["path_eff_mean_%"]   = round(df["path_efficiency_pct"].mean(), 2)
    out["path_len_p50"]      = round(df["path_length"].median(), 2)
    out["latency_p50_ms"]    = round(df["replan_latency_ms"].median(), 2)
    out["latency_p95_ms"]    = round(df["replan_latency_ms"].quantile(0.95), 2)
    col_sum = df["collisions_in_batch"].dropna().sum()
    out["collisions_per_1000"] = round((col_sum / max(1, len(df))) * 1000.0, 3)
    return pd.DataFrame([out])

def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1) 장애물 격자만 맵에서 로드
    my_map, *_ = load_map(config.MAP_PATH)

    # 2) 미션 CSV 로드/정합성 체크
    missions = pd.read_csv(config.ASSIGN_TASK_PATH)
    missions = validate_and_coerce(missions, my_map, drop_bad=True)

    # 3) 샘플링 정책 적용(필요 시)
    if config.SAMPLE_MODE == "head":
        missions = missions.head(config.MAX_TASKS_PER_AGENT)
    elif config.SAMPLE_MODE == "random":
        missions = missions.sample(n=min(config.MAX_TASKS_PER_AGENT, len(missions)),
                                   random_state=config.SAMPLE_SEED)

    # 4) PPO와 동일 규칙으로 큐 생성
    dispatch = build_dispatch_queue_from_df(missions)

    # 5) CBS 실행(동시 8개)
    df = run_cbs_batches(
        my_map=my_map,
        dispatch_df=dispatch,
        batch_size=8,                # 고정 8
        run_id=f"{config.EXP_NO}_CBS",
        shortest_mode="dijkstra"     # A* 최단거리 기준으로 효율 계산
    )

    # 6) 저장(미션별/요약)
    per_mission_path = config.get_result_path("stat", max_tasks=len(dispatch), extra="cbs8")
    df.to_csv(per_mission_path, index=False)

    summary = summarize_kpi(df)
    summary_path = config.get_result_path("stat", max_tasks=len(dispatch), extra="cbs8_summary")
    summary.to_csv(summary_path, index=False)

    print(f"[DONE] per-mission KPI: {per_mission_path}")
    print(f"[DONE] summary KPI    : {summary_path}")
    print(summary.to_string(index=False))

def summarize_kpi(df: pd.DataFrame, step_time_sec: float | None = None) -> pd.DataFrame:
    out = {}
    out["missions"]        = len(df)
    out["success_%"]       = round(df["success"].mean()*100, 2)
    out["path_eff_p50_%"]  = round(df["path_efficiency_pct"].median(), 2)
    out["path_eff_mean_%"] = round(df["path_efficiency_pct"].mean(), 2)
    out["path_len_p50"]    = round(df["path_length"].median(), 2)
    out["latency_p50_ms"]  = round(df["replan_latency_ms"].median(), 2)
    out["latency_p95_ms"]  = round(df["replan_latency_ms"].quantile(0.95), 2)
    # 충돌/1,000 환산(배치 합계 기준의 근사치)
    col_sum = df["collisions_in_batch"].dropna().sum()
    out["collisions_per_1000"] = round((col_sum / max(1, len(df))) * 1000.0, 3)
    # (선택) 미션 시간(s) = step * step_time
    if step_time_sec is not None:
        mission_time = df["path_length"].dropna() * float(step_time_sec)
        out["mission_time_p50_s"] = round(mission_time.median(), 2)
        out["mission_time_mean_s"]= round(mission_time.mean(), 2)
    return pd.DataFrame([out])


if __name__ == "__main__":
    main()
