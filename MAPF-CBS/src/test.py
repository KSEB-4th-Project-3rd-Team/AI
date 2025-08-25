# validate_missions.py
import pandas as pd
import numpy as np

REQUIRED = ["order_item_id","type","start_y","start_x","goal_y","goal_x","status","priority","requested_time"]

def validate_and_coerce(df: pd.DataFrame, my_map, drop_bad=True):
    # 1) 필수 컬럼 확인
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    # 2) 타입 강제(int)
    for c in ["start_y","start_x","goal_y","goal_x","priority","requested_time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    H, W = len(my_map), len(my_map[0]) if len(my_map)>0 else 0

    def in_bounds(y, x): return (0 <= y < H) and (0 <= x < W)
    def is_free(y, x):   return my_map[y][x] == 0

    # 3) 좌표 결측/범위/장애물 셀 검증
    mask_ok = []
    for r in df.itertuples(index=False):
        sy, sx, gy, gx = int(r.start_y), int(r.start_x), int(r.goal_y), int(r.goal_x)
        ok = (in_bounds(sy, sx) and in_bounds(gy, gx) and is_free(sy, sx) and is_free(gy, gx))
        mask_ok.append(ok)

    mask_ok = np.array(mask_ok, dtype=bool)
    if drop_bad:
        bad = (~mask_ok).sum()
        if bad:
            print(f"[WARN] 좌표 불량 {bad}건 제외(범위/장애물 위): order_item_id={df.loc[~mask_ok,'order_item_id'].tolist()}")
        df = df.loc[mask_ok].reset_index(drop=True)
    else:
        if (~mask_ok).any():
            raise ValueError("좌표 불량 행 존재")

    # 4) status 기본값/정렬용 보조필드
    df["status"] = df["status"].fillna("WAIT")
    return df

print(df)