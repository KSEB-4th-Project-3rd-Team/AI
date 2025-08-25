# scheduler.py
import pandas as pd

def build_dispatch_queue_from_df(df_in: pd.DataFrame):
    df = df_in.copy()

    # 맨해튼 거리(타이브레이커)
    mh = (df["start_y"] - df["goal_y"]).abs() + (df["start_x"] - df["goal_x"]).abs()
    df = df.assign(_mh=mh)

    status_order = pd.Categorical(df["status"], categories=["WAIT","PENDING","READY","RUN"], ordered=True)
    df = df.assign(_status=status_order)

    df = df.sort_values(by=["_status","requested_time","priority","_mh","order_item_id"]).reset_index(drop=True)
    return df[["order_item_id","type","start_y","start_x","goal_y","goal_x","priority","assigned_agent","requested_time"]]
