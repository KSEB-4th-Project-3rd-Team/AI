# -*- coding: utf-8 -*-
# Final v2: residual demand + lead-time (Huber) + per-item sequences + robust evaluation
import os, warnings, platform, logging
from pathlib import Path
warnings.filterwarnings('ignore')

# ---- Env (set BEFORE TF import) ----
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from datetime import date, timedelta
from collections import deque

# ---------------- Seeds & GPU ----------------
np.random.seed(42); tf.random.set_seed(42)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"Using {len(gpus)} GPU(s): {[d.name for d in gpus]}")
    except Exception as e:
        logging.warning(f"GPU setup failed: {e}")
else:
    logging.info("No GPU detected → CPU mode")

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 1) Load & fill ----------------
korean_holidays = [
    date(2010,1,1),date(2010,2,13),date(2010,2,14),date(2010,2,15),date(2010,3,1),date(2010,5,5),
    date(2010,6,6),date(2010,8,15),date(2010,9,21),date(2010,9,22),date(2010,9,23),date(2010,10,3),
    date(2010,10,9),date(2010,12,25),date(2011,1,1),date(2011,2,2),date(2011,2,3),date(2011,2,4),
    date(2011,3,1),date(2011,5,5),date(2011,6,6),date(2011,8,15),date(2011,9,12),date(2011,9,13),
    date(2011,9,14),date(2011,10,3),date(2011,10,9),date(2011,12,25),date(2012,1,1),date(2012,1,22),
    date(2012,1,23),date(2012,1,24),date(2012,3,1),date(2012,4,4),date(2012,5,5),date(2012,5,28),
    date(2012,6,6),date(2012,8,15),date(2012,9,29),date(2012,9,30),date(2012,10,1),date(2012,10,3),
    date(2012,10,9),date(2012,12,25),date(2013,1,1),date(2013,2,9),date(2013,2,10),date(2013,2,11),
    date(2013,3,1),date(2013,5,5),date(2013,5,17),date(2013,6,6),date(2013,8,15),date(2013,9,18),
    date(2013,9,19),date(2013,9,20),date(2013,10,3),date(2013,10,9),date(2013,12,25),date(2014,1,1),
    date(2014,1,30),date(2014,1,31),date(2014,2,1),date(2014,3,1),date(2014,5,5),date(2014,5,6),
    date(2014,6,4),date(2014,6,6),date(2014,8,15),date(2014,9,7),date(2014,9,8),date(2014,9,9),
    date(2014,10,3),date(2014,10,9),date(2014,12,25),date(2015,1,1),date(2015,2,18),date(2015,2,19),
    date(2015,2,20),date(2015,3,1),date(2015,5,1),date(2015,5,5),date(2015,5,25),date(2015,6,6),
    date(2015,8,15),date(2015,9,26),date(2015,9,27),date(2015,9,28),date(2015,10,3),date(2015,10,9),
    date(2015,12,25),date(2016,1,1),date(2016,2,7),date(2016,2,8),date(2016,2,9),date(2016,2,10),
    date(2016,3,1),date(2016,5,5),date(2016,5,6),date(2016,6,6),date(2016,8,15),date(2016,9,14),
    date(2016,9,15),date(2016,9,16),date(2016,10,3),date(2016,10,9),date(2016,12,25),date(2017,1,1),
    date(2017,1,27),date(2017,1,28),date(2017,1,29),date(2017,1,30),date(2017,3,1),date(2017,5,1),
    date(2017,5,3),date(2017,5,5),date(2017,5,9),date(2017,6,6),date(2017,8,15),date(2017,10,2),
    date(2017,10,3),date(2017,10,4),date(2017,10,5),date(2017,10,6),date(2017,10,9),date(2017,12,25),
    date(2018,1,1),date(2018,2,15),date(2018,2,16),date(2018,2,17),date(2018,3,1),date(2018,5,5),
    date(2018,5,7),date(2018,5,22),date(2018,6,6),date(2018,8,15),date(2018,9,23),date(2018,9,24),
    date(2018,9,25),date(2018,9,26),date(2018,10,3),date(2018,10,9),date(2018,12,25),date(2019,1,1),
    date(2019,2,4),date(2019,2,5),date(2019,2,6),date(2019,3,1),date(2019,5,5),date(2019,5,6),
    date(2019,6,6),date(2019,8,15),date(2019,9,12),date(2019,9,13),date(2019,9,14),date(2019,10,3),
    date(2019,10,9),date(2019,12,25)
]
korean_holidays_dt = pd.to_datetime(korean_holidays)

df = pd.read_csv("preprocessed_tire_demand.csv", index_col=0, parse_dates=True)
print(f"[INFO] Raw rows: {len(df)}")

# reindex per item (fill)
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
buf = []
for item in df['item_id'].unique():
    t = df[df['item_id']==item].reindex(full_range)
    t['number_sold'] = t['number_sold'].fillna(0)
    t['item_id'] = t['item_id'].fillna(item)
    t = t.fillna(method='ffill').fillna(method='bfill')
    buf.append(t)
df = pd.concat(buf).sort_index()
print(f"[INFO] After fill: {len(df)} rows")

# ---------------- 2) Feature engineering ----------------
df['is_holiday'] = df.index.isin(korean_holidays_dt).astype(int)
df['number_sold_log'] = np.log1p(df['number_sold'])
if 'promotion' not in df.columns: df['promotion'] = 0
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year

# Demand lags/MAs (log)
g = df.groupby('item_id')['number_sold_log']
df['lag_1']  = g.shift(1)
df['lag_7']  = g.shift(7)
df['ma_7']   = g.transform(lambda s: s.rolling(7,  min_periods=1).mean())
df['ma_28']  = g.transform(lambda s: s.rolling(28, min_periods=1).mean())

# Lead-time AR helpers
if 'lead_time' not in df.columns:
    raise ValueError("Missing 'lead_time' column for lead-time dashboard.")
gl = df.groupby('item_id')['lead_time']
df['lead_time_lag1'] = gl.shift(1)
df['lead_time_ma7']  = gl.transform(lambda s: s.rolling(7, min_periods=1).mean())

for c in ['lag_1','lag_7','ma_7','ma_28','lead_time_lag1','lead_time_ma7']:
    df[c] = df[c].fillna(method='bfill')

items = df['item_id'].unique()
df_enc = pd.get_dummies(df, columns=['item_id'], prefix='item', dtype=int)

# ---------------- 3) Split & scaling ----------------
TEST_START = "2019-01-01"
train_df = df_enc[df_enc.index < TEST_START].copy()
test_df  = df_enc[df_enc.index >= TEST_START].copy()

feature_cols = [c for c in df_enc.columns if c not in ['number_sold','number_sold_log','lead_time']]

# global feature scaler (fit on TRAIN only)
feat_scaler = MinMaxScaler().fit(train_df[feature_cols])

# per-item scaler for log demand; plus lag1_scaled using same scaler
train_s = train_df.copy()
test_s  = test_df.copy()
demand_scalers = {}

for col in ['number_sold_log_scaled','lag1_scaled','lead_time_scaled','demand_resid']:
    train_s[col] = np.nan
    test_s[col]  = np.nan

# lead-time scaler (global)
lead_scaler = MinMaxScaler().fit(train_df[['lead_time']])
train_s['lead_time_scaled'] = lead_scaler.transform(train_df[['lead_time']])
test_s['lead_time_scaled']  = lead_scaler.transform(test_df[['lead_time']])

for item in items:
    mtr = (train_df.get(f'item_{item}', 0) == 1)
    mte = (test_df.get(f'item_{item}', 0)  == 1)
    sc_d = MinMaxScaler()

    tr_log = train_df.loc[mtr, ['number_sold_log']]
    if not tr_log.empty:
        # demand (scaled)
        train_s.loc[mtr, 'number_sold_log_scaled'] = sc_d.fit_transform(tr_log)
        # lag1_scaled with same scaler (use .values to bypass sklearn feature-name check)
        train_s.loc[mtr, 'lag1_scaled'] = sc_d.transform(train_df.loc[mtr, ['lag_1']].values).ravel()

    if mte.sum() > 0:
        te_log = test_df.loc[mte, ['number_sold_log']]
        if not te_log.empty:
            test_s.loc[mte, 'number_sold_log_scaled'] = sc_d.transform(te_log)
            test_s.loc[mte, 'lag1_scaled'] = sc_d.transform(test_df.loc[mte, ['lag_1']].values).ravel()

    demand_scalers[item] = sc_d

# residual targets
train_s['demand_resid'] = train_s['number_sold_log_scaled'] - train_s['lag1_scaled']
test_s['demand_resid']  = test_s['number_sold_log_scaled']  - test_s['lag1_scaled']

# drop rows without targets
train_s.dropna(subset=['demand_resid','lead_time_scaled'], inplace=True)
test_s.dropna(subset=['demand_resid','lead_time_scaled'],  inplace=True)

# ---------------- 4) Sequences (per item; also return lag1_at_target!) ----------------
N_STEPS = 56
TARGETS = ['demand_resid','lead_time_scaled']

def build_sequences_per_item(df_feat, df_targ, which: str):
    X, y, dates, item_ids, lag1_targets = [], [], [], [], []
    for item in items:
        mask_feat = (df_feat.get(f'item_{item}',0) == 1)
        idx_all = df_feat.index[mask_feat]
        idx_valid = df_targ.index.intersection(idx_all)
        if len(idx_valid) <= N_STEPS:
            print(f"[WARN] {which}: {item} not enough length ({len(idx_valid)})")
            continue

        feat = feat_scaler.transform(df_feat.loc[idx_valid, feature_cols])
        targ = df_targ.loc[idx_valid, TARGETS].values
        lag1 = df_targ.loc[idx_valid, 'lag1_scaled'].values
        dts  = idx_valid

        for i in range(len(dts) - N_STEPS):
            X.append(feat[i:i+N_STEPS])
            y.append(targ[i+N_STEPS])
            dates.append(dts[i+N_STEPS])
            item_ids.append(item)
            lag1_targets.append(lag1[i+N_STEPS])  # <<=== key fix

    if len(X)==0:
        raise ValueError(f"No sequences built for {which}. Check data/steps.")
    return (np.asarray(X, np.float32),
            np.asarray(y, np.float32),
            np.array(dates),
            np.array(item_ids, dtype=object),
            np.asarray(lag1_targets, np.float32))

X_train, y_train, tr_dates, tr_items, lag1_tr = build_sequences_per_item(train_df, train_s, "train")
X_test,  y_test,  te_dates, te_items, lag1_te = build_sequences_per_item(test_df,  test_s,  "test")

print(f"[INFO] Train: X={X_train.shape}, y={y_train.shape}")
print(f"[INFO] Test : X={X_test.shape},  y={y_test.shape}")

# ---------------- 5) Model ----------------
inp = layers.Input(shape=(N_STEPS, X_train.shape[2]))
x = layers.LSTM(128, return_sequences=True)(inp)
x = layers.Dropout(0.15)(x)
x = layers.LSTM(64, return_sequences=False)(x)
x = layers.Dropout(0.15)(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)

h_d = layers.Dense(32, activation="relu")(x); out_d = layers.Dense(1, name="demand")(h_d)  # residual
h_l = layers.Dense(32, activation="relu")(x); out_l = layers.Dense(1, name="lead")(h_l)

model = Model(inp, [out_d, out_l])

# Lead-time loss: Huber(δ=0.1) -> 수치 안정화 (MAE는 metric으로 계속 보고 비교)
lead_huber = tf.keras.losses.Huber(delta=0.1)

try:
    opt = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0)
except AttributeError:
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

model.compile(
    optimizer=opt,
    loss={"demand": "mse", "lead": lead_huber},
    loss_weights={"demand": 0.9, "lead": 0.1},
    metrics={"demand": "mae", "lead": "mae"}
)

ckpt = 'best_multi_output_lstm_resid.keras'
cbs = [
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=1e-6, verbose=1, min_delta=1e-4),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1, min_delta=1e-4),
    ModelCheckpoint(filepath=ckpt, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
]

print("\n[INFO] --- Training ---")
model.fit(
    X_train, {"demand": y_train[:,0], "lead": y_train[:,1]},
    epochs=60, batch_size=64, shuffle=True,
    validation_data=(X_test, {"demand": y_test[:,0], "lead": y_test[:,1]}),
    callbacks=cbs, verbose=2
)
print("[INFO] --- Training done ---")

# ---------------- 6) Evaluation ----------------
print("\n[INFO] --- Evaluation ---")
pred_resid, pred_l = model.predict(X_test, verbose=0)
pred_resid = pred_resid.reshape(-1); pred_l = pred_l.reshape(-1)

L = min(len(pred_resid), len(pred_l), len(y_test), len(lag1_te))
pred_resid = pred_resid[:L]; pred_l = pred_l[:L]; y_eval = y_test[:L]
dates_eval = pd.to_datetime(te_dates[:L]); items_eval = te_items[:L]
lag1_at_target = lag1_te[:L]                    # <<=== key fix: 위치 정렬된 lag1

# restore scaled log-demand = residual + lag1_scaled(target row)
pred_d_scaled = pred_resid + lag1_at_target
act_d_scaled  = y_eval[:,0] + lag1_at_target  # for fair compare

res = pd.DataFrame({
    "Date": dates_eval,
    "item_id": items_eval,
    "Actual_Demand_Log_Scaled": act_d_scaled,
    "Predicted_Demand_Log_Scaled": pred_d_scaled,
    "Actual_LeadTime_Scaled": y_eval[:,1],
    "Predicted_LeadTime_Scaled": pred_l,
})

# inverse transforms
res[["Actual_Demand","Predicted_Demand","Actual_LeadTime","Predicted_LeadTime"]] = np.nan
for item in items:
    m = (res["item_id"] == item)
    if m.sum()==0: continue
    sc = demand_scalers[item]
    act_arr = res.loc[m, "Actual_Demand_Log_Scaled"].to_numpy().reshape(-1, 1)
    pre_arr = res.loc[m, "Predicted_Demand_Log_Scaled"].to_numpy().reshape(-1, 1)
    act_log = sc.inverse_transform(act_arr)
    pre_log = sc.inverse_transform(pre_arr)
    res.loc[m, "Actual_Demand"]    = np.expm1(act_log).ravel()
    res.loc[m, "Predicted_Demand"] = np.expm1(pre_log).ravel()

res["Actual_LeadTime"]    = lead_scaler.inverse_transform(res[["Actual_LeadTime_Scaled"]]).ravel()
res["Predicted_LeadTime"] = lead_scaler.inverse_transform(res[["Predicted_LeadTime_Scaled"]]).ravel()

# post-process
res["Predicted_Demand"]   = res["Predicted_Demand"].clip(lower=0)
res["Predicted_LeadTime"] = res["Predicted_LeadTime"].clip(lower=0)

# metrics
def mape(a,p,eps=1e-10): return float(np.mean(np.abs((a-p)/(a+eps)))*100)
def smape(a,p,eps=1e-10): return float(np.mean(np.abs(p-a)/((np.abs(a)+np.abs(p))/2+eps))*100)

metrics = pd.DataFrame({
    "Target":["Demand","Demand","Demand","Demand","Demand","Lead Time","Lead Time","Lead Time"],
    "Metric":["MAE","RMSE","R2 Score","MAPE (%)","SMAPE (%)","MAE","RMSE","R2 Score"],
    "Value":[
        mean_absolute_error(res["Actual_Demand"], res["Predicted_Demand"]),
        np.sqrt(mean_squared_error(res["Actual_Demand"], res["Predicted_Demand"])),
        r2_score(res["Actual_Demand"], res["Predicted_Demand"]),
        mape(res["Actual_Demand"], res["Predicted_Demand"]),
        smape(res["Actual_Demand"], res["Predicted_Demand"]),
        mean_absolute_error(res["Actual_LeadTime"], res["Predicted_LeadTime"]),
        np.sqrt(mean_squared_error(res["Actual_LeadTime"], res["Predicted_LeadTime"])),
        r2_score(res["Actual_LeadTime"], res["Predicted_LeadTime"]),
    ]
})
metrics.to_csv("model_evaluation_metrics.csv", index=False, encoding="utf-8-sig")
res.to_csv("test_predictions_expanded.csv", index=False, encoding="utf-8-sig")
print("[INFO] Saved: model_evaluation_metrics.csv, test_predictions_expanded.csv")

# ---------------- 7) Plots (evaluation) ----------------
os.makedirs("plots_eval", exist_ok=True)
for item in items:
    d = res[res["item_id"]==item].sort_values("Date")
    if d.empty: continue

    r2d = r2_score(d["Actual_Demand"], d["Predicted_Demand"])
    plt.figure(figsize=(14,6))
    plt.plot(d["Date"], d["Actual_Demand"], label="Actual demand", alpha=0.85)
    plt.plot(d["Date"], d["Predicted_Demand"], label="Predicted demand", alpha=0.85, linestyle="--")
    plt.title(f"{item} — Demand (R² = {r2d:.3f})")
    plt.xlabel("Date"); plt.ylabel("Units"); plt.legend(); plt.grid(True, linewidth=0.3)
    plt.tight_layout(); plt.savefig(f"plots_eval/demand_{item.replace('/','_')}.png"); plt.close()

    r2l = r2_score(d["Actual_LeadTime"], d["Predicted_LeadTime"])
    plt.figure(figsize=(14,6))
    plt.plot(d["Date"], d["Actual_LeadTime"], label="Actual lead time", alpha=0.85)
    plt.plot(d["Date"], d["Predicted_LeadTime"], label="Predicted lead time", alpha=0.85, linestyle="--")
    plt.title(f"{item} — Lead time (R² = {r2l:.3f})")
    plt.xlabel("Date"); plt.ylabel("Days"); plt.legend(); plt.grid(True, linewidth=0.3)
    plt.tight_layout(); plt.savefig(f"plots_eval/lead_{item.replace('/','_')}.png"); plt.close()

print("[INFO] Saved evaluation plots -> plots_eval/")

# ---------------- 8) 30-day forecast + order date ----------------
print("\n[INFO] --- 30-day simulation & order date ---")
INIT_INV_MIN, SAFETY_MIN, HORIZON = 50, 10, 30

last_date = df.index.max()
future_days = pd.date_range(start=last_date + timedelta(days=1), periods=HORIZON, freq='D')

# init inventory by recent 30d avg
init_inv, safety = {}, {}
last30 = df.loc[df.index.max() - timedelta(days=29):]
for item in items:
    mean_recent = float(last30.loc[last30['item_id']==item, 'number_sold'].mean() or 0.0)
    init_inv[item] = max(INIT_INV_MIN, int(mean_recent*14))
    safety[item]   = max(SAFETY_MIN,   int(mean_recent*7))

rows = []
os.makedirs("plots_forecast", exist_ok=True)

for item in items:
    mask_hist = (df_enc.get(f"item_{item}",0) == 1)
    hist_feats = feat_scaler.transform(df_enc.loc[mask_hist, feature_cols])
    if len(hist_feats) < N_STEPS:
        print(f"[WARN] {item}: not enough history ({len(hist_feats)})")
        continue
    seq = hist_feats[-N_STEPS:].copy()

    logs_hist = df.loc[mask_hist, "number_sold_log"].values
    if len(logs_hist)==0:
        print(f"[WARN] {item}: no log history")
        continue
    dq = deque(list(logs_hist[-28:]), maxlen=28)

    sc_d = demand_scalers[item]

    fut_dem, fut_lead = [], []
    for dt in future_days:
        row = {
            "promotion": 0,
            "day_of_week": dt.dayofweek,
            "month": dt.month,
            "year": dt.year,
            "is_holiday": int(dt in korean_holidays_dt),
        }
        for c in [c for c in feature_cols if c.startswith("item_")]:
            row[c] = 1 if c == f"item_{item}" else 0

        # demand AR (from deque; log space)
        lag1 = dq[-1] if len(dq)>=1 else dq[0]
        lag7 = dq[-7] if len(dq)>=7 else dq[0]
        ma7  = float(np.mean(list(dq)[-7:])) if len(dq)>=7 else float(np.mean(dq))
        ma28 = float(np.mean(dq)) if len(dq)>=1 else float(lag1)
        row["lag_1"] = float(lag1); row["lag_7"] = float(lag7)
        row["ma_7"]  = float(ma7);  row["ma_28"] = float(ma28)

        # lead AR helpers from last history (best-effort)
        lt_hist = df.loc[mask_hist, "lead_time"].values
        row["lead_time_lag1"] = float(lt_hist[-1]) if len(lt_hist) else 0.0
        row["lead_time_ma7"]  = float(np.mean(lt_hist[-7:])) if len(lt_hist) else 0.0

        for c in feature_cols:
            if c not in row: row[c]=0
        x = feat_scaler.transform(pd.DataFrame([row])[feature_cols])[0]

        # predict residual & lead (scaled)
        y_res_s, y_l_s = model.predict(np.array([seq]), verbose=0)
        res_s = float(y_res_s.squeeze()); l_s = float(y_l_s.squeeze())

        # combine residual with scaled lag1
        lag1_s = sc_d.transform(np.array([[lag1]]))[0][0]
        d_s    = res_s + lag1_s

        # inverse
        d_log = sc_d.inverse_transform(np.array([[d_s]]))[0][0]
        qty   = max(0.0, float(np.expm1(d_log)))
        lt    = max(0.0, float(lead_scaler.inverse_transform(np.array([[l_s]]))[0][0]))

        fut_dem.append(qty); fut_lead.append(lt)

        # AR update
        dq.append(d_log)
        seq = np.vstack((seq[1:], [x]))

    # inventory simulation (round lead only for scheduling)
    inv, ss = init_inv[item], safety[item]
    level = float(inv); stockout, lt_at = None, None
    for i, q in enumerate(fut_dem):
        level -= q
        if level <= ss:
            stockout = future_days[i]
            lt_at = int(round(fut_lead[i]))
            break

    if stockout:
        order_day = stockout - timedelta(days=lt_at)
        rows.append({
            "item": item,
            "order_date": order_day.strftime("%Y-%m-%d"),
            "stockout_date": stockout.strftime("%Y-%m-%d"),
            "pred_lead_at_stockout": lt_at,
            "init_inventory": int(inv),
            "safety_stock": int(ss)
        })
    else:
        rows.append({
            "item": item,
            "order_date": "N/A (no stockout in 30d)",
            "stockout_date": "N/A",
            "pred_lead_at_stockout": None,
            "init_inventory": int(inv),
            "safety_stock": int(ss)
        })

    # ---- Timeline (1회만) ----
    fig, ax1 = plt.subplots(figsize=(14,6))
    ax1.bar(future_days, fut_dem, alpha=0.85, label="Predicted demand (units)")
    ax1.set_ylabel("Units"); ax1.set_xlabel("Date"); ax1.grid(True, axis='y', linewidth=0.3)
    ax2 = ax1.twinx()
    ax2.plot(future_days, fut_lead, linestyle='--', label="Predicted lead time (days)")
    ax2.set_ylabel("Days")
    if stockout:
        ax1.axvline(stockout, linestyle=':', linewidth=1.5, label="Stockout date")
        ax1.axvline(order_day, linestyle='-.', linewidth=1.5, label="Order date")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper right")
    plt.title(f"{item} — 30-day forecast | init={inv}, safety={ss}")
    plt.tight_layout()
    plt.savefig(f"plots_forecast/forecast_{item.replace('/','_')}.png")
    plt.close()

    # ---- (A) 수요 히스토그램 ----
    plt.figure(figsize=(10,6))
    plt.hist(fut_dem, bins=20, alpha=0.9)
    plt.title(f"{item} - Predicted Demand Distribution (Next 30 Days)")
    plt.xlabel("Quantity"); plt.ylabel("Frequency"); plt.grid(True, alpha=0.3)
    if stockout:
        plt.annotate(f"Order on {order_day.strftime('%Y-%m-%d')}",
                     xy=(0.98, 0.92), xycoords='axes fraction',
                     ha='right', va='top', bbox=dict(boxstyle="round", fc="w", ec="0.5"))
    plt.tight_layout()
    plt.savefig(f"plots_forecast/hist_demand_{item.replace('/', '_')}.png")
    plt.close()

    # ---- (B) 리드타임 히스토그램 ----
    plt.figure(figsize=(10,6))
    plt.hist(fut_lead, bins=15, alpha=0.9)
    plt.title(f"{item} - Predicted Lead-Time Distribution (Next 30 Days)")
    plt.xlabel("Days"); plt.ylabel("Frequency"); plt.grid(True, alpha=0.3)
    if stockout:
        msg = f"Order: {order_day.strftime('%Y-%m-%d')} | Stockout: {stockout.strftime('%Y-%m-%d')}"
        plt.annotate(msg, xy=(0.98, 0.92), xycoords='axes fraction',
                     ha='right', va='top', bbox=dict(boxstyle="round", fc="w", ec="0.5"))
    plt.tight_layout()
    plt.savefig(f"plots_forecast/hist_lead_{item.replace('/', '_')}.png")
    plt.close()

pd.DataFrame(rows).to_csv("predicted_order_dates_with_lead_time.csv", index=False, encoding="utf-8-sig")
print("[INFO] Saved: predicted_order_dates_with_lead_time.csv")
print("[INFO] Saved forecast plots -> plots_forecast/")
print("[DONE]")