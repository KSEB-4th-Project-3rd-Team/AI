import os, warnings, platform
from pathlib import Path
warnings.filterwarnings('ignore')

# --- Env (GPU/Matplotlib) ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')  # headless
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

# --- Reproducibility ---
np.random.seed(42); tf.random.set_seed(42)

# --- GPU memory growth ---
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Using {len(gpus)} GPU(s): {[d.name for d in gpus]}")
    except Exception as e:
        print(f"[GPU] memory_growth setup failed: {e}")
else:
    print("[GPU] No GPU detected -> CPU mode")

# --- Fonts ---
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1) Load & fill calendar
# =========================
korean_holidays = [
    # (사용하신 리스트 그대로 유지)
    date(2010,1,1),date(2010,2,13),date(2010,2,14),date(2010,2,15),date(2010,3,1),date(2010,5,5),
    # ... 중간 생략 ...
    date(2019,10,9),date(2019,12,25)
]
korean_holidays_dt = pd.to_datetime(korean_holidays)

df = pd.read_csv("preprocessed_tire_demand.csv", index_col=0, parse_dates=True)
print(f"[INFO] Raw rows: {len(df)}")

# Fill missing days per item
full_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
stitched = []
for item in df['item_id'].unique():
    t = df[df['item_id'] == item].reindex(full_days)
    t['number_sold'] = t['number_sold'].fillna(0)
    t['item_id'] = t['item_id'].fillna(item)
    t = t.fillna(method='ffill').fillna(method='bfill')
    stitched.append(t)
df = pd.concat(stitched).sort_index()
print(f"[INFO] After fill: {len(df)} rows")

# =========================
# 2) Feature engineering
# =========================
df['is_holiday'] = df.index.isin(korean_holidays_dt).astype(int)
df['number_sold_log'] = np.log1p(df['number_sold'])
if 'promotion' not in df.columns: df['promotion'] = 0
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year

# Demand lags/MAs (LOG)
grp_d = df.groupby('item_id')['number_sold_log']
df['lag_1']  = grp_d.shift(1)
df['lag_7']  = grp_d.shift(7)
df['ma_7']   = grp_d.transform(lambda s: s.rolling(7,  min_periods=1).mean())
df['ma_28']  = grp_d.transform(lambda s: s.rolling(28, min_periods=1).mean())

# Lead-time lags/MAs
if 'lead_time' not in df.columns:
    raise ValueError("Missing 'lead_time' column for lead-time dashboard.")
grp_l = df.groupby('item_id')['lead_time']
df['lead_lag_1']  = grp_l.shift(1)
df['lead_lag_7']  = grp_l.shift(7)
df['lead_ma_7']   = grp_l.transform(lambda s: s.rolling(7,  min_periods=1).mean())
df['lead_ma_28']  = grp_l.transform(lambda s: s.rolling(28, min_periods=1).mean())

for c in ['lag_1','lag_7','ma_7','ma_28','lead_lag_1','lead_lag_7','lead_ma_7','lead_ma_28']:
    df[c] = df[c].fillna(method='ffill').fillna(method='bfill')

items = df['item_id'].unique()
df_enc = pd.get_dummies(df, columns=['item_id'], prefix='item', dtype=int)

# =========================
# 3) Split & scaling
# =========================
TEST_START = "2019-01-01"
train_df = df_enc[df_enc.index < TEST_START].copy()
test_df  = df_enc[df_enc.index >= TEST_START].copy()

drop_cols = ['number_sold','number_sold_log','lead_time']
feature_cols = [c for c in df_enc.columns if c not in drop_cols]

feat_scaler = MinMaxScaler().fit(train_df[feature_cols])

# Per-item scaler: demand(LOG), lead
demand_scalers = {}
lead_scalers = {}
train_s = train_df.copy()
test_s  = test_df.copy()

for item in items:
    mtr = (train_df.get(f'item_{item}', 0) == 1)
    mte = (test_df.get(f'item_{item}', 0) == 1)

    # demand scaler (LOG)
    sc_d = MinMaxScaler()
    tr_vals_d = train_df.loc[mtr, ['number_sold_log']]
    if not tr_vals_d.empty:
        train_s.loc[mtr, 'number_sold_log_scaled'] = sc_d.fit_transform(tr_vals_d)
    if mte.sum() > 0:
        te_vals_d = test_df.loc[mte, ['number_sold_log']]
        if not te_vals_d.empty:
            test_s.loc[mte, 'number_sold_log_scaled'] = sc_d.transform(te_vals_d)
    demand_scalers[item] = sc_d

    # lead scaler
    sc_l = MinMaxScaler()
    tr_vals_l = train_df.loc[mtr, ['lead_time']]
    if not tr_vals_l.empty:
        train_s.loc[mtr, 'lead_time_scaled'] = sc_l.fit_transform(tr_vals_l)
    if mte.sum() > 0:
        te_vals_l = test_df.loc[mte, ['lead_time']]
        if not te_vals_l.empty:
            test_s.loc[mte, 'lead_time_scaled'] = sc_l.transform(te_vals_l)
    lead_scalers[item] = sc_l

train_s.dropna(subset=['number_sold_log_scaled','lead_time_scaled'], inplace=True)
test_s.dropna(subset=['number_sold_log_scaled','lead_time_scaled'],  inplace=True)

# =========================
# 4) Sequences (by ITEM)
# =========================
N_STEPS = 56
targets = ['number_sold_log_scaled','lead_time_scaled']

def build_sequences_per_item(df_feat, df_targ, which: str):
    X_list, y_list, dates_list, items_list = [], [], [], []
    for item in items:
        mask_all = (df_feat.get(f'item_{item}', 0) == 1)
        idx_all = df_feat.index[mask_all]
        idx_valid = df_targ.index.intersection(idx_all)
        if len(idx_valid) <= N_STEPS:
            print(f"[WARN] {which}: {item} not enough length ({len(idx_valid)})")
            continue
        f_item = feat_scaler.transform(df_feat.loc[idx_valid, feature_cols])
        t_item = df_targ.loc[idx_valid, targets].values
        dates_item = idx_valid
        for i in range(len(dates_item) - N_STEPS):
            X_list.append(f_item[i:i+N_STEPS])
            y_list.append(t_item[i+N_STEPS])
            dates_list.append(dates_item[i+N_STEPS])
            items_list.append(item)
    if len(X_list)==0:
        raise RuntimeError(f"No sequences built for {which}.")
    X = np.asarray(X_list, np.float32)
    y = np.asarray(y_list, np.float32)
    return X, y, np.array(dates_list), np.array(items_list, dtype=object)

X_train, y_train, dates_tr_meta, items_tr_meta = build_sequences_per_item(train_df, train_s, "train")
X_test,  y_test,  dates_te_meta, items_te_meta  = build_sequences_per_item(test_df,  test_s,  "test")

print(f"[INFO] Train: X={X_train.shape}, y={y_train.shape}")
print(f"[INFO] Test : X={X_test.shape},  y={y_test.shape}")

# =========================
# 5) Model (demand: point, lead: P10/P50/P90)
# =========================
def pinball_loss(q: float):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q*e, (q-1.0)*e))
    return loss

inp = layers.Input(shape=(N_STEPS, X_train.shape[2]))
x = layers.LSTM(128, return_sequences=True)(inp)
x = layers.Dropout(0.15)(x)
x = layers.LSTM(64, return_sequences=False)(x)
x = layers.Dropout(0.15)(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)

# Demand (point)
h_d = layers.Dense(32, activation="relu")(x)
out_d = layers.Dense(1, name="demand")(h_d)

# Lead-time quantiles
h_l = layers.Dense(32, activation="relu")(x)
lead_p10 = layers.Dense(1, name="lead_p10")(h_l)
lead_p50 = layers.Dense(1, name="lead_p50")(h_l)
lead_p90 = layers.Dense(1, name="lead_p90")(h_l)

model = Model(inp, [out_d, lead_p10, lead_p50, lead_p90])

try:
    opt = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0)
except AttributeError:
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

model.compile(
    optimizer=opt,
    loss={
        "demand": tf.keras.losses.Huber(),
        "lead_p10": pinball_loss(0.10),
        "lead_p50": pinball_loss(0.50),  # 중앙값(=MAE와 동일 성질)
        "lead_p90": pinball_loss(0.90),
    },
    loss_weights={
        "demand": 0.4,
        "lead_p10": 0.1,
        "lead_p50": 0.4,
        "lead_p90": 0.1
    },
    metrics={"demand": "mae", "lead_p50": "mae"}
)

ckpt = "best_multi_output_lstm_qlead.keras"
cbs = [
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=1e-6, verbose=1, min_delta=1e-4),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1, min_delta=1e-4),
    ModelCheckpoint(filepath=ckpt, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
]

print("\n[INFO] --- Training ---")
model.fit(
    X_train,
    {
        "demand":   y_train[:,0],
        "lead_p10": y_train[:,1],
        "lead_p50": y_train[:,1],
        "lead_p90": y_train[:,1],
    },
    epochs=60, batch_size=64, shuffle=True,
    validation_data=(
        X_test,
        {
            "demand":   y_test[:,0],
            "lead_p10": y_test[:,1],
            "lead_p50": y_test[:,1],
            "lead_p90": y_test[:,1],
        }
    ),
    callbacks=cbs, verbose=2
)
print("[INFO] --- Training done ---")

# =========================
# 6) Evaluation (NO rounding for plots)
# =========================
print("\n[INFO] --- Evaluation ---")
pred_d, pred_l10, pred_l50, pred_l90 = model.predict(X_test, verbose=0)
pred_d   = pred_d.reshape(-1)
pred_l10 = pred_l10.reshape(-1)
pred_l50 = pred_l50.reshape(-1)
pred_l90 = pred_l90.reshape(-1)

Lmin = min(len(pred_d), len(pred_l10), len(pred_l50), len(pred_l90), len(y_test))
pred_d, pred_l10, pred_l50, pred_l90, y_eval = (
    pred_d[:Lmin], pred_l10[:Lmin], pred_l50[:Lmin], pred_l90[:Lmin], y_test[:Lmin]
)
dates_eval = pd.to_datetime(dates_te_meta[:Lmin])
items_eval = items_te_meta[:Lmin]

res = pd.DataFrame({
    "Date": dates_eval,
    "item_id": items_eval,
    "Actual_Demand_Log_Scaled": y_eval[:,0],
    "Pred_Demand_Log_Scaled": pred_d,
    "Actual_Lead_Scaled": y_eval[:,1],
    "Pred_Lead_Scaled_p10": pred_l10,
    "Pred_Lead_Scaled_p50": pred_l50,
    "Pred_Lead_Scaled_p90": pred_l90,
})

# inverse transforms
res[["Actual_Demand","Pred_Demand","Actual_Lead","Pred_Lead_p10","Pred_Lead_p50","Pred_Lead_p90"]] = np.nan
for item in items:
    m = (res["item_id"] == item)
    if m.sum()==0: continue

    sc_d = demand_scalers[item]
    act_log_d = sc_d.inverse_transform(res.loc[m, [["Actual_Demand_Log_Scaled"]]])
    pre_log_d = sc_d.inverse_transform(res.loc[m, [["Pred_Demand_Log_Scaled"]]])
    res.loc[m, "Actual_Demand"] = np.expm1(act_log_d).ravel()
    res.loc[m, "Pred_Demand"]   = np.expm1(pre_log_d).ravel()

    sc_l = lead_scalers[item]
    res.loc[m, "Actual_Lead"]    = sc_l.inverse_transform(res.loc[m, [["Actual_Lead_Scaled"]]]).ravel()
    res.loc[m, "Pred_Lead_p10"]  = sc_l.inverse_transform(res.loc[m, [["Pred_Lead_Scaled_p10"]]]).ravel()
    res.loc[m, "Pred_Lead_p50"]  = sc_l.inverse_transform(res.loc[m, [["Pred_Lead_Scaled_p50"]]]).ravel()
    res.loc[m, "Pred_Lead_p90"]  = sc_l.inverse_transform(res.loc[m, [["Pred_Lead_Scaled_p90"]]]).ravel()

res["Pred_Demand"] = res["Pred_Demand"].clip(lower=0)
res.dropna(inplace=True)

# metrics
def mape(a,p,eps=1e-10): return float(np.mean(np.abs((a-p)/(a+eps)))*100)
def smape(a,p,eps=1e-10): return float(np.mean(np.abs(p-a)/((np.abs(a)+np.abs(p))/2+eps))*100)

coverage = np.mean((res["Actual_Lead"] >= res["Pred_Lead_p10"]) & (res["Actual_Lead"] <= res["Pred_Lead_p90"])) * 100.0

metrics = pd.DataFrame({
    "Target":["Demand","Demand","Demand","Demand","Demand","Lead(P50)","Lead(P50)","Lead(P50)","Lead interval"],
    "Metric":["MAE","RMSE","R2 Score","MAPE (%)","SMAPE (%)","MAE","RMSE","R2 Score","P10–P90 coverage (%)"],
    "Value":[
        mean_absolute_error(res["Actual_Demand"], res["Pred_Demand"]),
        np.sqrt(mean_squared_error(res["Actual_Demand"], res["Pred_Demand"])),
        r2_score(res["Actual_Demand"], res["Pred_Demand"]),
        mape(res["Actual_Demand"], res["Pred_Demand"]),
        smape(res["Actual_Demand"], res["Pred_Demand"]),
        mean_absolute_error(res["Actual_Lead"], res["Pred_Lead_p50"]),
        np.sqrt(mean_squared_error(res["Actual_Lead"], res["Pred_Lead_p50"])),
        r2_score(res["Actual_Lead"], res["Pred_Lead_p50"]),
        coverage
    ]
})
metrics.to_csv("model_evaluation_metrics.csv", index=False, encoding="utf-8-sig")
res.to_csv("test_predictions_expanded_quantile.csv", index=False, encoding="utf-8-sig")
print("[INFO] Saved: model_evaluation_metrics.csv, test_predictions_expanded_quantile.csv")

# plots
os.makedirs("plots_eval", exist_ok=True)
for item in items:
    d = res[res["item_id"]==item].sort_values("Date")
    if d.empty: continue

    # Demand
    r2d = r2_score(d["Actual_Demand"], d["Pred_Demand"])
    plt.figure(figsize=(14,6))
    plt.plot(d["Date"], d["Actual_Demand"], label="Actual demand", alpha=0.85)
    plt.plot(d["Date"], d["Pred_Demand"], label="Pred demand", alpha=0.85, linestyle="--")
    plt.title(f"{item} — Demand (R² = {r2d:.3f})")
    plt.xlabel("Date"); plt.ylabel("Units"); plt.legend(); plt.grid(True, linewidth=0.3)
    plt.tight_layout(); plt.savefig(f"plots_eval/demand_{item.replace('/','_')}.png"); plt.close()

    # Lead (P10/P50/P90)
    r2l = r2_score(d["Actual_Lead"], d["Pred_Lead_p50"])
    plt.figure(figsize=(14,6))
    plt.plot(d["Date"], d["Actual_Lead"], label="Actual lead", alpha=0.85)
    plt.plot(d["Date"], d["Pred_Lead_p50"], label="Pred lead P50", alpha=0.85, linestyle="--")
    # 구간 음영
    plt.fill_between(d["Date"], d["Pred_Lead_p10"], d["Pred_Lead_p90"], alpha=0.15, label="Lead P10–P90")
    plt.title(f"{item} — Lead time (R²@P50 = {r2l:.3f})")
    plt.xlabel("Date"); plt.ylabel("Days"); plt.legend(); plt.grid(True, linewidth=0.3)
    plt.tight_layout(); plt.savefig(f"plots_eval/lead_{item.replace('/','_')}.png"); plt.close()

print("[INFO] Saved evaluation plots -> plots_eval/")

# =========================
# 7) 30-day forecast + order date (use P90) + histogram
# =========================
print("\n[INFO] --- 30-day forecast & order date (quantile) ---")
INIT_INV_MIN   = 50
SAFETY_MIN     = 10
HORIZON        = 30

last_date = df.index.max()
future_days = pd.date_range(start=last_date + timedelta(days=1), periods=HORIZON, freq='D')

# initial inventory/safety by recent 30-day avg
init_inv = {}; safety = {}
last30 = df.loc[df.index.max() - timedelta(days=29):]
for item in items:
    m = (last30['item_id']==item)
    mean_recent = float(last30.loc[m, 'number_sold'].mean() or 0.0)
    init_inv[item] = max(INIT_INV_MIN, int(mean_recent*14))
    safety[item]   = max(SAFETY_MIN,   int(mean_recent*7))

rows = []
os.makedirs("plots_forecast", exist_ok=True)

for item in items:
    mask_hist = (df_enc.get(f"item_{item}", 0) == 1)
    hist_feats = feat_scaler.transform(df_enc.loc[mask_hist, feature_cols])
    if len(hist_feats) < N_STEPS:
        print(f"[WARN] {item}: not enough history ({len(hist_feats)})")
        continue
    seq = hist_feats[-N_STEPS:].copy()

    # deque for autoregressive updates
    logs_hist = df.loc[mask_hist, "number_sold_log"].values
    lead_hist = df.loc[mask_hist, "lead_time"].values
    if len(logs_hist)==0 or len(lead_hist)==0:
        print(f"[WARN] {item}: insufficient history")
        continue
    dq_d = deque(list(logs_hist[-28:]),  maxlen=28)
    dq_l = deque(list(lead_hist[-28:]),  maxlen=28)

    fut_dem, fut_l10, fut_l50, fut_l90 = [], [], [], []
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

        # demand lag/MA (LOG)
        lag1_d = dq_d[-1] if len(dq_d)>=1 else dq_d[0]
        lag7_d = dq_d[-7] if len(dq_d)>=7 else dq_d[0]
        ma7_d  = float(np.mean(list(dq_d)[-7:])) if len(dq_d)>=7 else float(np.mean(dq_d))
        ma28_d = float(np.mean(dq_d)) if len(dq_d)>=1 else float(lag1_d)
        row["lag_1"] = float(lag1_d); row["lag_7"] = float(lag7_d)
        row["ma_7"]  = float(ma7_d);  row["ma_28"] = float(ma28_d)

        # lead lag/MA (CONT)
        lag1_l = dq_l[-1] if len(dq_l)>=1 else dq_l[0]
        lag7_l = dq_l[-7] if len(dq_l)>=7 else dq_l[0]
        ma7_l  = float(np.mean(list(dq_l)[-7:])) if len(dq_l)>=7 else float(np.mean(dq_l))
        ma28_l = float(np.mean(dq_l)) if len(dq_l)>=1 else float(lag1_l)
        row["lead_lag_1"] = float(lag1_l); row["lead_lag_7"] = float(lag7_l)
        row["lead_ma_7"]  = float(ma7_l);  row["lead_ma_28"] = float(ma28_l)

        for c in feature_cols:
            if c not in row: row[c]=0
        x = feat_scaler.transform(pd.DataFrame([row])[feature_cols])[0]

        # predict 1-step (demand + lead quantiles)
        pred = model.predict(np.array([seq]), verbose=0)
        y_d_s, y_l10_s, y_l50_s, y_l90_s = [float(p.squeeze()) for p in pred]

        # inverse transform
        d_log = demand_scalers[item].inverse_transform([[y_d_s]])[0][0]
        qty   = max(0.0, np.expm1(d_log))

        l10   = max(0.0, float(lead_scalers[item].inverse_transform([[y_l10_s]])[0][0]))
        l50   = max(0.0, float(lead_scalers[item].inverse_transform([[y_l50_s]])[0][0]))
        l90   = max(0.0, float(lead_scalers[item].inverse_transform([[y_l90_s]])[0][0]))

        fut_dem.append(qty); fut_l10.append(l10); fut_l50.append(l50); fut_l90.append(l90)

        # autoregressive update (lead는 P50을 관성값으로 사용)
        dq_d.append(d_log)
        dq_l.append(l50)
        seq = np.vstack((seq[1:], [x]))

    # 발주일: 보수적(P90) 리드타임 사용
    inv = init_inv[item]; ss = safety[item]
    stockout, lt_at = None, None
    level = float(inv)
    for i, dqty in enumerate(fut_dem):
        level -= dqty
        if level <= ss:
            stockout = future_days[i]
            lt_at = int(np.ceil(fut_l90[i]))  # 보수적
            break

    if stockout:
        order_day = stockout - timedelta(days=lt_at)
        rows.append({
            "item": item,
            "order_date": order_day.strftime("%Y-%m-%d"),
            "stockout_date": stockout.strftime("%Y-%m-%d"),
            "pred_lead_p50_at_stockout": float(fut_l50[i]),
            "pred_lead_p90_at_stockout": float(fut_l90[i]),
            "init_inventory": int(inv),
            "safety_stock": int(ss)
        })
    else:
        rows.append({
            "item": item,
            "order_date": "N/A (no stockout in 30d)",
            "stockout_date": "N/A",
            "pred_lead_p50_at_stockout": None,
            "pred_lead_p90_at_stockout": None,
            "init_inventory": int(inv),
            "safety_stock": int(ss)
        })

    # ---- Histogram + lead quantile bands ----
    fig, ax1 = plt.subplots(figsize=(14,6))
    ax1.bar(future_days, fut_dem, alpha=0.85, label="Predicted demand (units)")
    ax1.set_ylabel("Units"); ax1.set_xlabel("Date"); ax1.grid(True, axis='y', linewidth=0.3)

    ax2 = ax1.twinx()
    ax2.plot(future_days, fut_l50, linestyle='--', label="Lead P50 (days)")
    # 밴드
    ax2.fill_between(future_days, fut_l10, fut_l90, alpha=0.15, label="Lead P10–P90")

    ax2.set_ylabel("Days")

    if stockout:
        ax1.axvline(stockout, linestyle=':', linewidth=1.5, label="Stockout date")
        ax1.axvline(order_day, linestyle='-.', linewidth=1.5, label="Order date (P90)")

    title_extra = f" | init={inv}, safety={ss}"
    plt.title(f"{item} — 30-day forecast{title_extra}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"plots_forecast/forecast_{item.replace('/','_')}.png")
    plt.close()

pd.DataFrame(rows).to_csv("predicted_order_dates_with_lead_time_quantile.csv", index=False, encoding="utf-8-sig")
print("[INFO] Saved: predicted_order_dates_with_lead_time_quantile.csv")
print("[INFO] Saved forecast plots -> plots_forecast/")
print("[DONE]")