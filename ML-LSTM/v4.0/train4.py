# -*- coding: utf-8 -*-
# train4.py
# Final v4: MA7-residual demand + lead residual + time-based validation
# + AR-updated lead in rollout + arrival & inventory line + axis de-exaggeration
# + accuracy histograms + baselines + display-year shift (2015–2025)
'''
검증셋 분리(시간기반): VAL_START="2018-07-01" 이후는 검증, 테스트는 그대로 2019년
수요 타깃 = MA7 기준선 잔차(lag1 → ma7로 교체). 복원 시 pred = resid + ma7_scaled@target
리드타임 타깃 = 잔차(lead_resid = lead_scaled - lead_lag1_scaled)
모델 손실 = Huber(수요/리드타임 둘 다) + 기존 AdamW/정규화 유지
30일 시뮬에서 리드타임도 예측값으로 AR 업데이트(분산 살아남)
타임라인 플롯에 도착일(Arrival), 잔여재고 라인 추가, 오른쪽 축 과장 방지(ax2.set_ylim 패딩)
주문일 P50/P90 병기, 리드타임 히스토그램에 분위선(P10/P50/P90) 표시
베이스라인 비교(수요: Lag-1, MA7 / 리드타임: Lag-1) 메트릭을 함께 CSV로 저장
평가/예측 플롯은 표시용 연도 이동(2015–2025) 유지
'''
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
from pandas.tseries.offsets import DateOffset

# ---------------- Display-only settings ----------------
DISPLAY_SHIFT_YEARS = 5                                # 2010→2015, 2019→2024
DISPLAY_FORECAST_START = pd.Timestamp("2025-09-01")    # 30일 창 표시 시작일

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
# (A) 2010–2014 (학습 안정용)
korean_holidays_2010_2014 = [
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
    date(2014,10,3),date(2014,10,9),date(2014,12,25)
]
# (B) 2015–2025 실제 공휴일
korean_holidays_2015_2025 = [
    # 2015
    date(2015,1,1),date(2015,2,18),date(2015,2,19),date(2015,2,20),
    date(2015,3,1),date(2015,5,5),date(2015,5,25),date(2015,6,6),
    date(2015,8,14),date(2015,8,15),date(2015,9,26),date(2015,9,27),
    date(2015,9,28),date(2015,9,29),date(2015,10,3),date(2015,10,9),date(2015,12,25),
    # 2016
    date(2016,1,1),date(2016,2,7),date(2016,2,8),date(2016,2,9),date(2016,2,10),
    date(2016,3,1),date(2016,4,13),date(2016,5,5),date(2016,5,6),date(2016,5,14),
    date(2016,6,6),date(2016,8,15),date(2016,9,14),date(2016,9,15),date(2016,9,16),date(2016,10,3),
    date(2016,12,25),
    # 2017
    date(2017,1,1),date(2017,1,27),date(2017,1,28),date(2017,1,29),date(2017,1,30),
    date(2017,3,1),date(2017,5,1),date(2017,5,3),date(2017,5,5),date(2017,5,9),
    date(2017,6,6),date(2017,8,15),date(2017,10,2),date(2017,10,3),date(2017,10,4),
    date(2017,10,5),date(2017,10,6),date(2017,10,9),date(2017,12,20),date(2017,12,25),
    # 2018
    date(2018,1,1),date(2018,2,15),date(2018,2,16),date(2018,2,17),
    date(2018,3,1),date(2018,5,5),date(2018,5,7),date(2018,5,22),
    date(2018,6,6),date(2018,6,13),date(2018,8,15),date(2018,9,23),date(2018,9,24),
    date(2018,9,25),date(2018,9,26),date(2018,10,3),date(2018,10,9),date(2018,12,25),
    # 2019
    date(2019,1,1),date(2019,2,4),date(2019,2,5),date(2019,2,6),
    date(2019,3,1),date(2019,5,5),date(2019,5,6),date(2019,5,12),
    date(2019,6,6),date(2019,8,15),date(2019,9,12),date(2019,9,13),date(2019,9,14),
    date(2019,10,3),date(2019,10,9),date(2019,12,25),
    # 2020
    date(2020,1,1),date(2020,1,24),date(2020,1,25),date(2020,1,26),date(2020,1,27),
    date(2020,3,1),date(2020,4,15),date(2020,4,30),date(2020,5,5),date(2020,6,6),
    date(2020,8,15),date(2020,8,17),date(2020,9,30),date(2020,10,1),date(2020,10,2),date(2020,10,3),
    date(2020,10,9),date(2020,12,25),
    # 2021
    date(2021,1,1),date(2021,2,11),date(2021,2,12),date(2021,2,13),
    date(2021,3,1),date(2021,5,5),date(2021,5,19),date(2021,6,6),
    date(2021,8,15),date(2021,8,16),date(2021,9,20),date(2021,9,21),date(2021,9,22),
    date(2021,10,3),date(2021,10,4),date(2021,10,9),date(2021,10,11),date(2021,12,25),
    # 2022
    date(2022,1,1),date(2022,1,31),date(2022,2,1),date(2022,2,2),
    date(2022,3,1),date(2022,3,9),date(2022,5,5),date(2022,5,8),
    date(2022,6,1),date(2022,6,6),date(2022,8,15),date(2022,9,9),date(2022,9,10),date(2022,9,11),date(2022,9,12),
    date(2022,10,3),date(2022,10,9),date(2022,10,10),date(2022,12,25),
    # 2023
    date(2023,1,1),date(2023,1,21),date(2023,1,22),date(2023,1,23),date(2023,1,24),
    date(2023,3,1),date(2023,5,5),date(2023,5,27),date(2023,5,29),date(2023,6,6),
    date(2023,8,15),date(2023,9,28),date(2023,9,29),date(2023,9,30),date(2023,10,2),date(2023,10,3),
    date(2023,10,9),date(2023,12,25),
    # 2024
    date(2024,1,1),date(2024,2,9),date(2024,2,10),date(2024,2,11),date(2024,2,12),
    date(2024,3,1),date(2024,4,10),date(2024,5,5),date(2024,5,6),date(2024,5,15),
    date(2024,6,6),date(2024,8,15),date(2024,9,16),date(2024,9,17),date(2024,9,18),
    date(2024,10,3),date(2024,10,9),date(2024,12,25),
    # 2025
    date(2025,1,1),date(2025,1,28),date(2025,1,29),date(2025,1,30),
    date(2025,3,1),date(2025,3,3),date(2025,5,5),date(2025,5,6),date(2025,6,6),
    date(2025,8,15),date(2025,10,3),date(2025,10,6),date(2025,10,7),date(2025,10,8),date(2025,10,9),date(2025,12,25),
]
korean_holidays = korean_holidays_2010_2014 + korean_holidays_2015_2025
korean_holidays_dt = pd.to_datetime(korean_holidays)

df = pd.read_csv("preprocessed_tire_demand.csv", index_col=0, parse_dates=True)
print(f"[INFO] Raw rows: {len(df)}")

# Reindex per item (daily)
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
    df[c] = df.groupby('item_id')[c].transform(lambda s: s.bfill())

# Baseline series (for metrics)
g_num  = df.groupby('item_id')['number_sold']
g_lead = df.groupby('item_id')['lead_time']

# 하루 전 실적
df['baseline_d_lag1'] = g_num.shift(1)

# 직전 7일 평균(타깃 날짜 포함 금지 → shift(1) 후 rolling)
df['baseline_d_ma7'] = g_num.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())

# 리드타임 하루 전 값
df['baseline_l_lag1'] = g_lead.shift(1)

# 초반부 NaN 보정(아이템별 앞부분만 bfill) 
df[['baseline_d_lag1','baseline_d_ma7','baseline_l_lag1']] = (
    df.groupby('item_id')[['baseline_d_lag1','baseline_d_ma7','baseline_l_lag1']]
      .transform(lambda s: s.bfill())
)
# ---------------- 3) Split & scaling ----------------
items = df['item_id'].unique()
df_enc = pd.get_dummies(df, columns=['item_id'], prefix='item', dtype=int)

TEST_START = "2019-01-01"
VAL_START  = "2018-07-01"   # 검증 시작 (train 기간 후반)
train_df = df_enc[df_enc.index < VAL_START].copy()
val_df   = df_enc[(df_enc.index >= VAL_START) & (df_enc.index < TEST_START)].copy()
test_df  = df_enc[df_enc.index >= TEST_START].copy()

exclude_cols = {
    'number_sold','number_sold_log','lead_time',
    'baseline_d_lag1','baseline_d_ma7','baseline_l_lag1'
}
feature_cols = [c for c in df_enc.columns if c not in exclude_cols]

# global feature scaler (fit on TRAIN only)
feat_scaler = MinMaxScaler().fit(train_df[feature_cols])

# per-item scaler for log demand (fit per item on TRAIN only)
train_s = train_df.copy(); val_s = val_df.copy(); test_s = test_df.copy()
demand_scalers = {}
for col in ['number_sold_log_scaled','ma7_scaled','lag1_scaled',
            'lead_time_scaled','lead_lag1_scaled','demand_resid','lead_resid']:
    for d in (train_s, val_s, test_s):
        d[col] = np.nan

# lead-time scaler (global, fit on TRAIN only)
lead_scaler = MinMaxScaler().fit(train_df[['lead_time']].values)
for d in (train_s, val_s, test_s):
    d['lead_time_scaled'] = lead_scaler.transform(d[['lead_time']].values).ravel()

for item in items:
    mtr = (train_df.get(f'item_{item}', 0) == 1)
    mva = (val_df.get(f'item_{item}',   0) == 1)
    mte = (test_df.get(f'item_{item}',  0) == 1)
    sc_d = MinMaxScaler()

    tr_log = train_df.loc[mtr, ['number_sold_log']]
    if not tr_log.empty:
        train_s.loc[mtr, 'number_sold_log_scaled'] = sc_d.fit_transform(tr_log)
        train_s.loc[mtr, 'ma7_scaled'] = sc_d.transform(train_df.loc[mtr, ['ma_7']].values).ravel()
        train_s.loc[mtr, 'lag1_scaled'] = sc_d.transform(train_df.loc[mtr, ['lag_1']].values).ravel()
    if mva.sum() > 0:
        va_log = val_df.loc[mva, ['number_sold_log']]
        if not va_log.empty:
            val_s.loc[mva, 'number_sold_log_scaled'] = sc_d.transform(va_log)
            val_s.loc[mva, 'ma7_scaled'] = sc_d.transform(val_df.loc[mva, ['ma_7']].values).ravel()
            val_s.loc[mva, 'lag1_scaled'] = sc_d.transform(val_df.loc[mva, ['lag_1']].values).ravel()
    if mte.sum() > 0:
        te_log = test_df.loc[mte, ['number_sold_log']]
        if not te_log.empty:
            test_s.loc[mte, 'number_sold_log_scaled'] = sc_d.transform(te_log)
            test_s.loc[mte, 'ma7_scaled'] = sc_d.transform(test_df.loc[mte, ['ma_7']].values).ravel()
            test_s.loc[mte, 'lag1_scaled'] = sc_d.transform(test_df.loc[mte, ['lag_1']].values).ravel()

    demand_scalers[item] = sc_d

# residual targets
for d in (train_s, val_s, test_s):
    d['demand_resid'] = d['number_sold_log_scaled'] - d['ma7_scaled']
    d['lead_lag1_scaled'] = lead_scaler.transform(d[['lead_time_lag1']].values).ravel()
    d['lead_resid'] = d['lead_time_scaled'] - d['lead_lag1_scaled']

# drop rows without targets
for name, d in [('train',train_s), ('val',val_s), ('test',test_s)]:
    before = len(d)
    d.dropna(subset=['demand_resid','lead_resid'], inplace=True)
    print(f"[INFO] {name}: drop NaN rows {before - len(d)}")

# ---------------- 4) Sequences (per item; return MA7@target & LEAD_LAG1@target) ----------------
N_STEPS = 56
TARGETS = ['demand_resid','lead_resid']

def build_sequences_per_item(df_feat, df_targ, which: str):
    X, y, dates, item_ids, ma7_targets, ltlag1_targets = [], [], [], [], [], []
    for item in items:
        mask_feat = (df_feat.get(f'item_{item}',0) == 1)
        idx_all = df_feat.index[mask_feat]
        idx_valid = df_targ.index.intersection(idx_all)
        if len(idx_valid) <= N_STEPS:
            print(f"[WARN] {which}: {item} not enough length ({len(idx_valid)})")
            continue

        feat = feat_scaler.transform(df_feat.loc[idx_valid, feature_cols])
        targ = df_targ.loc[idx_valid, TARGETS].values
        ma7t = df_targ.loc[idx_valid, 'ma7_scaled'].values
        lt1t = df_targ.loc[idx_valid, 'lead_lag1_scaled'].values
        dts  = idx_valid

        for i in range(len(dts) - N_STEPS):
            X.append(feat[i:i+N_STEPS])
            y.append(targ[i+N_STEPS])
            dates.append(dts[i+N_STEPS])
            item_ids.append(item)
            ma7_targets.append(ma7t[i+N_STEPS])
            ltlag1_targets.append(lt1t[i+N_STEPS])

    if len(X)==0:
        raise ValueError(f"No sequences built for {which}. Check data/steps.")
    return (np.asarray(X, np.float32),
            np.asarray(y, np.float32),
            np.array(dates),
            np.array(item_ids, dtype=object),
            np.asarray(ma7_targets, np.float32),
            np.asarray(ltlag1_targets, np.float32))

X_tr, y_tr, tr_dates, tr_items, ma7_tr, lt1_tr = build_sequences_per_item(train_df, train_s, "train")
X_val, y_val, va_dates, va_items, ma7_va, lt1_va = build_sequences_per_item(val_df,   val_s,   "val")
X_test, y_test, te_dates, te_items, ma7_te, lt1_te = build_sequences_per_item(test_df, test_s, "test")

print(f"[INFO] Train: X={X_tr.shape}, y={y_tr.shape}")
print(f"[INFO] Val  : X={X_val.shape}, y={y_val.shape}")
print(f"[INFO] Test : X={X_test.shape}, y={y_test.shape}")

# ---------------- 5) Model ----------------
inp = layers.Input(shape=(N_STEPS, X_tr.shape[2]))
x = layers.LSTM(128, return_sequences=True)(inp)
x = layers.Dropout(0.15)(x)
x = layers.LSTM(64, return_sequences=False)(x)
x = layers.Dropout(0.15)(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)

h_d = layers.Dense(32, activation="relu")(x); out_d = layers.Dense(1, name="demand")(h_d)  # demand residual
h_l = layers.Dense(32, activation="relu")(x); out_l = layers.Dense(1, name="lead")(h_l)    # lead residual

model = Model(inp, [out_d, out_l])

huber = tf.keras.losses.Huber(delta=0.1)  # robust to spikes
try:
    opt = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4, clipnorm=1.0)
except AttributeError:
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

model.compile(
    optimizer=opt,
    loss={"demand": huber, "lead": huber},
    loss_weights={"demand": 0.8, "lead": 0.2},
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
    X_tr, {"demand": y_tr[:,0], "lead": y_tr[:,1]},
    epochs=60, batch_size=64, shuffle=True,
    validation_data=(X_val, {"demand": y_val[:,0], "lead": y_val[:,1]}),
    callbacks=cbs, verbose=2
)
print("[INFO] --- Training done ---")

# ---------------- 6) Evaluation ----------------
print("\n[INFO] --- Evaluation ---")
pred_resid_d, pred_resid_l = model.predict(X_test, verbose=0)
pred_resid_d = pred_resid_d.reshape(-1)
pred_resid_l = pred_resid_l.reshape(-1)

L = min(len(pred_resid_d), len(pred_resid_l), len(y_test), len(ma7_te), len(lt1_te))
pred_resid_d = pred_resid_d[:L]; pred_resid_l = pred_resid_l[:L]; y_eval = y_test[:L]
dates_eval = pd.to_datetime(te_dates[:L]); items_eval = te_items[:L]
ma7_at_target = ma7_te[:L]; lt1_at_target = lt1_te[:L]

# Restore scaled targets from residuals
pred_d_scaled = pred_resid_d + ma7_at_target
act_d_scaled  = y_eval[:,0] + ma7_at_target

pred_l_scaled = pred_resid_l + lt1_at_target
act_l_scaled  = y_eval[:,1] + lt1_at_target

res = pd.DataFrame({
    "Date": dates_eval,
    "item_id": items_eval,
    "Actual_Demand_Log_Scaled": act_d_scaled,
    "Predicted_Demand_Log_Scaled": pred_d_scaled,
    "Actual_LeadTime_Scaled": act_l_scaled,
    "Predicted_LeadTime_Scaled": pred_l_scaled,
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

res["Actual_LeadTime"]    = lead_scaler.inverse_transform(res[["Actual_LeadTime_Scaled"]].values).ravel()
res["Predicted_LeadTime"] = lead_scaler.inverse_transform(res[["Predicted_LeadTime_Scaled"]].values).ravel()
# clip
res["Predicted_Demand"]   = res["Predicted_Demand"].clip(lower=0)
res["Predicted_LeadTime"] = res["Predicted_LeadTime"].clip(lower=0)

# ---- Baselines (merge from df) ----
df_for_merge = df.reset_index().rename(columns={'index':'Date'})
base_cols = ["Date","item_id","baseline_d_lag1","baseline_d_ma7","baseline_l_lag1"]
res = res.merge(df_for_merge[base_cols], on=["Date","item_id"], how="left")

# metrics helpers
def mape(a,p,eps=1e-10): return float(np.mean(np.abs((a-p)/(a+eps)))*100)
def smape(a,p,eps=1e-10): return float(np.mean(np.abs(p-a)/((np.abs(a)+np.abs(p))/2+eps))*100)

# overall metrics
metrics_rows = []
def add_metrics(tag, actual, pred):
    metrics_rows.extend([
        [tag,"MAE", mean_absolute_error(actual, pred)],
        [tag,"RMSE", np.sqrt(mean_squared_error(actual, pred))],
        [tag,"R2 Score", r2_score(actual, pred)],
        [tag,"MAPE (%)", mape(actual, pred)],
        [tag,"SMAPE (%)", smape(actual, pred)],
    ])

add_metrics("Demand(Model)", res["Actual_Demand"], res["Predicted_Demand"])
add_metrics("Demand(Baseline_Lag1)", res["Actual_Demand"], res["baseline_d_lag1"])
add_metrics("Demand(Baseline_MA7)",  res["Actual_Demand"], res["baseline_d_ma7"])
add_metrics("Lead(Model)",   res["Actual_LeadTime"], res["Predicted_LeadTime"])
add_metrics("Lead(Baseline_Lag1)", res["Actual_LeadTime"], res["baseline_l_lag1"])

metrics = pd.DataFrame(metrics_rows, columns=["Target","Metric","Value"])
metrics.to_csv("model_evaluation_metrics.csv", index=False, encoding="utf-8-sig")
res.to_csv("test_predictions_expanded.csv", index=False, encoding="utf-8-sig")
print("[INFO] Saved: model_evaluation_metrics.csv, test_predictions_expanded.csv")

# ---------------- 6-1) Accuracy histograms on TEST period ----------------
os.makedirs("plots_eval_hist", exist_ok=True)
def safe_name(s): return str(s).replace('/', '_')

all_residual, all_ape = [], []
for item in items:
    d = res[res["item_id"] == item].sort_values("Date")
    if d.empty: continue
    residual = (d["Predicted_Demand"] - d["Actual_Demand"]).to_numpy()
    ape = 100.0 * np.abs(residual) / (d["Actual_Demand"].to_numpy() + 1e-10)

    plt.figure(figsize=(10,6))
    plt.hist(residual, bins=40, alpha=0.9)
    plt.title(f"{item} — Residuals on TEST (Pred - Actual)")
    plt.xlabel("Units"); plt.ylabel("Frequency"); plt.grid(True, alpha=0.3)
    plt.annotate(f"mean={np.mean(residual):.2f}\nstd={np.std(residual):.2f}",
                 xy=(0.98,0.92), xycoords="axes fraction", ha="right", va="top",
                 bbox=dict(boxstyle="round", fc="w", ec="0.5"))
    plt.tight_layout()
    plt.savefig(f"plots_eval_hist/residual_hist_{safe_name(item)}.png"); plt.close()

    plt.figure(figsize=(10,6))
    clipped_ape = ape[np.isfinite(ape)]
    plt.hist(clipped_ape, bins=np.arange(0, 101, 5), alpha=0.9)
    plt.title(f"{item} — APE% on TEST")
    plt.xlabel("APE (%)"); plt.ylabel("Frequency"); plt.grid(True, alpha=0.3)
    plt.annotate(f"median={np.median(clipped_ape):.1f}%\nP90={np.percentile(clipped_ape,90):.1f}%",
                 xy=(0.98,0.92), xycoords="axes fraction", ha="right", va="top",
                 bbox=dict(boxstyle="round", fc="w", ec="0.5"))
    plt.tight_layout()
    plt.savefig(f"plots_eval_hist/ape_hist_{safe_name(item)}.png"); plt.close()

    all_residual.append(residual); all_ape.append(clipped_ape)

if all_residual:
    all_residual = np.concatenate(all_residual)
    all_ape = np.concatenate(all_ape)
    plt.figure(figsize=(10,6)); plt.hist(all_residual, bins=60, alpha=0.9)
    plt.title("ALL ITEMS — Residuals on TEST"); plt.xlabel("Units"); plt.ylabel("Frequency"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("plots_eval_hist/residual_hist_ALL.png"); plt.close()
    plt.figure(figsize=(10,6)); plt.hist(all_ape, bins=np.arange(0,101,2), alpha=0.9)
    plt.title("ALL ITEMS — APE% on TEST"); plt.xlabel("APE (%)"); plt.ylabel("Frequency"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("plots_eval_hist/ape_hist_ALL.png"); plt.close()

# ---------------- 7) Plots (evaluation, DISPLAY shift) ----------------
os.makedirs("plots_eval", exist_ok=True)
for item in items:
    d = res[res["item_id"]==item].sort_values("Date")
    if d.empty: continue
    d_plot = d.copy()
    d_plot["Date"] = d_plot["Date"] + DateOffset(years=DISPLAY_SHIFT_YEARS)

    r2d = r2_score(d_plot["Actual_Demand"], d_plot["Predicted_Demand"])
    plt.figure(figsize=(14,6))
    plt.plot(d_plot["Date"], d_plot["Actual_Demand"], label="Actual demand", alpha=0.85)
    plt.plot(d_plot["Date"], d_plot["Predicted_Demand"], label="Predicted demand", alpha=0.85, linestyle="--")
    plt.title(f"{item} — Demand (R² = {r2d:.3f})")
    plt.xlabel("Date"); plt.ylabel("Units"); plt.legend(); plt.grid(True, linewidth=0.3)
    plt.tight_layout(); plt.savefig(f"plots_eval/demand_{item.replace('/','_')}.png"); plt.close()

    r2l = r2_score(d_plot["Actual_LeadTime"], d_plot["Predicted_LeadTime"])
    plt.figure(figsize=(14,6))
    plt.plot(d_plot["Date"], d_plot["Actual_LeadTime"], label="Actual lead time", alpha=0.85)
    plt.plot(d_plot["Date"], d_plot["Predicted_LeadTime"], label="Predicted lead time", alpha=0.85, linestyle="--")
    plt.title(f"{item} — Lead time (R² = {r2l:.3f})")
    plt.xlabel("Date"); plt.ylabel("Days"); plt.legend(); plt.grid(True, linewidth=0.3)
    plt.tight_layout(); plt.savefig(f"plots_eval/lead_{item.replace('/','_')}.png"); plt.close()

print("[INFO] Saved evaluation plots -> plots_eval/")

# ---------------- 8) 30-day forecast + order date (DISPLAY shift) ----------------
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

    # lead-time deque (AR 업데이트용)
    lt_hist = df.loc[mask_hist, "lead_time"].values
    lead_dq = deque(list(lt_hist[-7:]), maxlen=7)  # 최근 7일

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

        # --- demand AR (log space from deque) ---
        lag1 = dq[-1] if len(dq)>=1 else dq[0]
        lag7 = dq[-7] if len(dq)>=7 else dq[0]
        ma7  = float(np.mean(list(dq)[-7:])) if len(dq)>=7 else float(np.mean(dq))
        ma28 = float(np.mean(dq)) if len(dq)>=1 else float(lag1)
        row["lag_1"] = float(lag1); row["lag_7"] = float(lag7)
        row["ma_7"]  = float(ma7);  row["ma_28"] = float(ma28)

        # --- lead AR (from deque that we update with predictions) ---
        lead_lag1 = float(lead_dq[-1]) if len(lead_dq) else 0.0
        lead_ma7  = float(np.mean(lead_dq)) if len(lead_dq) else 0.0
        row["lead_time_lag1"] = lead_lag1
        row["lead_time_ma7"]  = lead_ma7

        for c in feature_cols:
            if c not in row: row[c]=0
        x = feat_scaler.transform(pd.DataFrame([row])[feature_cols])[0]

        # predict residuals (scaled)
        y_res_s, y_lres_s = model.predict(np.array([seq]), verbose=0)
        d_res_s  = float(y_res_s.squeeze())
        l_res_s  = float(y_lres_s.squeeze())

        # combine with baselines @step
        ma7_s   = sc_d.transform(np.array([[ma7]]))[0][0]
        d_s     = d_res_s + ma7_s

        lead_lag1_s = lead_scaler.transform(np.array([[lead_lag1]]))[0][0]
        l_s         = l_res_s + lead_lag1_s

        # inverse to real units
        d_log = sc_d.inverse_transform(np.array([[d_s]]))[0][0]
        qty   = max(0.0, float(np.expm1(d_log)))
        lt    = max(0.0, float(lead_scaler.inverse_transform(np.array([[l_s]]))[0][0]))

        fut_dem.append(qty); fut_lead.append(lt)

        # AR update for next step
        dq.append(d_log)           # demand log
        lead_dq.append(lt)         # lead time
        seq = np.vstack((seq[1:], [x]))

    # inventory simulation (remaining inventory track)
    inv, ss = init_inv[item], safety[item]
    level = float(inv); stockout, stockout_idx, lt_at = None, None, None
    inv_curve = []
    for i, q in enumerate(fut_dem):
        level -= q
        inv_curve.append(level)
        if level <= ss and stockout is None:
            stockout = future_days[i]; stockout_idx = i
            lt_at = int(round(fut_lead[i]))
    order_day = None; arrival_day = None
    if stockout is not None:
        order_idx = max(0, stockout_idx - int(round(fut_lead[stockout_idx])))
        order_day = future_days[order_idx]
        # arrival based on lead at order day (more realistic)
        arrival_idx = min(len(fut_lead)-1, order_idx + int(round(fut_lead[order_idx])))
        arrival_day = future_days[arrival_idx]

    # conservative P90 order day as well
    lt_p90 = int(np.ceil(np.percentile(fut_lead, 90)))
    order_day_p90 = stockout - timedelta(days=lt_p90) if stockout is not None else None

    # --- DISPLAY timeline dates ---
    display_days = pd.date_range(start=DISPLAY_FORECAST_START, periods=len(fut_dem), freq="D")
    def to_display(t):
        if t is None: return None
        return display_days[0] + (t - future_days[0])

    order_day_disp    = to_display(order_day)
    order_day_p90_disp= to_display(order_day_p90)
    stockout_disp     = to_display(stockout)
    arrival_disp      = to_display(arrival_day)

    rows.append({
        "item": item,
        "order_date_p50": order_day_disp.strftime("%Y-%m-%d") if order_day_disp is not None else "N/A",
        "order_date_p90": order_day_p90_disp.strftime("%Y-%m-%d") if order_day_p90_disp is not None else "N/A",
        "arrival_date":   arrival_disp.strftime("%Y-%m-%d") if arrival_disp is not None else "N/A",
        "stockout_date":  stockout_disp.strftime("%Y-%m-%d") if stockout_disp is not None else "N/A",
        "pred_lead_at_stockout": lt_at if lt_at is not None else "",
        "init_inventory": int(inv),
        "safety_stock": int(ss)
    })

    # ---- Timeline (DISPLAY) ----
    fig, ax1 = plt.subplots(figsize=(14,6))
    ax1.bar(display_days, fut_dem, alpha=0.85, label="Predicted demand (units)")
    ax1.plot(display_days, inv_curve, linewidth=2.0, label="Remaining inventory", alpha=0.9)
    ax1.axhline(ss, color='gray', linestyle=':', linewidth=1.0, label="Safety stock")
    ax1.set_ylabel("Units"); ax1.set_xlabel("Date"); ax1.grid(True, axis='y', linewidth=0.3)

    ax2 = ax1.twinx()
    ax2.plot(display_days, fut_lead, linestyle='--', label="Predicted lead time (days)")
    ax2.set_ylabel("Days")
    # 축 과장 방지: 약간의 패딩
    lt_min, lt_max = float(np.min(fut_lead)), float(np.max(fut_lead))
    pad = max(0.1, 0.1*(lt_max - lt_min))
    ax2.set_ylim(lt_min - pad, lt_max + pad)

    # 수직선: 주문/도착/품절
    if stockout_disp is not None: ax1.axvline(stockout_disp, linestyle=':', linewidth=1.5, label="Stockout date")
    if order_day_disp is not None: ax1.axvline(order_day_disp, linestyle='-.', linewidth=1.5, label="Order date (P50)")
    if order_day_p90_disp is not None: ax1.axvline(order_day_p90_disp, linestyle='--', linewidth=1.0, label="Order date (P90)")
    if arrival_disp is not None: ax1.axvline(arrival_disp, color='tab:green', linestyle='-', linewidth=1.2, label="Arrival date")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper right")
    plt.title(f"{item} — 30-day forecast | init={inv}, safety={ss}")
    plt.tight_layout()
    plt.savefig(f"plots_forecast/forecast_{item.replace('/','_')}.png")
    plt.close()

    # ---- (A) 수요 히스토그램 (30일) ----
    plt.figure(figsize=(10,6))
    plt.hist(fut_dem, bins=20, alpha=0.9)
    plt.title(f"{item} - Predicted Demand Distribution (Next 30 Days)")
    plt.xlabel("Quantity"); plt.ylabel("Frequency"); plt.grid(True, alpha=0.3)
    if order_day_disp is not None:
        plt.annotate(f"Order P50: {order_day_disp.strftime('%Y-%m-%d')}",
                     xy=(0.98, 0.92), xycoords='axes fraction', ha='right', va='top',
                     bbox=dict(boxstyle="round", fc="w", ec="0.5"))
    plt.tight_layout(); plt.savefig(f"plots_forecast/hist_demand_{item.replace('/', '_')}.png"); plt.close()

    # ---- (B) 리드타임 히스토그램 (분위선 표시) ----
    plt.figure(figsize=(10,6))
    plt.hist(fut_lead, bins=15, alpha=0.9)
    q10, q50, q90 = np.percentile(fut_lead, [10,50,90])
    for q, ls in [(q10, ':'), (q50, '--'), (q90, ':')]:
        plt.axvline(q, linestyle=ls)
    plt.title(f"{item} - Predicted Lead-Time Distribution (Next 30 Days)")
    plt.xlabel("Days"); plt.ylabel("Frequency"); plt.grid(True, alpha=0.3)
    if stockout_disp is not None and order_day_disp is not None:
        msg = f"Order P50: {order_day_disp.strftime('%Y-%m-%d')} | Stockout: {stockout_disp.strftime('%Y-%m-%d')}\nP10={q10:.2f}, P50={q50:.2f}, P90={q90:.2f}"
        plt.annotate(msg, xy=(0.98, 0.92), xycoords='axes fraction',
                     ha='right', va='top', bbox=dict(boxstyle="round", fc="w", ec="0.5"))
    plt.tight_layout(); plt.savefig(f"plots_forecast/hist_lead_{item.replace('/', '_')}.png"); plt.close()

# 저장(표시 날짜 기준)
pd.DataFrame(rows).to_csv("predicted_order_dates_with_lead_time.csv", index=False, encoding="utf-8-sig")
print("[INFO] Saved: predicted_order_dates_with_lead_time.csv")
print("[INFO] Saved forecast plots -> plots_forecast/")

# ---------------- 9) Orders within NEXT 30 DAYS only (DISPLAY window) ----------------
orders = pd.DataFrame(rows)
orders["order_date_dt"] = pd.to_datetime(orders["order_date_p50"], errors="coerce")
start_disp = DISPLAY_FORECAST_START
end_disp = DISPLAY_FORECAST_START + pd.Timedelta(days=29)
mask_in_30d = orders["order_date_dt"].between(start_disp, end_disp)
orders_next30 = (orders.loc[mask_in_30d, ["item","order_date_p50","order_date_p90","arrival_date",
                                          "stockout_date","pred_lead_at_stockout","init_inventory","safety_stock"]]
                        .sort_values("order_date_p50"))
orders_next30.to_csv("order_dates_next30.csv", index=False, encoding="utf-8-sig")
print("[INFO] Saved: order_dates_next30.csv (orders within next 30 days)")
print("[DONE]")
