"""
Flood early-warning ML pipeline (Vadodara demo)

Usage:
    python flood_ml.py generate     # create flood_dataset.csv (synthetic)
    python flood_ml.py train        # train + evaluate + save flood_model.joblib
    python flood_ml.py predict      # run an example prediction

To use real data: replace flood_dataset.csv with your own file that has the
columns: timestamp, station, rain_mmh, level_cm  (5-minute interval).
The feature engineering below will work on it unchanged.
"""
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, confusion_matrix

DATA_FILE = "flood_dataset.csv"
MODEL_FILE = "flood_model.joblib"
STEP_MIN = 5                      # minutes between readings
HORIZONS = [30, 60, 120]          # forecast horizons in minutes
SEED = 42

STATIONS = {
    # danger level (cm), catchment factor c, response speed k
    "akota_bridge":     {"danger": 120, "c": 2.6, "k": 0.05},
    "alkapuri_underpass": {"danger": 60,  "c": 1.5, "k": 0.06},
    "kamnath_river":    {"danger": 150, "c": 3.4, "k": 0.025},
}


# ----------------------------------------------------------------------------
# 1. Synthetic dataset
# ----------------------------------------------------------------------------
def simulate_rain(n_steps, rng):
    """Monsoon-like rain: random storm events with a bell-shaped intensity."""
    rain = np.zeros(n_steps)
    t = 0
    while t < n_steps:
        t += int(rng.exponential(12 * 24))              # gap between storms (~12 h)
        dur = int(rng.integers(6, 60))                  # storm length (30 min - 5 h)
        peak = rng.choice([15, 30, 50, 80], p=[.35, .35, .2, .1])
        if t + dur >= n_steps:
            break
        shape = np.sin(np.linspace(0, np.pi, dur)) ** 1.5
        rain[t:t + dur] += peak * shape * rng.uniform(0.7, 1.2, dur)
        t += dur
    return rain


def generate_dataset(days=120, seed=SEED):
    rng = np.random.default_rng(seed)
    n = days * 24 * 60 // STEP_MIN
    ts = pd.date_range("2025-06-15", periods=n, freq=f"{STEP_MIN}min")
    rain = simulate_rain(n, rng)                        # same weather for all stations
    frames = []
    for name, p in STATIONS.items():
        level = np.zeros(n)
        L = 5.0
        for i in range(n):
            drive = rain[i] * p["c"]
            L += (drive - L) * p["k"] + rng.normal(0, 0.4)
            L = max(0.0, L)
            level[i] = L
        frames.append(pd.DataFrame({
            "timestamp": ts, "station": name,
            "rain_mmh": rain.round(2), "level_cm": level.round(2),
        }))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"Saved {DATA_FILE}: {len(df):,} rows, {days} days, {len(STATIONS)} stations")
    return df


# ----------------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------------
def build_features(df, with_targets=True):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    out = []
    for name, g in df.groupby("station"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        danger = STATIONS[name]["danger"]
        s = STEP_MIN
        g["danger_cm"] = danger
        g["level_ratio"] = g["level_cm"] / danger
        for m in (15, 30, 60):
            k = m // s
            g[f"level_change_{m}m"] = g["level_cm"] - g["level_cm"].shift(k)
            g[f"rain_sum_{m}m"] = g["rain_mmh"].rolling(k, min_periods=1).sum() * s / 60
        g["rain_sum_180m"] = g["rain_mmh"].rolling(180 // s, min_periods=1).sum() * s / 60
        g["trend_slope"] = (g["level_cm"] - g["level_cm"].shift(8)) / 8   # cm per step
        g["hour"] = g["timestamp"].dt.hour
        g["station_code"] = list(STATIONS).index(name)
        if with_targets:
            for h in HORIZONS:
                g[f"level_in_{h}m"] = g["level_cm"].shift(-(h // s))
            window = 120 // s
            future_max = g["level_cm"][::-1].rolling(window, min_periods=window).max()[::-1].shift(-1)
            g["flood_in_120m"] = (future_max >= danger).astype(float)
            g.loc[future_max.isna(), "flood_in_120m"] = np.nan
        out.append(g)
    res = pd.concat(out, ignore_index=True).sort_values(["timestamp", "station"])
    return res.dropna().reset_index(drop=True)


FEATURES = ["level_cm", "rain_mmh", "level_ratio", "danger_cm",
            "level_change_15m", "level_change_30m", "level_change_60m",
            "rain_sum_15m", "rain_sum_30m", "rain_sum_60m", "rain_sum_180m",
            "trend_slope", "hour", "station_code"]


# ----------------------------------------------------------------------------
# 3. Training and evaluation
# ----------------------------------------------------------------------------
def train():
    try:
        raw = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        raw = generate_dataset()
    data = build_features(raw)

    # Chronological split (80/20): never shuffle time series
    cut = data["timestamp"].quantile(0.8)
    tr, te = data[data["timestamp"] <= cut], data[data["timestamp"] > cut]
    print(f"Train rows: {len(tr):,} | Test rows: {len(te):,}\n")

    model = {"regressors": {}, "features": FEATURES, "horizons": HORIZONS}

    # a) Regressors: water level at +30, +60, +120 minutes
    for h in HORIZONS:
        reg = GradientBoostingRegressor(n_estimators=250, max_depth=4,
                                        learning_rate=0.08, random_state=SEED)
        reg.fit(tr[FEATURES], tr[f"level_in_{h}m"])
        pred = reg.predict(te[FEATURES])
        # baseline: "level stays the same"
        base = mean_absolute_error(te[f"level_in_{h}m"], te["level_cm"])
        mae = mean_absolute_error(te[f"level_in_{h}m"], pred)
        r2 = r2_score(te[f"level_in_{h}m"], pred)
        print(f"+{h:>3} min level | MAE {mae:5.2f} cm (baseline {base:5.2f}) | R2 {r2:.3f}")
        model["regressors"][h] = reg

    # b) Classifier: will the level cross the danger line within 2 hours?
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                 class_weight="balanced", random_state=SEED, n_jobs=-1)
    clf.fit(tr[FEATURES], tr["flood_in_120m"])
    prob = clf.predict_proba(te[FEATURES])[:, 1]
    pred_c = (prob >= 0.4).astype(int)       # lower threshold = catch more floods
    print("\nFlood within 120 min (threshold 0.40):")
    print(classification_report(te["flood_in_120m"], pred_c,
                                target_names=["safe", "flood"], digits=3))
    print("Confusion matrix [rows=actual, cols=predicted]:")
    print(confusion_matrix(te["flood_in_120m"], pred_c))
    model["classifier"] = clf
    model["threshold"] = 0.4

    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop features:\n", imp.head(6).round(3).to_string())

    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved {MODEL_FILE}")


# ----------------------------------------------------------------------------
# 4. Prediction (use this inside your MQTT / FastAPI backend)
# ----------------------------------------------------------------------------
def predict(recent_df):
    """
    recent_df: last >= 40 readings (3+ hours) for ONE station, columns:
               timestamp, station, rain_mmh, level_cm   (5-minute interval)
    Returns dict with forecast levels, flood probability and ETA to danger.
    """
    model = joblib.load(MODEL_FILE)
    station = recent_df["station"].iloc[-1]
    feats = build_features(recent_df, with_targets=False)
    x = feats[FEATURES].iloc[[-1]]
    danger = STATIONS[station]["danger"]

    levels = {h: float(model["regressors"][h].predict(x)[0]) for h in HORIZONS}
    prob = float(model["classifier"].predict_proba(x)[0, 1])
    eta = next((h for h in HORIZONS if levels[h] >= danger), None)
    status = ("danger" if x["level_cm"].iloc[0] >= danger
              else "warning" if prob >= model["threshold"] else "safe")
    return {"station": station, "current_cm": float(x["level_cm"].iloc[0]),
            "danger_cm": danger, "forecast_cm": levels,
            "flood_probability": round(prob, 3), "eta_minutes": eta, "status": status}


def demo_predict():
    raw = pd.read_csv(DATA_FILE)
    st = "akota_bridge"
    g = raw[raw["station"] == st].reset_index(drop=True)
    # pick a moment when water is rising fast
    i = int(g["level_cm"].diff(6).iloc[60:-30].idxmax())
    window = g.iloc[i - 40: i + 1]
    print(predict(window))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"generate": generate_dataset, "train": train, "predict": demo_predict}[cmd]()
