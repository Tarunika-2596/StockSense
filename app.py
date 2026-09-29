# ============================================================
# STOCKSENSE
# AI-Powered Inventory Intelligence System
# Complete Hackathon Pipeline
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy.stats import ttest_ind, f_oneway, chi2_contingency

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# XGBoost
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

@st.cache_data
def load_raw_data():

    transactions = pd.read_csv(
        os.path.join(RAW_DIR, "transactions.csv")
    )

    products = pd.read_csv(
        os.path.join(RAW_DIR, "products.csv")
    )

    stores = pd.read_csv(
        os.path.join(RAW_DIR, "stores.csv")
    )

    inventory = pd.read_csv(
        os.path.join(RAW_DIR, "inventory.csv")
    )

    external = pd.read_csv(
        os.path.join(RAW_DIR, "external_factors.csv")
    )

    return transactions, products, stores, inventory, external


# ============================================================
# 3. CLEAN DATA
# ============================================================

@st.cache_data
def clean_data():

    transactions, products, stores, inventory, external = (
        load_raw_data()
    )

    # -----------------------------
    # Date conversion
    # -----------------------------

    transactions["date"] = pd.to_datetime(
        transactions["date"]
    )

    inventory["date"] = pd.to_datetime(
        inventory["date"]
    )

    external["date"] = pd.to_datetime(
        external["date"]
    )

    # -----------------------------
    # Remove duplicate transactions
    # -----------------------------

    duplicate_transactions = transactions.duplicated(
        subset="transaction_id"
    ).sum()

    transactions = transactions.drop_duplicates(
        subset="transaction_id"
    )

    # -----------------------------
    # Invalid quantity
    # -----------------------------

    invalid_quantity = (
        transactions["quantity"] < 0
    ).sum()

    transactions = transactions[
        transactions["quantity"] >= 0
    ].copy()

    # -----------------------------
    # Standardize categories
    # -----------------------------

    products["category"] = (
        products["category"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    category_map = {
        "beverage": "beverages",
        "beverages": "beverages",
        "personalcare": "personal care",
        "personal care": "personal care",
        "dairy": "dairy",
        "snacks": "snacks",
        "household": "household"
    }

    products["category"] = (
        products["category"]
        .replace(category_map)
    )

    # -----------------------------
    # Missing value handling
    # -----------------------------

    transactions = transactions.dropna(
        subset=[
            "date",
            "store_id",
            "product_id",
            "quantity"
        ]
    )

    products = products.dropna(
        subset=[
            "product_id",
            "category"
        ]
    )

    stores = stores.dropna(
        subset=[
            "store_id",
            "city"
        ]
    )

    inventory = inventory.dropna(
        subset=[
            "date",
            "store_id",
            "product_id"
        ]
    )

    external = external.dropna(
        subset=[
            "date",
            "city"
        ]
    )

    # -----------------------------
    # Data quality report
    # -----------------------------

    quality = pd.DataFrame({
        "Issue": [
            "Duplicate transactions",
            "Invalid negative quantities"
        ],
        "Count": [
            duplicate_transactions,
            invalid_quantity
        ],
        "Action": [
            "Removed",
            "Removed"
        ]
    })

    quality.to_csv(
        os.path.join(
            REPORT_DIR,
            "data_quality_report.csv"
        ),
        index=False
    )

    return (
        transactions,
        products,
        stores,
        inventory,
        external
    )


# ============================================================
# 4. CREATE MASTER DATASET
# ============================================================

@st.cache_data
def create_master():

    (
        transactions,
        products,
        stores,
        inventory,
        external
    ) = clean_data()

    # -----------------------------
    # Daily sales
    # -----------------------------

    daily_sales = (
        transactions
        .groupby(
            [
                "date",
                "store_id",
                "product_id"
            ]
        )
        .agg(
            demand=("quantity", "sum"),
            selling_price=(
                "selling_price",
                "mean"
            ),
            discount_pct=(
                "discount_pct",
                "mean"
            ),
            promotion_flag=(
                "promotion_flag",
                "max"
            )
        )
        .reset_index()
    )

    # -----------------------------
    # Merge inventory
    # -----------------------------

    master = daily_sales.merge(
        inventory,
        on=[
            "date",
            "store_id",
            "product_id"
        ],
        how="left"
    )

    # -----------------------------
    # Merge products
    # -----------------------------

    master = master.merge(
        products,
        on="product_id",
        how="left"
    )

    # -----------------------------
    # Merge stores
    # -----------------------------

    master = master.merge(
        stores,
        on="store_id",
        how="left"
    )

    # -----------------------------
    # Merge external factors
    # -----------------------------

    master = master.merge(
        external,
        on=[
            "date",
            "city"
        ],
        how="left"
    )

    master = master.sort_values(
        [
            "store_id",
            "product_id",
            "date"
        ]
    )

    # -----------------------------
    # Revenue
    # -----------------------------

    master["revenue"] = (
        master["demand"] *
        master["selling_price"]
    )

    # -----------------------------
    # Current stock-out
    # -----------------------------

    master["stockout_flag"] = (
        (master["closing"] <= 0) &
        (master["demand"] > 0)
    ).astype(int)

    master.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "master_dataset.csv"
        ),
        index=False
    )

    return master


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

@st.cache_data
def create_features(master):

    df = master.copy()

    df = df.sort_values(
        [
            "store_id",
            "product_id",
            "date"
        ]
    ).reset_index(drop=True)

    # -----------------------------
    # Time features
    # -----------------------------

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["weekend_flag"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["month"] = (
        df["date"].dt.month
    )

    df["week_no"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["festival_flag"] = (
        df["festival"]
        .fillna(0)
        .astype(int)
    )

    # -----------------------------
    # Group
    # -----------------------------

    grouped = df.groupby(
        [
            "store_id",
            "product_id"
        ]
    )["demand"]

    # -----------------------------
    # Lag features
    # -----------------------------

    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["lag_14"] = grouped.shift(14)

    # -----------------------------
    # Rolling features
    # IMPORTANT:
    # shift(1) prevents leakage
    # -----------------------------

    df["rolling_mean_7"] = (
        grouped.transform(
            lambda x:
            x.shift(1)
             .rolling(7)
             .mean()
        )
    )

    df["rolling_mean_14"] = (
        grouped.transform(
            lambda x:
            x.shift(1)
             .rolling(14)
             .mean()
        )
    )

    df["rolling_std_7"] = (
        grouped.transform(
            lambda x:
            x.shift(1)
             .rolling(7)
             .std()
        )
    )

    # -----------------------------
    # Inventory features
    # -----------------------------

    df["days_of_inventory"] = (
        df["closing"] /
        df["rolling_mean_7"].replace(
            0,
            np.nan
        )
    )

    df["inventory_to_demand_ratio"] = (
        df["closing"] /
        df["rolling_mean_7"].replace(
            0,
            np.nan
        )
    )

    df["reorder_gap"] = (
        df["closing"] -
        df["reorder_lvl"]
    )

    # -----------------------------
    # Price change
    # -----------------------------

    df["price_change"] = (
        df
        .groupby(
            [
                "store_id",
                "product_id"
            ]
        )["selling_price"]
        .pct_change()
    )

    # ========================================================
    # DEMAND TARGET
    # Next 7 days demand
    # ========================================================

    df["next_7_day_demand"] = (
        df
        .groupby(
            [
                "store_id",
                "product_id"
            ]
        )["demand"]
        .transform(
            lambda x:
            sum(
                x.shift(-i)
                for i in range(1, 8)
            )
        )
    )

    # ========================================================
    # STOCK-OUT TARGET
    #
    # Predict NEXT DAY stock-out instead of current
    # stock-out to avoid target leakage.
    # ========================================================

    current_stockout = (
        (df["closing"] <= 0) &
        (df["demand"] > 0)
    ).astype(int)

    df["future_stockout_flag"] = (
        current_stockout
        .groupby(
            [
                df["store_id"],
                df["product_id"]
            ]
        )
        .shift(-1)
    )

    # -----------------------------
    # Save
    # -----------------------------

    df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "feature_dataset.csv"
        ),
        index=False
    )

    return df


# ============================================================
# 6. FEATURE LIST
# ============================================================

FEATURES = [
    "day_of_week",
    "weekend_flag",
    "month",
    "week_no",
    "festival_flag",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "days_of_inventory",
    "inventory_to_demand_ratio",
    "reorder_gap",
    "discount_pct",
    "promotion_flag",
    "price_change"
]


# ============================================================
# 7. DEMAND FORECAST MODEL
# ============================================================

@st.cache_resource
def train_demand_model(df):

    model_df = df.dropna(
        subset=FEATURES +
        ["next_7_day_demand"]
    ).copy()

    model_df = model_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    model_df = model_df.dropna(
        subset=FEATURES +
        ["next_7_day_demand"]
    )

    model_df = model_df.sort_values(
        "date"
    ).reset_index(drop=True)

    X = model_df[FEATURES]
    y = model_df["next_7_day_demand"]

    # -----------------------------
    # Time split
    # -----------------------------

    split = int(
        len(model_df) * 0.80
    )

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # -----------------------------
    # Baseline
    # -----------------------------

    baseline_pred = (
        model_df
        .loc[
            X_test.index,
            "rolling_mean_7"
        ]
    )

    # -----------------------------
    # Random Forest
    # -----------------------------

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_pred = rf.predict(
        X_test
    )

    # -----------------------------
    # XGBoost
    # -----------------------------

    xgb = None
    xgb_pred = None

    if XGB_AVAILABLE:

        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )

        xgb.fit(
            X_train,
            y_train
        )

        xgb_pred = xgb.predict(
            X_test
        )

    # -----------------------------
    # Evaluation function
    # -----------------------------

    def evaluate(
        actual,
        predicted
    ):

        mae = mean_absolute_error(
            actual,
            predicted
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        mape = np.mean(
            np.abs(
                (
                    actual - predicted
                ) /
                np.maximum(
                    np.abs(actual),
                    1
                )
            )
        ) * 100

        r2 = r2_score(
            actual,
            predicted
        )

        return [
            mae,
            rmse,
            mape,
            r2
        ]

    results = []

    results.append(
        [
            "Baseline",
            *evaluate(
                y_test,
                baseline_pred
            )
        ]
    )

    results.append(
        [
            "Random Forest",
            *evaluate(
                y_test,
                rf_pred
            )
        ]
    )

    if xgb_pred is not None:

        results.append(
            [
                "XGBoost",
                *evaluate(
                    y_test,
                    xgb_pred
                )
            ]
        )

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "MAE",
            "RMSE",
            "MAPE",
            "R2"
        ]
    )

    # -----------------------------
    # Select model
    # -----------------------------

    # Select based on lowest MAE
    best_model_name = (
        results_df
        .sort_values("MAE")
        .iloc[0]["Model"]
    )

    if best_model_name == "XGBoost" and xgb is not None:
        selected_model = xgb
    else:
        selected_model = rf
        best_model_name = "Random Forest"

    # -----------------------------
    # Save model/results
    # -----------------------------

    joblib.dump(
        selected_model,
        os.path.join(
            MODEL_DIR,
            "demand_forecasting_model.pkl"
        )
    )

    results_df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "demand_model_results.csv"
        ),
        index=False
    )

    return (
        selected_model,
        results_df,
        model_df
    )


# ============================================================
# 8. STOCK-OUT CLASSIFICATION MODEL
# ============================================================

@st.cache_resource
def train_stockout_model(df):

    model_df = df.dropna(
        subset=FEATURES +
        ["future_stockout_flag"]
    ).copy()

    model_df = model_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    model_df = model_df.dropna(
        subset=FEATURES +
        ["future_stockout_flag"]
    )

    model_df = model_df.sort_values(
        "date"
    ).reset_index(drop=True)

    X = model_df[FEATURES]

    y = (
        model_df[
            "future_stockout_flag"
        ].astype(int)
    )

    # -----------------------------
    # Time split
    # -----------------------------

    split = int(
        len(model_df) * 0.80
    )

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # -----------------------------
    # Random Forest classifier
    # -----------------------------

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_pred = rf.predict(
        X_test
    )

    rf_prob = rf.predict_proba(
        X_test
    )[:, 1]

    # -----------------------------
    # Metrics
    # -----------------------------

    rf_accuracy = accuracy_score(
        y_test,
        rf_pred
    )

    rf_precision = precision_score(
        y_test,
        rf_pred,
        zero_division=0
    )

    rf_recall = recall_score(
        y_test,
        rf_pred,
        zero_division=0
    )

    rf_f1 = f1_score(
        y_test,
        rf_pred,
        zero_division=0
    )

    try:
        rf_auc = roc_auc_score(
            y_test,
            rf_prob
        )
    except:
        rf_auc = 0

    results = [
        [
            "Random Forest",
            rf_accuracy,
            rf_precision,
            rf_recall,
            rf_f1,
            rf_auc
        ]
    ]

    # -----------------------------
    # XGBoost
    # -----------------------------

    xgb = None

    if XGB_AVAILABLE:

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        )

        xgb.fit(
            X_train,
            y_train
        )

        xgb_pred = xgb.predict(
            X_test
        )

        xgb_prob = xgb.predict_proba(
            X_test
        )[:, 1]

        xgb_accuracy = accuracy_score(
            y_test,
            xgb_pred
        )

        xgb_precision = precision_score(
            y_test,
            xgb_pred,
            zero_division=0
        )

        xgb_recall = recall_score(
            y_test,
            xgb_pred,
            zero_division=0
        )

        xgb_f1 = f1_score(
            y_test,
            xgb_pred,
            zero_division=0
        )

        try:
            xgb_auc = roc_auc_score(
                y_test,
                xgb_prob
            )
        except:
            xgb_auc = 0

        results.append(
            [
                "XGBoost",
                xgb_accuracy,
                xgb_precision,
                xgb_recall,
                xgb_f1,
                xgb_auc
            ]
        )

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]
    )

    # -----------------------------
    # Select using F1
    # -----------------------------

    best_index = (
        results_df["F1"]
        .idxmax()
    )

    best_name = (
        results_df.loc[
            best_index,
            "Model"
        ]
    )

    if (
        best_name == "XGBoost"
        and xgb is not None
    ):
        selected_model = xgb
    else:
        selected_model = rf
        best_name = "Random Forest"

    # -----------------------------
    # Save
    # -----------------------------

    joblib.dump(
        selected_model,
        os.path.join(
            MODEL_DIR,
            "stockout_model.pkl"
        )
    )

    results_df.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "stockout_model_results.csv"
        ),
        index=False
    )

    cm = confusion_matrix(
        y_test,
        rf_pred
    )

    return (
        selected_model,
        results_df,
        cm,
        model_df
    )


# ============================================================
# 9. CREATE RECOMMENDATIONS
# ============================================================

@st.cache_data
def create_recommendations(
    df,
    _demand_model,
    _stockout_model
):

    latest = (
        df.sort_values("date")
        .groupby(
            [
                "store_id",
                "product_id"
            ]
        )
        .tail(1)
        .copy()
    )

    X_latest = latest[
        FEATURES
    ].copy()

    X_latest = X_latest.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_latest = X_latest.fillna(0)

    # -----------------------------
    # Demand forecast
    # -----------------------------

    latest["forecast_7_day"] = (
        _demand_model.predict(
    X_latest
)
    )

    latest["forecast_7_day"] = (
        latest["forecast_7_day"]
        .clip(lower=0)
    )

    # -----------------------------
    # Stock-out probability
    # -----------------------------

    latest[
        "stockout_probability"
    ] = (
        _stockout_model.predict_proba(
    X_latest
)[:, 1]
    )

    # -----------------------------
    # Risk
    # -----------------------------

    def risk(prob):

        if prob >= 0.70:
            return "HIGH"

        elif prob >= 0.40:
            return "MEDIUM"

        return "LOW"

    latest["risk_level"] = (
        latest[
            "stockout_probability"
        ].apply(risk)
    )

    # -----------------------------
    # Recommended stock
    #
    # 10% safety buffer
    # -----------------------------

    latest[
        "recommended_stock"
    ] = (
        latest[
            "forecast_7_day"
        ] * 1.10
    )

    # -----------------------------
    # Reorder quantity
    #
    # Reorder =
    # max(0,
    # recommended stock
    # - current stock
    # - incoming stock)
    # -----------------------------

    latest[
        "reorder_quantity"
    ] = (
        latest[
            "recommended_stock"
        ]
        - latest["closing"]
        - latest["received"]
    ).clip(lower=0)

    latest[
        "reorder_quantity"
    ] = (
        latest[
            "reorder_quantity"
        ]
        .round()
        .astype(int)
    )

    # -----------------------------
    # Manager action
    # -----------------------------

    def action(row):

        if row["risk_level"] == "HIGH":
            return "Order immediately"

        elif row["risk_level"] == "MEDIUM":
            return "Review and reorder"

        elif row["reorder_quantity"] > 0:
            return "Monitor stock"

        return "No action"

    latest[
        "manager_action"
    ] = latest.apply(
        action,
        axis=1
    )

    # -----------------------------
    # Explanation
    # -----------------------------

    def explanation(row):

        reasons = []

        if row[
            "stockout_probability"
        ] >= 0.70:

            reasons.append(
                "high stock-out probability"
            )

        if row[
            "days_of_inventory"
        ] < 7:

            reasons.append(
                "low inventory coverage"
            )

        if row[
            "reorder_gap"
        ] < 0:

            reasons.append(
                "stock below reorder level"
            )

        if row[
            "rolling_mean_7"
        ] > row["lag_7"]:

            reasons.append(
                "recent demand is increasing"
            )

        if not reasons:

            reasons.append(
                "stock level currently appears adequate"
            )

        return ", ".join(reasons)

    latest["reason"] = latest.apply(
        explanation,
        axis=1
    )

    # -----------------------------
    # Final columns
    # -----------------------------

    recommendations = latest[
        [
            "date",
            "store_id",
            "product_id",
            "forecast_7_day",
            "closing",
            "received",
            "stockout_probability",
            "risk_level",
            "recommended_stock",
            "reorder_quantity",
            "reason",
            "manager_action"
        ]
    ].copy()

    recommendations.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "reorder_recommendations.csv"
        ),
        index=False
    )

    return recommendations


# ============================================================
# 10. STATISTICAL TESTS
# ============================================================

@st.cache_data
def statistical_tests(master):

    results = []

    # ========================================================
    # TEST 1
    # Promotion -> Demand
    # ========================================================

    promotion = master[
        master["promotion_flag"] == 1
    ]["demand"]

    no_promotion = master[
        master["promotion_flag"] == 0
    ]["demand"]

    t_stat, p1 = ttest_ind(
        promotion,
        no_promotion,
        equal_var=False
    )

    results.append(
        [
            "Promotion vs Demand",
            "Independent T-test",
            p1
        ]
    )

    # ========================================================
    # TEST 2
    # Store Type -> Demand
    # ========================================================

    groups = [
        group["demand"].dropna()
        for _, group
        in master.groupby(
            "store_type"
        )
    ]

    if len(groups) >= 2:

        f_stat, p2 = f_oneway(
            *groups
        )

    else:

        p2 = 1

    results.append(
        [
            "Store Type vs Demand",
            "ANOVA",
            p2
        ]
    )

    # ========================================================
    # TEST 3
    # Promotion -> Stock-out
    # ========================================================

    table = pd.crosstab(
        master["promotion_flag"],
        master["stockout_flag"]
    )

    try:

        chi2, p3, dof, expected = (
            chi2_contingency(table)
        )

    except:

        p3 = 1

    results.append(
        [
            "Promotion vs Stock-out",
            "Chi-square",
            p3
        ]
    )

    stats_df = pd.DataFrame(
        results,
        columns=[
            "Analysis",
            "Test",
            "P-value"
        ]
    )

    stats_df["Significant"] = (
        stats_df["P-value"] < 0.05
    )

    stats_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "statistical_tests.csv"
        ),
        index=False
    )

    return stats_df


# ============================================================
# 11. RUN COMPLETE PIPELINE
# ============================================================

(
    transactions,
    products,
    stores,
    inventory,
    external
) = clean_data()

master = create_master()

feature_df = create_features(
    master
)

(
    demand_model,
    demand_results,
    demand_model_df
) = train_demand_model(
    feature_df
)

(
    stockout_model,
    stockout_results,
    confusion,
    stockout_model_df
) = train_stockout_model(
    feature_df
)

recommendations = create_recommendations(
    feature_df,
    demand_model,
    stockout_model
)

statistics = statistical_tests(
    master
)


# ============================================================
# 12. STREAMLIT DASHBOARD
# ============================================================

st.set_page_config(
    page_title="StockSense | Inventory Intelligence",
    page_icon="📦",
    layout="wide"
)

st.markdown(
    """
    <style>
    :root {
        --ink: #24352d;
        --muted: #68766d;
        --green: #47785e;
        --green-dark: #315e48;
        --green-pale: #e7eee5;
        --paper: #f7f6f0;
        --surface: #fffefa;
        --line: #e3e5db;
        --red: #a44d42;
        --amber: #a77b35;
    }
    .stApp { background: var(--paper); color: var(--ink); font-family: "Segoe UI", "Aptos", sans-serif; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1440px; padding-top: 1.6rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink); font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif; font-weight: 500; letter-spacing: 0; }
    h1 { font-size: 2.35rem; margin-bottom: .25rem; }
    h2 { font-size: 1.55rem; margin-top: 1.1rem; }
    h3 { font-size: 1.2rem; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    .ss-masthead { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: 0 0 1rem; margin-bottom: 1.6rem; border-bottom: 1px solid var(--line); }
    .ss-brand { color: var(--ink); font-size: .95rem; font-weight: 700; letter-spacing: .04em; }
    .ss-brand-mark { display: inline-block; width: 9px; height: 9px; margin-right: .55rem; border-radius: 2px; background: var(--green); }
    .ss-workspace { color: var(--green-dark); background: var(--green-pale); border-radius: 4px; padding: .35rem .65rem; font-size: .75rem; font-weight: 600; }
    [data-testid="stSidebar"] { background: #eff0e8; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 1.3rem; }
    [data-testid="stMetric"] { min-height: 108px; padding: 1rem 1.05rem; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }
    [data-testid="stMetricLabel"] p { color: var(--muted); font-size: .82rem; }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif; }
    [data-testid="stTabs"] [role="tablist"] { gap: .35rem; border-bottom: 1px solid var(--line); }
    [data-testid="stTabs"] button[role="tab"] { color: var(--muted); padding: .7rem .9rem; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--green-dark); border-bottom-color: var(--green); }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 5px; overflow: hidden; background: var(--surface); }
    .stButton > button, .stDownloadButton > button { color: #fffefa; background: var(--green); border: 1px solid var(--green); border-radius: 5px; font-weight: 600; }
    .stButton > button p, .stDownloadButton > button p { color: #fffefa; }
    .stButton > button:hover, .stDownloadButton > button:hover { color: #fffefa; background: var(--green-dark); border-color: var(--green-dark); }
    [data-testid="stAlert"] { border-radius: 5px; }
    hr { border-color: var(--line); }
    @media (max-width: 700px) {
        .block-container { padding: 1rem .8rem 2rem; }
        .ss-masthead { align-items: flex-start; }
        h1 { font-size: 1.9rem; }
        [data-testid="stTabs"] [role="tablist"] { overflow-x: auto; flex-wrap: nowrap; }
    }
    </style>
    <div class="ss-masthead">
        <div class="ss-brand"><span class="ss-brand-mark"></span>STOCKSENSE</div>
        <div class="ss-workspace">INVENTORY WORKSPACE</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Inventory at a glance")
st.markdown("Demand, availability, and replenishment signals for your stores.")


def reset_dashboard_filters():
    st.session_state["filter_store"] = "All stores"
    st.session_state["filter_risk"] = "All risk levels"


st.sidebar.title("Workspace filters")
st.sidebar.caption("Apply to forecasts, risk, and action lists.")
st.sidebar.button(
    "Reset filters",
    on_click=reset_dashboard_filters,
    use_container_width=True
)

store_options = ["All stores"] + sorted(
    recommendations["store_id"].unique().tolist()
)
risk_options = ["All risk levels", "HIGH", "MEDIUM", "LOW"]

selected_store = st.sidebar.selectbox(
    "Store",
    store_options,
    key="filter_store"
)
selected_risk = st.sidebar.selectbox(
    "Stock-out risk",
    risk_options,
    key="filter_risk"
)

filtered = recommendations.copy()
if selected_store != "All stores":
    filtered = filtered[filtered["store_id"] == selected_store]
if selected_risk != "All risk levels":
    filtered = filtered[filtered["risk_level"] == selected_risk]

st.sidebar.divider()
st.sidebar.caption(f"{len(filtered):,} products match these filters")

start_date = pd.to_datetime(master["date"]).min().strftime("%b %d, %Y")
end_date = pd.to_datetime(master["date"]).max().strftime("%b %d, %Y")
st.caption(f"Data coverage: {start_date} to {end_date}  ·  {master['store_id'].nunique()} stores")

# Prepare shared views once so every tab reflects the same filters.
total_revenue = master["revenue"].sum()
total_units = master["demand"].sum()
stockout_rate = master["stockout_flag"].mean() * 100
filtered_high_risk = filtered[filtered["risk_level"] == "HIGH"]

forecast_display = filtered[["store_id", "product_id", "forecast_7_day"]].copy()
forecast_display["forecast_7_day"] = forecast_display["forecast_7_day"].round(0)
forecast_display = forecast_display.rename(columns={
    "store_id": "Store",
    "product_id": "Product",
    "forecast_7_day": "7-day forecast"
})

daily_demand = master.groupby("date")["demand"].sum().sort_index()
risk_counts = filtered["risk_level"].value_counts().reindex(
    ["HIGH", "MEDIUM", "LOW"], fill_value=0
)
actions = filtered[filtered["reorder_quantity"] > 0].sort_values(
    "reorder_quantity", ascending=False
)


def style_axis(axis):
    axis.set_facecolor("#fffefa")
    axis.grid(axis="y", color="#e5e7de", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors="#68766d", labelsize=9)
    for spine in axis.spines.values():
        spine.set_visible(False)


overview_tab, demand_tab, inventory_tab, product_tab, model_tab, export_tab = st.tabs([
    "Overview",
    "Demand",
    "Inventory & actions",
    "Product analysis",
    "Model & data",
    "Export"
])

with overview_tab:
    st.subheader("Executive overview")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Total revenue", f"₹{total_revenue:,.0f}")
    metric_columns[1].metric("Units sold", f"{total_units:,.0f}")
    metric_columns[2].metric("Stock-out rate", f"{stockout_rate:.2f}%")
    metric_columns[3].metric("High-risk products", f"{len(filtered_high_risk):,}")

    trend_column, risk_column = st.columns([1.65, 1])
    with trend_column:
        st.subheader("Daily demand")
        figure, axis = plt.subplots(figsize=(9, 3.4))
        figure.patch.set_facecolor("#fffefa")
        style_axis(axis)
        axis.plot(daily_demand.index, daily_demand.values, color="#47785e", linewidth=2)
        axis.fill_between(daily_demand.index, daily_demand.values, color="#47785e", alpha=.10)
        axis.set_ylabel("Units", color="#68766d")
        axis.set_xlabel("")
        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
    with risk_column:
        st.subheader("Risk distribution")
        figure, axis = plt.subplots(figsize=(5, 3.4))
        figure.patch.set_facecolor("#fffefa")
        axis.set_facecolor("#fffefa")
        risk_colors = ["#b96354", "#c69a4d", "#6f9677"]
        axis.bar(risk_counts.index, risk_counts.values, color=risk_colors, width=.58)
        axis.set_ylabel("Products", color="#68766d")
        style_axis(axis)
        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    st.subheader("Priority replenishment")
    if actions.empty:
        st.info("No reorder actions match the current filters.")
    else:
        priority = actions.head(8)[[
            "store_id", "product_id", "risk_level", "stockout_probability",
            "closing", "reorder_quantity", "manager_action"
        ]].copy()
        priority["stockout_probability"] = (priority["stockout_probability"] * 100).round(1)
        priority = priority.rename(columns={
            "store_id": "Store", "product_id": "Product", "risk_level": "Risk",
            "stockout_probability": "Stock-out chance (%)", "closing": "On hand",
            "reorder_quantity": "Reorder units", "manager_action": "Suggested action"
        })
        st.dataframe(priority, use_container_width=True, hide_index=True)

with demand_tab:
    st.subheader("Demand forecast")
    st.caption("Estimated demand over the next seven days, filtered by store and risk.")
    if forecast_display.empty:
        st.info("No forecast rows match the current filters.")
    else:
        st.dataframe(forecast_display, use_container_width=True, hide_index=True)

    st.subheader("Demand history")
    figure, axis = plt.subplots(figsize=(11, 3.8))
    figure.patch.set_facecolor("#fffefa")
    style_axis(axis)
    axis.plot(daily_demand.index, daily_demand.values, color="#47785e", linewidth=2)
    axis.fill_between(daily_demand.index, daily_demand.values, color="#47785e", alpha=.10)
    axis.set_ylabel("Units", color="#68766d")
    axis.set_xlabel("")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

    st.subheader("Forecast drivers")
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": demand_model.feature_importances_
    }).sort_values("Importance", ascending=True).tail(10)
    figure, axis = plt.subplots(figsize=(9, 4.2))
    figure.patch.set_facecolor("#fffefa")
    axis.set_facecolor("#fffefa")
    axis.barh(importance["Feature"], importance["Importance"], color="#47785e")
    axis.grid(axis="x", color="#e5e7de", linewidth=.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors="#68766d", labelsize=9)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

with inventory_tab:
    st.subheader("Inventory risk")
    risk_column, risk_detail_column = st.columns([1, 1.6])
    with risk_column:
        figure, axis = plt.subplots(figsize=(5, 3.2))
        figure.patch.set_facecolor("#fffefa")
        axis.set_facecolor("#fffefa")
        axis.bar(risk_counts.index, risk_counts.values,
                 color=["#b96354", "#c69a4d", "#6f9677"], width=.58)
        axis.set_ylabel("Products", color="#68766d")
        style_axis(axis)
        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
    with risk_detail_column:
        risk_table = filtered[[
            "store_id", "product_id", "stockout_probability", "risk_level", "closing"
        ]].copy()
        risk_table["stockout_probability"] = (risk_table["stockout_probability"] * 100).round(2)
        risk_table = risk_table.rename(columns={
            "store_id": "Store", "product_id": "Product",
            "stockout_probability": "Stock-out probability (%)",
            "risk_level": "Risk", "closing": "On hand"
        })
        st.dataframe(risk_table, use_container_width=True, hide_index=True)

    st.subheader("Stock-out frequency by store and category")
    heatmap_data = pd.pivot_table(
        master, values="stockout_flag", index="store_id", columns="category",
        aggfunc="sum", fill_value=0
    )
    figure, axis = plt.subplots(figsize=(10, max(3, min(7, len(heatmap_data) * .42))))
    figure.patch.set_facecolor("#fffefa")
    sns.heatmap(heatmap_data, annot=True, fmt="g", ax=axis, cmap="YlGn",
                linewidths=.5, linecolor="#fffefa", cbar_kws={"shrink": .8})
    axis.set_title("Stock-out events", color="#24352d", loc="left", pad=12)
    axis.tick_params(colors="#68766d", labelsize=9)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

    st.subheader("Manager action list")
    if actions.empty:
        st.info("No reorder actions match the current filters.")
    else:
        action_display = actions[[
            "store_id", "product_id", "risk_level", "forecast_7_day", "closing",
            "reorder_quantity", "manager_action"
        ]].rename(columns={
            "store_id": "Store", "product_id": "Product", "risk_level": "Risk",
            "forecast_7_day": "7-day forecast", "closing": "On hand",
            "reorder_quantity": "Reorder units", "manager_action": "Suggested action"
        })
        st.dataframe(action_display, use_container_width=True, hide_index=True)

with product_tab:
    st.subheader("Product risk explanation")
    if filtered.empty:
        st.info("No products match the current filters. Adjust the filters to continue.")
    else:
        product_options = sorted(filtered["product_id"].unique().tolist())
        selected_product = st.selectbox("Product", product_options, key="analysis_product")
        selected_item = filtered[filtered["product_id"] == selected_product].iloc[0]
        detail_columns = st.columns(3)
        detail_columns[0].metric("Risk level", selected_item["risk_level"])
        detail_columns[1].metric("Stock-out probability", f"{selected_item['stockout_probability']:.1%}")
        detail_columns[2].metric("Recommended reorder", f"{int(selected_item['reorder_quantity']):,} units")
        st.info(selected_item["reason"])
        st.subheader("Related recommendation")
        st.dataframe(
            filtered[filtered["product_id"] == selected_product][[
                "store_id", "product_id", "forecast_7_day", "closing",
                "reorder_quantity", "manager_action"
            ]].rename(columns={
                "store_id": "Store", "product_id": "Product",
                "forecast_7_day": "7-day forecast", "closing": "On hand",
                "reorder_quantity": "Reorder units", "manager_action": "Suggested action"
            }),
            use_container_width=True,
            hide_index=True
        )

with model_tab:
    st.subheader("Demand forecasting")
    st.dataframe(demand_results, use_container_width=True, hide_index=True)
    best_demand_model = demand_results.sort_values("MAE").iloc[0]["Model"]
    st.success(f"Selected demand model: {best_demand_model}")

    st.subheader("Stock-out classification")
    st.dataframe(stockout_results, use_container_width=True, hide_index=True)
    st.subheader("Classification confusion matrix")
    figure, axis = plt.subplots(figsize=(5, 4))
    figure.patch.set_facecolor("#fffefa")
    sns.heatmap(confusion, annot=True, fmt="d", ax=axis, cmap="YlGn",
                linewidths=.5, linecolor="#fffefa")
    axis.set_xlabel("Predicted", color="#68766d")
    axis.set_ylabel("Actual", color="#68766d")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

    st.subheader("Statistical analysis")
    st.dataframe(statistics, use_container_width=True, hide_index=True)
    st.caption("Significance threshold: p < 0.05")

with export_tab:
    st.subheader("Download recommendations")
    st.caption("Export either the current filtered view or the complete recommendation list.")
    export_scope = st.radio(
        "Rows to include",
        ["Current filtered view", "All recommendations"],
        horizontal=True
    )
    export_data = filtered if export_scope == "Current filtered view" else recommendations
    csv = export_data.to_csv(index=False).encode("utf-8")
    export_name = (
        "filtered_reorder_recommendations.csv"
        if export_scope == "Current filtered view"
        else "reorder_recommendations.csv"
    )
    st.download_button(
        label=f"Download {len(export_data):,} recommendations",
        data=csv,
        file_name=export_name,
        mime="text/csv"
    )
    st.subheader("Export preview")
    st.dataframe(export_data.head(12), use_container_width=True, hide_index=True)

st.divider()
st.caption("STOCKSENSE  ·  Inventory intelligence")