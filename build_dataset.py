"""
build_dataset.py
Builds the startup dataset:
FRED macro signals  — Fed Funds rate + VIX 
Google Trends       — keywords/timeframe confirmed working in notebook
Binary target       — acquired=1, closed=0
"""

import time
import requests
import pandas as pd
from pytrends.request import TrendReq

FRED_API_KEY = "2edd2c4189fa06c8416a925b6bfa5464"
INPUT_CSV    = "startup data.csv"
OUTPUT_CSV   = "startup_data_clean.csv"

# FRED helper 
def fred_get(series_id: str, start: str, end: str) -> pd.Series:
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&observation_start={start}&observation_end={end}"
        f"&api_key={FRED_API_KEY}&file_type=json"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    obs = r.json()["observations"]
    df  = pd.DataFrame(obs)[["date", "value"]]
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"]


# Load & clean startup data
print("═" * 60)
print("STEP 1 — Loading startup data")
print("═" * 60)

df = pd.read_csv(INPUT_CSV)
print(f"  Raw shape: {df.shape}")

df = df.drop(columns=[
    "Unnamed: 0", "Unnamed: 6",
    "closed_at",     
    "id", "object_id",
    "state_code.1",   
], errors="ignore")

df["target"] = (df["status"] == "acquired").astype(int)
df = df.drop(columns=["status", "labels"])

df["first_funding_at"] = pd.to_datetime(df["first_funding_at"])
df["funding_month"]    = df["first_funding_at"].dt.to_period("M")

print(f"  Cleaned shape : {df.shape}")
print(f"  Acquired      : {df['target'].sum()}")
print(f"  Closed        : {(df['target']==0).sum()}")
print(f"  Date range    : {df['first_funding_at'].min().date()} → "
      f"{df['first_funding_at'].max().date()}")

# FRED macro signals
print("\n" + "═" * 60)
print("STEP 2 — Fetching FRED macro data")
print("═" * 60)

# Fed Funds Rate — monthly
fed = fred_get("FEDFUNDS", "2000-01-01", "2013-12-31").reset_index()
fed.columns      = ["date", "fed_rate"]
fed["funding_month"] = pd.to_datetime(fed["date"]).dt.to_period("M")
print(f"  Fed Funds : {len(fed)} months  "
      f"[{fed['fed_rate'].min():.2f}% – {fed['fed_rate'].max():.2f}%]")

vix = (
    fred_get("VIXCLS", "2000-01-01", "2013-12-31")
    .resample("MS").mean()
    .reset_index()
)
vix.columns      = ["date", "vix"]
vix["funding_month"] = pd.to_datetime(vix["date"]).dt.to_period("M")
print(f"  VIX       : {len(vix)} months  "
      f"[{vix['vix'].min():.1f} – {vix['vix'].max():.1f}]")

# Google Trends sector signals
print("\n" + "═" * 60)
print("STEP 3 — Fetching Google Trends (~30 s)")
print("═" * 60)

KEYWORDS  = ["fintech", "SaaS", "biotech", "AI", "ecommerce"]
TIMEFRAME = "2005-01-01 2014-12-31"

pytrends = TrendReq(hl="en-US", tz=360)
time.sleep(1)
pytrends.build_payload(KEYWORDS, timeframe=TIMEFRAME, geo="US")
trends_raw = pytrends.interest_over_time().reset_index()
trends_raw = trends_raw.drop(columns=["isPartial"], errors="ignore")
trends_raw["funding_month"] = pd.to_datetime(trends_raw["date"]).dt.to_period("M")
print(f"  Trends: {len(trends_raw)} months  "
      f"({trends_raw['date'].min().date()} → {trends_raw['date'].max().date()})")

SECTOR_MAP = [
    ("is_biotech",    "biotech"),
    ("is_ecommerce",  "ecommerce"),
    ("is_software",   "SaaS"),
    ("is_web",        "SaaS"),
    ("is_enterprise", "SaaS"),
    ("is_mobile",     "AI"),
    ("is_advertising","fintech"),
]

def get_trend_score(row):
    match = trends_raw[trends_raw["funding_month"] == row["funding_month"]]
    if match.empty:
        return None        
    m = match.iloc[0]
    for col, kw in SECTOR_MAP:
        if row.get(col, 0) == 1:
            return float(m[kw])
    return float(m[KEYWORDS].mean())   

df["sector_trend"] = df.apply(get_trend_score, axis=1)
covered = df["sector_trend"].notna().sum()
print(f"  Coverage: {covered}/{len(df)} rows ({100*covered/len(df):.1f}%) "
      f"— rows before 2005 get NaN")

# Merge & finalize
print("\n" + "═" * 60)
print("STEP 4 — Merging + saving")
print("═" * 60)

df = df.merge(fed[["funding_month", "fed_rate"]], on="funding_month", how="left")
df = df.merge(vix[["funding_month", "vix"]],      on="funding_month", how="left")
df = df.drop(columns=[c for c in
    ["first_funding_at", "last_funding_at", "founded_at", "funding_month"]
    if c in df.columns])

df.to_csv(OUTPUT_CSV, index=False)

# REPORT
print(f"\n Saved  {OUTPUT_CSV}   ({df.shape[0]} rows × {df.shape[1]} cols)")
print(f"\nAll columns:\n  {df.columns.tolist()}")

print("\n New feature statistics ")
print(df[["fed_rate", "vix", "sector_trend"]].describe().round(3).to_string())

print("\n Missing values ")
print(df[["fed_rate", "vix", "sector_trend"]].isnull().sum().to_string())

print("\n Target split ")
vc = df["target"].value_counts()
print(f"  acquired (1): {vc[1]}  ({100*vc[1]/len(df):.1f}%)")
print(f"  closed   (0): {vc[0]}  ({100*vc[0]/len(df):.1f}%)")

print("\n Sample")
print(df[["fed_rate", "vix", "sector_trend", "target"]].head(15).to_string())
