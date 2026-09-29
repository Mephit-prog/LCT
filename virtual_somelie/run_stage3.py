"""Offline conditional ranking, NOT an online policy evaluation.
Run: python -m virtual_somelie.run_stage3 wines.csv ratings.csv
Check column mapping, timestamps and source dates before running.
"""
import sys
import pandas as pd
from .build_table import build_table
from .evaluate import evaluate, summarize

COLS = dict(wine_id="WineID", user="UserID", rating="Rating", timestamp="Timestamp",
            cat=["Type"], multi=["Grapes"], geo=["Country", "RegionName"],
            num=["ABV", "Body", "Acidity"])  # NO public average/critic rating as input

if __name__ == "__main__":
    wines = pd.read_csv(sys.argv[1]); ratings = pd.read_csv(sys.argv[2])
    ratings = ratings[ratings[COLS["wine_id"]].isin(wines[COLS["wine_id"]])]
    wines = wines.reset_index(drop=True)
    if wines[COLS["wine_id"]].isna().any() or wines[COLS["wine_id"]].duplicated().any():
        raise ValueError("wine IDs must uniquely identify producer, cuvée and vintage")
    pos = {w: i for i, w in enumerate(wines[COLS["wine_id"]])}
    ratings = ratings.assign(wine_idx=ratings[COLS["wine_id"]].map(pos))
    ratings = ratings.rename(columns={COLS["user"]: "user", COLS["rating"]: "rating"})
    ratings = ratings.rename(columns={COLS["timestamp"]: "timestamp"})
    # Explicit repeat tastings must be resolved before splitting; keep the last.
    ratings = ratings.sort_values("timestamp").drop_duplicates(["user", "wine_idx"], keep="last")
    table = build_table(wines, cat_cols=COLS["cat"], multi_cols=COLS["multi"],
                        geo_cols=COLS["geo"], num_cols=COLS["num"])
    res = evaluate(ratings, table, manifest_dir="stage3_manifests")
    print(summarize(res)); res.to_csv("stage3_results.csv", index=False)
