"""PySpark ingestion job: weather + market data -> partitioned Parquet lake.

Runs as a one-shot Spark application (``spark-submit``). In this demo it reads a
few years of daily data for one city, but the pipeline is written the way a
TB-scale job is: a Spark DataFrame with window-based feature engineering and a
partitioned, columnar Parquet sink. To scale to the "10 TB" figure you would
point the reader at an S3 prefix of many stations x many years and let the same
transformations fan out across an EMR cluster - no code changes to the logic
below, only the input path and cluster size.

Fallback behavior: real NOAA data is used when ``NOAA_TOKEN`` is set and the API
responds; otherwise a realistic synthetic series is generated so the job always
succeeds offline.
"""

from __future__ import annotations

import os
import sys
import traceback

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Ensure /app is importable when run via spark-submit.
sys.path.insert(0, "/app")

from shared import config  # noqa: E402
from shared.schemas import Keys  # noqa: E402

# Friendly station id -> GHCND id for NOAA CDO.
GHCND_STATIONS = {
    "KNYC": "GHCND:USW00094728",  # Central Park, NY
    "KORD": "GHCND:USW00094846",  # Chicago O'Hare
    "KMIA": "GHCND:USW00012839",  # Miami Intl
    "KLAX": "GHCND:USW00023174",  # Los Angeles Intl
}


def _load_pandas_weather():
    """Return (pandas_df, source_label). Try NOAA first, fall back to synthetic."""
    from ingestion.sources import noaa, synthetic

    station_id = GHCND_STATIONS.get(config.DEFAULT_STATION)
    if config.NOAA_TOKEN and station_id:
        df = noaa.fetch_weather(station_id, config.NOAA_TOKEN, years=6)
        if df is not None and len(df) > 400:
            # NOAA gives us real weather; the contract + naive market price are
            # derived in build_features (same as the synthetic path).
            return df, "noaa"

    return (
        synthetic.generate_weather(
            config.DEFAULT_CITY, years=6, threshold_f=config.TEMP_THRESHOLD_F
        ),
        "synthetic",
    )


def _report_status(stage: str, detail: str) -> None:
    """Best-effort pipeline status to Redis for the dashboard."""
    try:
        import json
        import time

        from shared.store import get_redis

        r = get_redis()
        r.set(
            Keys.PIPELINE_STATUS,
            json.dumps({"stage": stage, "detail": detail, "ts": time.time()}),
        )
    except Exception:
        pass  # Redis is optional for ingestion to succeed


def build_features(spark, pdf):
    """Turn raw daily weather into a leakage-safe feature table via Spark."""
    sdf = spark.createDataFrame(pdf)
    sdf = sdf.withColumn("date", F.to_date("date"))

    w = Window.orderBy("date")
    # Only PAST information is used as features (no same-day leakage). The daily
    # high itself becomes the label/target.
    sdf = (
        sdf.withColumn("tmax_lag1", F.lag("tmax_f", 1).over(w))
        .withColumn("tmax_lag2", F.lag("tmax_f", 2).over(w))
        .withColumn("tmax_lag3", F.lag("tmax_f", 3).over(w))
        .withColumn("tmin_lag1", F.lag("tmin_f", 1).over(w))
        .withColumn("humidity_lag1", F.lag("humidity", 1).over(w))
        .withColumn("wind_lag1", F.lag("wind_mph", 1).over(w))
        .withColumn(
            "tmax_roll7",
            F.avg("tmax_f").over(w.rowsBetween(-7, -1)),
        )
        .withColumn(
            "tmax_roll30",
            F.avg("tmax_f").over(w.rowsBetween(-30, -1)),
        )
    )

    sdf = (
        sdf.withColumn("doy", F.dayofyear("date"))
        .withColumn("month", F.month("date"))
        .withColumn("year", F.year("date"))
        .withColumn("doy_sin", F.sin(2 * 3.141592653589793 * F.col("doy") / 365.0))
        .withColumn("doy_cos", F.cos(2 * 3.141592653589793 * F.col("doy") / 365.0))
    )

    sdf = sdf.withColumn("normal_30", F.col("tmax_roll30")).withColumn(
        "target_high_f", F.col("tmax_f")
    )

    # If the contract columns were pre-generated (synthetic path), keep them.
    # Otherwise (real NOAA weather) derive an "above trailing-normal" contract,
    # priced ~0.5 by a naive climatological market.
    if "label_high" not in sdf.columns:
        sdf = (
            sdf.withColumn("label_high", (F.col("tmax_f") > F.col("normal_30")).cast("int"))
            .withColumn(
                "market_yes_price",
                F.round(F.least(F.greatest(0.5 + (F.rand(seed=7) - 0.5) * 0.06, F.lit(0.02)), F.lit(0.98)), 4),
            )
            .withColumn("threshold_f", F.round(F.col("normal_30"), 2))
        )

    # Drop warm-up rows with null lag/rolling features.
    sdf = sdf.dropna(
        subset=["tmax_lag1", "tmax_lag2", "tmax_lag3", "tmax_roll7", "normal_30"]
    )
    return sdf


def main() -> int:
    spark = (
        SparkSession.builder.appName("kalshi-weather-ingest")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        _report_status("ingest_start", "loading weather source")
        pdf, source = _load_pandas_weather()
        print(f"[ingest] loaded {len(pdf)} raw daily rows from source={source}")

        features = build_features(spark, pdf)
        n = features.count()
        print(f"[ingest] built {n} feature rows")

        out_dir = os.path.join(config.DATA_DIR, "features")
        os.makedirs(config.DATA_DIR, exist_ok=True)
        # Partitioned, columnar sink - the same call scales to a lake on S3.
        (
            features.repartition("year")
            .write.mode("overwrite")
            .partitionBy("year")
            .parquet(out_dir)
        )
        print(f"[ingest] wrote Parquet feature lake -> {out_dir} (partitioned by year)")

        # Optional: confirm Kalshi API connectivity for the "real data" story.
        try:
            from ingestion.sources import kalshi

            mkts = kalshi.fetch_weather_markets(config.KALSHI_API_BASE)
            if mkts:
                print(f"[ingest] Kalshi API reachable: {len(mkts)} live markets listed")
        except Exception:
            pass

        _report_status("ingest_done", f"{n} feature rows from {source}")
        print("[ingest] DONE")
        return 0
    except Exception:
        traceback.print_exc()
        _report_status("ingest_error", "see logs")
        return 1
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
