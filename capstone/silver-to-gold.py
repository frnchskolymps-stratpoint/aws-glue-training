import sys
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F, Window

# Paths and names for easier reference
SILVER_DATABASE = "silver-cheska-capstone-db"
SILVER_TABLE = "passed"
GOLD_DATABASE = "gold-cheska-capstone-db"
GOLD_WAREHOUSE_PATH = "s3://cheska-s3-capstone/gold/"
ICEBERG_CATALOG = "glue_catalog"

#  Configuring iceberg 
def configure_iceberg(spark):
    spark.conf.set(
        f"spark.sql.catalog.{ICEBERG_CATALOG}",
        "org.apache.iceberg.spark.SparkCatalog",
    )
    spark.conf.set(
        f"spark.sql.catalog.{ICEBERG_CATALOG}.catalog-impl",
        "org.apache.iceberg.aws.glue.GlueCatalog",
    )
    spark.conf.set(
        f"spark.sql.catalog.{ICEBERG_CATALOG}.io-impl",
        "org.apache.iceberg.aws.s3.S3FileIO",
    )
    spark.conf.set(
        f"spark.sql.catalog.{ICEBERG_CATALOG}.warehouse",
        GOLD_WAREHOUSE_PATH,
    )

# Read the silver table function
def read_silver(spark):
    silver_table = f"{ICEBERG_CATALOG}.`{SILVER_DATABASE}`.{SILVER_TABLE}"
    return spark.table(silver_table)

# Creation of the dim_product table function
def build_dim_product(silver_df):
    return (
        silver_df
        .select(
            F.col("product_id").cast("long").alias("product_id"),
            F.col("category_id").cast("long").alias("category_id"),
            F.col("category_code").alias("category_code"),
            F.col("brand").alias("brand"),
        )
        .where(F.col("product_id").isNotNull())
        .dropDuplicates(["product_id"])
    )

# Creation of the fact_events table function -- partitioned by event_year and event_month
def build_fact_events(silver_df):
    event_order = Window.partitionBy("event_year", "event_month").orderBy(
        F.col("event_time").asc_nulls_last(),
        F.col("product_id").asc_nulls_last(),
        F.col("user_id").asc_nulls_last(),
    )

    return (
        silver_df
        .select(
            F.row_number().over(event_order).cast("long").alias("event_id"),
            F.col("event_time").alias("event_time"),
            F.col("event_type").alias("event_type"),
            F.col("product_id").cast("long").alias("product_id"),
            F.col("user_id").cast("long").alias("user_id"),
            F.col("user_session").alias("user_session"),
            F.col("price").cast("double").alias("price"),
        )
        .where(F.col("event_time").isNotNull())
        .withColumn("event_year", F.year(F.col("event_time")))
        .withColumn("event_month", F.month(F.col("event_time")))
    )

# Creation of fact_finance_data_quality table function & creation of total_records and valid_records columns
def build_fact_finance_data_quality(fact_events_df):
    return (
        fact_events_df
        .withColumn("audit_date", F.to_date(F.date_trunc("month", F.col("event_time"))))
        .withColumn("audit_year", F.year(F.col("audit_date")))
        .withColumn("audit_month", F.month(F.col("audit_date")))
        .groupBy("audit_date", "audit_year", "audit_month")
        .agg(
            F.count("*").alias("total_records"),
            F.sum(F.when(F.col("price").isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias("valid_records"),
            F.sum(F.col("price")).alias("monthly_revenue_audit"),
        )
        .withColumn(
            "data_health_score",
            (F.col("valid_records") / F.col("total_records") * F.lit(100)).cast("double"),
        )
        .select(
            "audit_date",
            "audit_year",
            "audit_month",
            "total_records",
            "valid_records",
            "data_health_score",
            "monthly_revenue_audit",
        )
    )


def write_iceberg(df, table_name, partition_cols=None):
    table_identifier = f"{ICEBERG_CATALOG}.`{GOLD_DATABASE}`.{table_name}"
    table_location = f"{GOLD_WAREHOUSE_PATH}{table_name}/"

    writer = (
        df.writeTo(table_identifier)
        .using("iceberg")
        .tableProperty("location", table_location)
        .tableProperty("format-version", "2")
        .tableProperty("write.format.default", "parquet")
    )

    if partition_cols:
        writer = writer.partitionedBy(*partition_cols)

    writer.createOrReplace()

def main():
    args = getResolvedOptions(sys.argv, ["JOB_NAME"])
    spark_context = SparkContext()
    glue_context = GlueContext(spark_context)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)

    try:
        configure_iceberg(spark)
        print(f"DEBUG: Iceberg catalog configured. Warehouse = {GOLD_WAREHOUSE_PATH}")
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {ICEBERG_CATALOG}.`{GOLD_DATABASE}`")
        print(f"DEBUG: Database created/verified = {ICEBERG_CATALOG}.{GOLD_DATABASE}")

        silver_df = read_silver(spark)

        dim_product_df = build_dim_product(silver_df)
        fact_events_df = build_fact_events(silver_df)
        fact_finance_df = build_fact_finance_data_quality(fact_events_df)

        print(f"DEBUG: Silver rows = {silver_df.count()}")
        print(f"DEBUG: dim_product rows = {dim_product_df.count()}")
        print(f"DEBUG: fact_events rows = {fact_events_df.count()}")
        print(f"DEBUG: fact_finance_data_quality rows = {fact_finance_df.count()}")

        print("DEBUG: Starting creation of dim_product Iceberg table")
        write_iceberg(dim_product_df, "dim_product")
        print("DEBUG: dim_product table creation complete")

        print("DEBUG: Starting creation of fact_events Iceberg table with year/month partitioning")
        write_iceberg(fact_events_df, "fact_events", ["event_year", "event_month"])
        print("DEBUG: fact_events table creation complete")

        print("DEBUG: Starting creation of fact_finance_data_quality Iceberg table with year/month partitioning")
        write_iceberg(fact_finance_df, "fact_finance_data_quality", ["audit_year", "audit_month"])
        print("DEBUG: fact_finance_data_quality table creation complete")

        print(f"DEBUG: Final Iceberg gold setup complete for {ICEBERG_CATALOG}.{GOLD_DATABASE}")

        job.commit()
        print("job completed successfully")
    except Exception as error:
        print(f"job failed: {error}")
        raise


if __name__ == "__main__":
    main()
