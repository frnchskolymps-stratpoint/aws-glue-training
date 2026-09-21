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
GOLD_WAREHOUSE_PATH = "s3://cheska-s3-capstone/03_gold/"
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
# Dropping duplicates in product to avoid duplicate product_id values in the dimension table
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

# Creation of the fact_events table function -- partitioned by year and month
def build_fact_events(silver_df):
    events_df = (
        silver_df
        .select(
            F.col("event_time").alias("event_time"),
            F.col("event_type").alias("event_type"),
            F.col("product_id").cast("long").alias("product_id"),
            F.col("user_id").cast("long").alias("user_id"),
            F.col("user_session").alias("user_session"),
            F.col("price").cast("double").alias("price"),
        )
        .where(F.col("event_time").isNotNull())
    )

    event_order = Window.orderBy(
        F.col("event_time").asc_nulls_last(),
        F.col("product_id").asc_nulls_last(),
        F.col("user_id").asc_nulls_last(),
    )

    return (
        events_df
        .withColumn(
            "event_id",
            F.row_number().over(event_order).cast("long").alias("event_id"),
        )
    )

# Creation of fact_finance_audit table function & creation of total_records and valid_records columns
def build_fact_finance_audit(silver_df):
    return (
        silver_df
        .withColumn("audit_date", F.to_date(F.date_trunc("month", F.col("event_time"))))
        .groupBy("audit_date")
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
            "total_records",
            "valid_records",
            "data_health_score",
            "monthly_revenue_audit",
        )
    )

def write_iceberg(df, table_name, partition_exprs=None):
    table_identifier = f"{ICEBERG_CATALOG}.`{GOLD_DATABASE}`.{table_name}"
    table_location = f"{GOLD_WAREHOUSE_PATH}{table_name}/"

    writer = (
        df.writeTo(table_identifier)
        .using("iceberg")
        .tableProperty("location", table_location)
        .tableProperty("format-version", "2")
        .tableProperty("write.format.default", "parquet")
    )

    if partition_exprs:
        writer = writer.partitionedBy(*partition_exprs)

    writer.createOrReplace()

# MAIN EXECUTION FLOW OF THE JOB SCRIPT
def main():
    args = getResolvedOptions(sys.argv, ["JOB_NAME"])
    spark_context = SparkContext()
    glue_context = GlueContext(spark_context)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)

    try:
        configure_iceberg(spark)
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {ICEBERG_CATALOG}.`{GOLD_DATABASE}`")

        # Cache the silver dataframe to avoid multiple reads from S3
        silver_df = read_silver(spark).cache()

        # Calling the functions to build the dimension and fact tables
        dim_product_df = build_dim_product(silver_df)
        fact_events_df = build_fact_events(silver_df)
        fact_finance_df = build_fact_finance_audit(silver_df)

        # Debug prints of counts for each dataframe -- to verify the number of rows before writing to Iceberg tables
        print(f"DEBUG: Silver rows = {silver_df.count()}")
        print(f"DEBUG: dim_product rows = {dim_product_df.count()}")
        print(f"DEBUG: fact_events rows = {fact_events_df.count()}")
        print(f"DEBUG: fact_finance_audit rows = {fact_finance_df.count()}")

        # Writing the dataframes of dim_product, fact_events, and fact_finance_audit to their respective Iceberg tables
        write_iceberg(dim_product_df, "dim_product")
        print("DEBUG: dim_product table creation complete")

        write_iceberg(fact_events_df, "fact_events", [F.months("event_time")])
        print("DEBUG: fact_events table creation complete")

        write_iceberg(fact_finance_df, "fact_finance_audit", [F.months("audit_date")])
        print("DEBUG: fact_finance_audit table creation complete")
        
        job.commit()
        print("job completed successfully")
    except Exception as error:
        print(f"job failed: {error}")
        raise


if __name__ == "__main__":
    main()