import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Paths for call-out later
SILVER_PASSED_PATH = "s3://cheska-s3-medallion/silver/passed/"
GOLD_WAREHOUSE_PATH = "s3://cheska-s3-medallion/gold/"
GOLD_DATABASE = "gold-cheska-glue-training"
ICEBERG_CATALOG = "glue_catalog"


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

def read_silver(spark):
    return spark.read.parquet(SILVER_PASSED_PATH)

# Normalization of the data in the silver layer to prepare for gold layer processing
def normalize_silver(silver_df):
    return silver_df.select(
        F.trim(F.col("order_id")).cast("string").alias("order_id"),
        F.trim(F.col("customer_id")).cast("string").alias("customer_id"),
        F.trim(F.col("order_status")).cast("string").alias("order_status"),
        F.to_timestamp(F.col("order_purchase_timestamp")).alias(
            "order_purchase_timestamp"
        ),
        F.to_timestamp(F.col("order_approved_at")).alias("order_approved_at"),
        F.to_timestamp(F.col("order_delivered_carrier_date")).alias(
            "order_delivered_carrier_date"
        ),
        F.to_timestamp(F.col("order_delivered_customer_date")).alias(
            "order_delivered_customer_date"
        ),
        F.to_timestamp(F.col("order_estimated_delivery_date")).alias(
            "order_estimated_delivery_date"
        ),
        F.col("payment_sequential").cast("int").alias("payment_sequential"),
        F.trim(F.col("payment_type")).cast("string").alias("payment_type"),
        F.col("payment_installments").cast("int").alias("payment_installments"),
        F.col("payment_value").cast("decimal(18,2)").alias("payment_value"),
    )

# Creation of dim_orders table
def build_dim_orders(normalized_df):
    order_window = Window.partitionBy("order_id").orderBy(
        F.col("payment_sequential").asc_nulls_last()
    )

    return normalized_df.select(
        "order_id",
        "order_status",
        "payment_type",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "payment_sequential",
    ).withColumn("order_row_number", F.row_number().over(order_window)) \
        .where(F.col("order_row_number") == 1) \
        .drop("order_row_number", "payment_sequential")


# Creation of the dim_orders table
def build_dim_date(dim_orders_df):
    purchase_dates = dim_orders_df.select(
        F.to_date("order_purchase_timestamp").alias("calendar_date")
    ).where(F.col("calendar_date").isNotNull()).dropDuplicates()

    return purchase_dates.select(
        F.date_format("calendar_date", "yyyyMMdd").cast("int").alias("date_key"),
        F.dayofweek("calendar_date").cast("int").alias("day_of_week"),
        F.month("calendar_date").cast("int").alias("month"),
        F.year("calendar_date").cast("int").alias("year"),
    )

# Creation of the fact_orders table
def build_fact_orders(normalized_df):
    return normalized_df.select(
        "order_id",
        "customer_id",
        F.date_format(
            F.to_date("order_purchase_timestamp"), "yyyyMMdd"
        ).cast("int").alias("order_purchase_date_key"),
        "payment_value",
        "payment_installments",
        "payment_sequential",
        F.when(
            F.col("order_delivered_customer_date").isNotNull()
            & F.col("order_estimated_delivery_date").isNotNull()
            & (
                F.col("order_delivered_customer_date")
                > F.col("order_estimated_delivery_date")
            ),
            F.lit(1),
        ).otherwise(F.lit(0)).cast("int").alias("is_late_delivery"),
    )

# Save / write to Iceberg tables in the gold layer
def write_iceberg(df, table_name):
    table_identifier = f"{ICEBERG_CATALOG}.`{GOLD_DATABASE}`.{table_name}"
    table_path = f"{GOLD_WAREHOUSE_PATH}{table_name}/"
    (
        df.writeTo(table_identifier)
        .using("iceberg")
        .tableProperty("location", table_path)
        .tableProperty("format-version", "2")
        .tableProperty("write.format.default", "parquet")
        .createOrReplace()
    )

# Main execution flow
def main():
    args = getResolvedOptions(sys.argv, ["JOB_NAME"])
    spark_context = SparkContext()
    glue_context = GlueContext(spark_context)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)

    try:
        configure_iceberg(spark)
        spark.sql(
            f"CREATE DATABASE IF NOT EXISTS {ICEBERG_CATALOG}.`{GOLD_DATABASE}`"
        )

        silver_df = read_silver(spark)
        normalized_df = normalize_silver(silver_df)

        dim_orders_df = build_dim_orders(normalized_df)
        dim_date_df = build_dim_date(dim_orders_df)
        fact_orders_df = build_fact_orders(normalized_df)

        print(f"DEBUG: Silver rows = {normalized_df.count()}")
        print(f"DEBUG: dim_orders rows = {dim_orders_df.count()}")
        print(f"DEBUG: dim_date rows = {dim_date_df.count()}")
        print(f"DEBUG: fact_orders rows = {fact_orders_df.count()}")

        write_iceberg(dim_orders_df, "dim_orders")
        write_iceberg(dim_date_df, "dim_date")
        write_iceberg(fact_orders_df, "fact_orders")

        job.commit()
        print("job completed successfully")
    except Exception as error:
        print(f"job failed: {error}")
        raise


if __name__ == "__main__":
    main()
