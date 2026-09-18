import sys
import boto3
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsgluedq.transforms import EvaluateDataQuality
from pyspark.sql import functions as F


BRONZE_DATABASE = "bronze-cheska-capstone-db"
BRONZE_TABLE = "bronze"
DQDL_RULESET = "capstone-dqdl-ruleset"
SILVER_DATABASE = "silver-cheska-capstone-db"
SILVER_WAREHOUSE_PATH = "s3://cheska-s3-capstone/02_silver/"
WAREHOUSE_BASE_PATH = "s3://cheska-s3-capstone/"
QUARANTINE_WAREHOUSE_PATH = "s3://cheska-s3-capstone/04_quarantine/"
ICEBERG_CATALOG = "glue_catalog"

# Reading from the Data Catalog using GlueContext
def read_datacatalog(glueContext, db_name, table_name):
    try:
        return glueContext.create_dynamic_frame.from_catalog(
            database = db_name,
            table_name = table_name
            )
    except Exception as e:
        print(f"Error reading table {table_name}: {e}")
        raise e

# Fetching the DQDL ruleset made from the Data Catalog using boto3
def get_catalog_ruleset(ruleset_name):
    try:
        glue_client = boto3.client('glue')
        response = glue_client.get_data_quality_ruleset(Name=ruleset_name)
        ruleset = response.get('Ruleset')
        if not ruleset:
            raise ValueError(f"Data Catalog ruleset '{ruleset_name}' is empty")
        return ruleset
    except Exception as e:
        print(f"Error retrieving ruleset '{ruleset_name}' from Data Catalog: {e}")
        raise e

# Evaluating the DQDL ruleset retrieved and splitting the data into passed and quarantine DataFrames
def evaluate_and_split_dqdl(glueContext, dynamic_frame, ruleset_name):
    try:
        dqdl_string = get_catalog_ruleset(ruleset_name)
        
        # Execute DQ processing
        dq_results = EvaluateDataQuality().process_rows(
            frame=dynamic_frame,
            ruleset=dqdl_string,
            publishing_options={
                "dataQualityEvaluationContext": ruleset_name,
                "enableDataQualityCloudWatchMetrics": True,
                "enableDataQualityResultsPublishing": True
            },
            additional_options={
                "performanceTuning.caching": "CACHE_NOTHING",
                "observations.scope": "ALL"
            }
        )
        
        # Extract 'rowLevelOutcomes' which holds the original rows + DataQualityEvaluationResult
        df = dq_results["rowLevelOutcomes"].toDF()
        
        # Dropping the DQDL evaluation result columns
        dq_cols = ["DataQualityEvaluationResult", "DataQualityRulesPass", "DataQualityRulesFail", "DataQualityRulesSkip"]
        cols_to_drop = [c for c in dq_cols if c in df.columns]
        
        passed_df = df.filter(df["DataQualityEvaluationResult"] == "Passed").drop(*cols_to_drop)
        quarantine_df = df.filter(df["DataQualityEvaluationResult"] == "Failed").drop(*cols_to_drop)
        
        return passed_df, quarantine_df
    except Exception as e:
        print(f"Error processing DQDL evaluation for {ruleset_name}: {e}")
        raise e

# Configure Iceberg catalog settings for Spark
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
        SILVER_WAREHOUSE_PATH,
    )

# Replace null or whitespace-only values and trim string columns.
def normalize_events(events_df):
    return (
        events_df
        .withColumn(
            "category_code",
            F.coalesce(
                F.nullif(F.trim(F.col("category_code")), F.lit("")),
                F.lit("unknown category code"),
            ),
        )
        .withColumn(
            "brand",
            F.coalesce(
                F.nullif(F.trim(F.col("brand")), F.lit("")),
                F.lit("unknown brand"),
            ),
        )
        .withColumn(
            "user_session",
            F.coalesce(
                F.nullif(F.trim(F.col("user_session")), F.lit("")),
                F.lit("unknown session"),
            ),
        )
        .withColumn("event_type", F.trim(F.col("event_type")))
        .withColumn("event_time", F.to_timestamp("event_time"))
    )

# Creation of the Iceberg schema table
def create_tables_if_not_exists(spark):
    # Silver Table
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {ICEBERG_CATALOG}.`{SILVER_DATABASE}`.passed (
            event_time TIMESTAMP,
            event_type STRING,
            product_id BIGINT,
            category_id BIGINT,
            category_code STRING,
            brand STRING,
            price DOUBLE,
            user_id BIGINT,
            user_session STRING
        )
        USING iceberg
        PARTITIONED BY (months(event_time))
        LOCATION '{SILVER_WAREHOUSE_PATH}passed/'
        TBLPROPERTIES (
            'format-version' = '2',
            'write.format.default' = 'parquet',
            'write.spark.fanout.enabled' = 'true'
        )
    """)

    # Quarantine Table
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {ICEBERG_CATALOG}.`{SILVER_DATABASE}`.quarantined (
            event_time TIMESTAMP,
            event_type STRING,
            product_id BIGINT,
            category_id BIGINT,
            category_code STRING,
            brand STRING,
            price DOUBLE,
            user_id BIGINT,
            user_session STRING
        )
        USING iceberg
        PARTITIONED BY (months(event_time))
        LOCATION '{QUARANTINE_WAREHOUSE_PATH}quarantined/'
        TBLPROPERTIES (
            'format-version' = '2',
            'write.format.default' = 'parquet',
            'write.spark.fanout.enabled' = 'true'
        )
    """)

# Writing the data to the created iceberg table
def write_to_iceberg(spark, events_df, table_name):
    table_identifier = f"{ICEBERG_CATALOG}.`{SILVER_DATABASE}`.`{table_name}`"
    source_view = f"{table_name}_source"

    events_df.createOrReplaceTempView(source_view)

    spark.sql(f"""
        INSERT INTO {table_identifier} (
            event_time, event_type, product_id, category_id,
            category_code, brand, price, user_id, user_session
        )
        SELECT 
            event_time, event_type, product_id, category_id,
            category_code, brand, price, user_id, user_session
        FROM {source_view}
    """)

# Main function for the execution flow of the Glue job
def main():
    args = getResolvedOptions(sys.argv, ["JOB_NAME"])
    spark_context = SparkContext()
    glue_context = GlueContext(spark_context)
    spark = glue_context.spark_session
    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)

    try:
        configure_iceberg(spark)
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {ICEBERG_CATALOG}.`{SILVER_DATABASE}`")

        # Initialize Iceberg tables cleanly before processing
        create_tables_if_not_exists(spark)

        # Ingest from Data Catalog
        events_dyf = read_datacatalog(glue_context, BRONZE_DATABASE, BRONZE_TABLE)

        # Run DQDL split
        passed_df, quarantined_df = evaluate_and_split_dqdl(
            glue_context, events_dyf, DQDL_RULESET
        )

        passed_df.cache()
        quarantined_df.cache()

        print(f"DEBUG: Passed Events Count = {passed_df.count()}")
        print(f"DEBUG: Quarantined Events Count = {quarantined_df.count()}")

        # Clean and prepare data
        passed_events_df = normalize_events(passed_df)
        quarantine_events_df = normalize_events(quarantined_df)

        # FOR DEBUGGING: counting nulls before normalization
        null_category_before = passed_df.filter(F.col("category_code").isNull() | (F.trim(F.col("category_code")) == "")).count()
        null_brand_before = passed_df.filter(F.col("brand").isNull() | (F.trim(F.col("brand")) == "")).count()
        null_session_before = passed_df.filter(F.col("user_session").isNull() | (F.trim(F.col("user_session")) == "")).count()

        # FOR DEBUGGING: counting nulls after normalization
        null_category_after = passed_events_df.filter(F.col("category_code").isNull()).count()
        null_brand_after = passed_events_df.filter(F.col("brand").isNull()).count()
        null_session_after = passed_events_df.filter(F.col("user_session").isNull()).count()

        # FOR DEBUGGING: counting the replaced values during normalization
        unknown_category_count = passed_events_df.filter(F.col("category_code") == "unknown category code").count()
        unknown_brand_count = passed_events_df.filter(F.col("brand") == "unknown brand").count()
        unknown_session_count = passed_events_df.filter(F.col("user_session") == "unknown session").count()

        print(f"DEBUG: null category_code = {null_category_before} | Replaced null category_code = {unknown_category_count} | Remaining NULLs = {null_category_after}")
        print(f"DEBUG: null brand = {null_brand_before} | Replaced null brand = {unknown_brand_count} | Remaining NULLs = {null_brand_after}")
        print(f"DEBUG: null user_session = {null_session_before} | Replaced null user_session = {unknown_session_count} | Remaining NULLs = {null_session_after}")

        # Write to Iceberg tables
        write_to_iceberg(spark, passed_events_df, "passed")
        write_to_iceberg(spark, quarantine_events_df, "quarantined")

        job.commit()
        print("job completed successfully")
    except Exception as error:
        print(f"job failed: {error}")
        raise


if __name__ == "__main__":
    main()
