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
SILVER_WAREHOUSE_PATH = "s3://cheska-s3-capstone/silver/"
QUARANTINE_WAREHOUSE_PATH = "s3://cheska-s3-capstone/quarantine/"
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

# Replace null values in columns that allow nulls with default values for better data quality
def replace_null_values(events_df):
    return events_df.select(
        *[
            F.coalesce(F.col("category_code"), F.lit("unknown category code")).alias(
                "category_code"
            )
            if column_name == "category_code"
            else F.coalesce(F.col("brand"), F.lit("unknown brand")).alias("brand")
            if column_name == "brand"
            else F.coalesce(F.col("user_session"), F.lit("unknown session")).alias(
                "user_session"
            )
            if column_name == "user_session"
            else F.col(column_name)
            for column_name in events_df.columns
        ]
    )

# Write the DataFrame to an Iceberg table
def write_to_iceberg(spark, events_df, table_name, warehouse_path):
    table_identifier = f"{ICEBERG_CATALOG}.`{SILVER_DATABASE}`.{table_name}"
    table_location = f"{warehouse_path}{table_name}/"
    writer = (
        events_df.writeTo(table_identifier)
        .using("iceberg")
        .tableProperty("location", table_location)
        .tableProperty("format-version", "2")
        .tableProperty("write.format.default", "parquet")
    )

    if spark.catalog.tableExists(table_identifier):
        events_df.writeTo(table_identifier).append()
    else:
       writer.partitionedBy("year(event_time)", "month(event_time)").create() # partition the data

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
        spark.sql(
            f"CREATE DATABASE IF NOT EXISTS {ICEBERG_CATALOG}.`{SILVER_DATABASE}`"
        )

        events_dyf = read_datacatalog(
            glue_context, BRONZE_DATABASE, BRONZE_TABLE
        )
        print(f"DEBUG: Raw Events Count = {events_dyf.count()}")

        passed_events_df, quarantine_events_df = evaluate_and_split_dqdl(
            glue_context, events_dyf, DQDL_RULESET
        )
        print(f"DEBUG: Passed Events Count = {passed_events_df.count()}")
        print(f"DEBUG: Quarantined Events Count = {quarantine_events_df.count()}")

        clean_events_df = replace_null_values(passed_events_df).withColumn(
            "event_time", F.to_timestamp("event_time")
        )
        quarantine_events_df = quarantine_events_df.withColumn(
            "event_time", F.to_timestamp("event_time")
        )
        write_to_iceberg(
            spark, clean_events_df, "silver_events", SILVER_WAREHOUSE_PATH
        )
        write_to_iceberg(
            spark,
            quarantine_events_df,
            "quarantine_events",
            QUARANTINE_WAREHOUSE_PATH,
        )

        job.commit()
        print("job completed successfully")
    except Exception as error:
        print(f"job failed: {error}")
        raise


if __name__ == "__main__":
    main()
