import sys
import boto3
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col
from awsglue.dynamicframe import DynamicFrame
from awsgluedq.transforms import EvaluateDataQuality
from pyspark.sql.functions import trim


def read_datacatalog(glueContext, db_name, table_name):
    try:
        return glueContext.create_dynamic_frame.from_catalog(
            database = db_name,
            table_name = table_name
            )
    except Exception as e:
        print(f"Error reading table {table_name}: {e}")
        raise e

# fetching the DQDL ruleset from the Data Catalog using boto3
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
        
        # Safe column dropping list
        dq_cols = ["DataQualityEvaluationResult", "DataQualityRulesPass", "DataQualityRulesFail", "DataQualityRulesSkip"]
        cols_to_drop = [c for c in dq_cols if c in df.columns]
        
        passed_df = df.filter(df["DataQualityEvaluationResult"] == "Passed").drop(*cols_to_drop)
        quarantine_df = df.filter(df["DataQualityEvaluationResult"] == "Failed").drop(*cols_to_drop)
        
        return passed_df, quarantine_df
    except Exception as e:
        print(f"Error processing DQDL evaluation for {ruleset_name}: {e}")
        raise e

        
def write_to_silver_s3 (df, silver_path):
    try:
        df.write.mode("append").format("parquet").save(silver_path)
    except Exception as e:
        print(f"Error saving df to {silver_path}. {e}")
        raise e
    
def main():
    args = getResolvedOptions(sys.argv, ['JOB_NAME'])
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)
    
    try:
        db_name = "bronze-cheska-glue-training"
        orders_table = "orders"
        payments_table = "payments"
        silver_passed_path = "s3://cheska-s3-medallion/silver/passed/"
        silver_quarantine_orders_path = "s3://cheska-s3-medallion/silver/quarantine/orders/"
        silver_quarantine_payments_path = "s3://cheska-s3-medallion/silver/quarantine/payments/"

        # reading the data as catalog tables in DynamicFrames
        orders_dyf = read_datacatalog(glueContext, db_name, orders_table)
        payments_dyf = read_datacatalog(glueContext, db_name, payments_table)

        ### DEBUG PRINT FOR CLOUDWATCH TRACKING -- PRE DQDL
        print(f"DEBUG: Raw Orders Count = {orders_dyf.count()}")
        print(f"DEBUG: Raw Payments Count = {payments_dyf.count()}")
    
        # call the dqdl ruleset function on the orders dataset
        passed_orders_df, quarantine_orders_df = evaluate_and_split_dqdl(
            glueContext, orders_dyf, "orders_dqdl"
        )

        # call the dqdl ruleset function on the payments dataset
        passed_payments_df, quarantine_payments_df = evaluate_and_split_dqdl(
            glueContext, payments_dyf, "payments_dqdl"
        )

        ### DEBUG PRINT FOR CLOUDWATCH TRACKING -- POST DQDL
        print(f"DEBUG: Passed Orders Count = {passed_orders_df.count()}")
        print(f"DEBUG: Quarantined Orders Count = {quarantine_orders_df.count()}")
        print(f"DEBUG: Passed Payments Count = {passed_payments_df.count()}")
        print(f"DEBUG: Quarantined Payments Count = {quarantine_payments_df.count()}")

        write_to_silver_s3(quarantine_orders_df, silver_quarantine_orders_path)
        write_to_silver_s3(quarantine_payments_df, silver_quarantine_payments_path)
        
        # Clean order_id column before join
        passed_orders_df = passed_orders_df.withColumn("order_id", trim(col("order_id")))
        passed_payments_df = passed_payments_df.withColumn("order_id", trim(col("order_id")))

        # Perform inner join
        join_pass_df = passed_orders_df.join(passed_payments_df, on="order_id", how="inner")

        ### DEBUG PRINT FOR CLOUDWATCH TRACKING
        print(f"DEBUG: Joined Passed Count = {join_pass_df.count()}")
        
        write_to_silver_s3(join_pass_df, silver_passed_path)
        
        job.commit()
        print(f"job completed successfully")
        
    except Exception as e:
        print(f"job failed: {e}")
        raise e
        
if __name__ == "__main__":
    main()
