"""Pure Spark transformations usable both in Lakeflow and independent QA tests."""
from pyspark.sql import functions as F,Window
from .contracts import CHECK_DDL


def clean_id(col):
    return F.when(F.length(F.trim(col))>0,col)


def check_candidates(raw,snapshot):
    b=raw.filter(F.col('snapshot_id')==snapshot)
    b=b.groupBy('record_uid').agg(F.min('source_file').alias('source_file'),F.min('source_row').alias('source_row'),
        F.min('payload').alias('payload'),F.countDistinct('payload').alias('uid_versions'),
        F.max('_rescued_data').alias('_rescued_data'),F.max('_corrupt_record').alias('_transport_corrupt'))
    b=b.withColumn('c',F.from_json('payload',CHECK_DDL)).withColumn('check_id',clean_id(F.col('c.Id')))
    b=b.withColumn('location_id',clean_id(F.col('c.LocationId'))).withColumn('business_day',F.expr('try_cast(c.BusinessDate AS DATE)'))
    b=b.withColumn('logical_key_count',F.count('*').over(Window.partitionBy('location_id','business_day','check_id')))
    arrays=(F.col('c.ItemsSold').isNotNull()&F.col('c.Discounts').isNotNull()&F.col('c.Payments').isNotNull()&
        F.col('c.Voids').isNotNull()&F.col('c.Tax').isNotNull()&F.coalesce(F.forall('c.ItemsSold',lambda x:x['Discounts'].isNotNull()),F.lit(False)))
    b=b.withColumn('arrays_complete',arrays)
    return b.withColumn('selection_status',F.when(F.col('uid_versions')!=1,'conflicting_source_uid')
       .when(F.col('_transport_corrupt').isNotNull()|F.col('c._corrupt_record').isNotNull(),'malformed_payload')
       .when(F.col('check_id').isNull()|F.col('location_id').isNull()|F.col('business_day').isNull(),'invalid_key')
       .when(F.col('logical_key_count')>1,'duplicate_key').when(~F.col('arrays_complete'),'incomplete_arrays').otherwise('included'))
