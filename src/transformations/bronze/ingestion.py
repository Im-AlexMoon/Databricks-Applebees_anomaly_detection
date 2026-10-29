"""Streaming ingestion only. JSON payload and transport provenance stay intact."""
import sys
sys.path.insert(0,spark.conf.get('phase2.source_root'))
from pyspark import pipelines as dp
from pyspark.sql import functions as F
from src.transformations.contracts import TRANSPORT_DDL


def source(folder):
    return (spark.readStream.format('cloudFiles').option('cloudFiles.format','json')
        .option('cloudFiles.schemaEvolutionMode','rescue').option('cloudFiles.maxBytesPerTrigger','256m')
        .option('mode','PERMISSIVE').schema(TRANSPORT_DDL)
        .load(spark.conf.get('phase2.landing_root')+'/'+folder)
        .select('*',F.col('_metadata.file_path').alias('landing_file'))
        .withColumn('ingested_at',F.current_timestamp()))


@dp.table(name='bronze_checks',comment='Immutable check payloads, incremental file ingestion and provenance')
@dp.expect('transport_present','record_uid IS NOT NULL AND payload IS NOT NULL')
@dp.expect('transport_schema','_rescued_data IS NULL AND _corrupt_record IS NULL')
def bronze_checks():
    return source('checks')


@dp.table(name='bronze_catalogs',comment='Versioned catalog and reason-dictionary payloads')
@dp.expect('catalog_transport_present','record_uid IS NOT NULL AND entity IS NOT NULL AND payload IS NOT NULL')
def bronze_catalogs():
    return source('catalogs')
