# Databricks notebook source
"""A failed cross-table check fails the job and prevents presentation export."""
import json,sys
from pathlib import Path
from decimal import Decimal
for name in ('catalog','schema','manifest_path','source_root'):
    dbutils.widgets.text(name,'')
sys.path.insert(0,dbutils.widgets.get('source_root'))
from src.transformations.contracts import identifier
from src.validation.audit import assess

catalog=identifier(dbutils.widgets.get('catalog'));schema=identifier(dbutils.widgets.get('schema'))
manifest=json.loads(Path(dbutils.widgets.get('manifest_path')).read_text())
metrics=spark.read.table(f'{catalog}.{schema}.gold_quality').first().asDict()
rules=assess(metrics,manifest)
result={'snapshot_id':manifest['snapshot_id'],'manifest_version':manifest.get('batch_version'),
        'status':'passed' if all(r['passed'] for r in rules) else 'failed','metrics':metrics,'rules':rules}
destination=Path(manifest['evidence_root']);destination.mkdir(parents=True,exist_ok=True)
(destination/'validation.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
print(json.dumps(result,default=str))
if result['status']!='passed':
    raise AssertionError('Cross-table validation failed; downstream evidence export is blocked')
dbutils.jobs.taskValues.set(key='quality_passed',value=True)
