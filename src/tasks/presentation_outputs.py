# Databricks notebook source
"""Only bounded aggregates leave Spark for the presentation. Executed after validation."""
import json
from pathlib import Path
for name in ('catalog','schema','manifest_path','source_root'):
    dbutils.widgets.text(name,'')
import sys
sys.path.insert(0,dbutils.widgets.get('source_root'))
from src.transformations.contracts import identifier
catalog=identifier(dbutils.widgets.get('catalog'));schema=identifier(dbutils.widgets.get('schema'))
manifest=json.loads(Path(dbutils.widgets.get('manifest_path')).read_text())
root=Path(manifest['evidence_root']);root.mkdir(parents=True,exist_ok=True)
validation=json.loads((root/'validation.json').read_text())
if validation['status']!='passed':
    raise AssertionError('No approved validation for this batch')
prefix=f'`{catalog}`.`{schema}`'
queries={
 'reason_groups':f'''SELECT kind,operational_group,review_status,sum(events) events,
    sum(reconciled_discount_amount) reconciled_discount_amount,sum(exported_amount) exported_amount
    FROM {prefix}.gold_discount_reasons GROUP BY ALL ORDER BY kind,reconciled_discount_amount DESC''',
 'weekly_operations':f'''SELECT week_start,sum(accounts) accounts,sum(correction_accounts) correction_accounts,
    sum(generic_accounts) generic_accounts,sum(discount_amount) discount_amount,
    sum(generic_discount_amount) generic_discount_amount FROM {prefix}.gold_operations_weekly GROUP BY week_start ORDER BY week_start''',
 'location_definitions':f'''WITH a AS (SELECT location_id,sum(accounts) accounts,sum(correction_accounts) corrections,
    sum(broad_accounts) broad FROM {prefix}.gold_operations_weekly GROUP BY location_id)
    SELECT *,1000.0*corrections/accounts specific_per_1000,1000.0*broad/accounts broad_per_1000,
    rank() OVER(ORDER BY corrections/accounts DESC) specific_rank,rank() OVER(ORDER BY broad/accounts DESC) broad_rank FROM a''',
 'catalog_roles':f'''SELECT item_role_candidate,sum(accounts_with_product) product_presences,
    sum(correction_accounts) correction_presences,sum(directly_linked_accounts) direct_presences
    FROM {prefix}.gold_product_metrics GROUP BY item_role_candidate''',
 'product_examples':f'''SELECT item_id,min(item_name) item_name,min(item_role_candidate) role,
    sum(accounts_with_product) accounts,sum(correction_accounts) correction_accounts,sum(directly_linked_accounts) direct_accounts,
    1000.0*sum(correction_accounts)/sum(accounts_with_product) corrections_per_1000
    FROM {prefix}.gold_product_metrics GROUP BY item_id HAVING sum(accounts_with_product)>=100
    ORDER BY corrections_per_1000 DESC LIMIT 12'''
}
result={'source':'Databricks Gold tables','snapshot_id':manifest['snapshot_id'],'batch_version':manifest['batch_version'],
        'quality':validation['metrics'],'datasets':{},'query_definitions':queries}
for name,query in queries.items():
    rows=spark.sql(query).limit(5000).collect()
    result['datasets'][name]=[row.asDict() for row in rows]
(root/'presentation_data.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
print(json.dumps({'status':'exported','datasets':{k:len(v) for k,v in result['datasets'].items()},'source':'Databricks Gold'}))

# COMMAND ----------
# Bounded read-only outputs are also available in the run's notebook for the demo.
display(spark.table(prefix+'.gold_quality'))

# COMMAND ----------
display(spark.sql(queries['reason_groups']))

# COMMAND ----------
display(spark.sql(queries['weekly_operations']))

# COMMAND ----------
display(spark.sql(queries['product_examples']))
