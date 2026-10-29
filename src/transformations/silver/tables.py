"""Silver normalization, account-safe child aggregates, quarantine and civil time."""
import sys
sys.path.insert(0,spark.conf.get('phase2.source_root'))
from pyspark import pipelines as dp
from pyspark.sql import functions as F,types as T,Window
from src.transformations.contracts import CHECK_DDL,CATALOG_DDLS,environment,qualified
from src.transformations.shared_time import local_time_info
from src.transformations import shared_time
from pyspark import cloudpickle
from src.transformations.normalization import check_candidates

# Workers may not mount the driver's Workspace source tree. Serialize this small
# stdlib-only module by value so new executors do not need to import `src`.
cloudpickle.register_pickle_by_value(shared_time)

CATALOG,SCHEMA,SNAPSHOT,RULES=environment(spark)
q=lambda name:qualified(spark,name)


def clean_id(col):
    return F.when(F.length(F.trim(col))>0,col)


def money(name):
    return F.expr(f'try_cast({name} AS DECIMAL(20,6))')


def catalog(entity):
    b=spark.read.table(q('bronze_catalogs')).filter((F.col('entity')==entity)&(F.col('snapshot_id')==SNAPSHOT))
    if entity=='reason_rules':
        b=b.filter(F.col('rules_version')==RULES)
    b=b.groupBy('record_uid').agg(F.min('payload').alias('payload'),F.countDistinct('payload').alias('uid_versions'))
    return b.filter('uid_versions=1').select(F.from_json('payload',CATALOG_DDLS[entity]).alias('r')).select('r.*')


def unique_catalog(entity):
    df=catalog(entity)
    return df.withColumn('id',clean_id(F.col('Id'))).withColumn('id_count',F.count('*').over(Window.partitionBy('Id'))).filter('id_count=1 AND id IS NOT NULL')


@dp.materialized_view(name='silver_locations')
@dp.expect_or_fail('location_unique_id','id IS NOT NULL')
def silver_locations():
    return unique_catalog('locations').select('id',F.col('Name').alias('name'),F.col('TimeZoneName').alias('time_zone'),
        F.expr('try_cast(OpenDate AS DATE)').alias('open_date'),F.expr('try_cast(ClosedDate AS DATE)').alias('closed_date'))


@dp.materialized_view(name='silver_revenue_centers')
def silver_revenue_centers():
    return unique_catalog('revenue_centers').select('id',F.col('Name').alias('name'))


@dp.materialized_view(name='silver_items')
def silver_items():
    return unique_catalog('items').select('id',F.col('Name').alias('name'),F.col('Plu').alias('plu'),F.col('SalesCategoryId').alias('sales_category_id'))


@dp.materialized_view(name='silver_employees')
def silver_employees():
    return unique_catalog('employees').select('id',F.col('LocationId').alias('location_id'),
        F.expr('try_cast(Deleted AS BOOLEAN)').alias('deleted'),F.expr('try_cast(Borrowed AS BOOLEAN)').alias('borrowed'),
        F.col('HomeLocationId').alias('home_location_id'),F.col('PrimaryJobID').alias('primary_job_id'),F.col('JobID').alias('job_ids'))


@dp.materialized_view(name='silver_employee_jobs')
def silver_employee_jobs():
    return spark.read.table(q('silver_employees')).select(F.col('id').alias('employee_id'),F.explode('job_ids').alias('job_id')).distinct()


@dp.materialized_view(name='silver_reason_rules')
def silver_reason_rules():
    df=catalog('reason_rules')
    return df.withColumn('_n',F.count('*').over(Window.partitionBy('reason_key'))).filter('_n=1').drop('_n')


@dp.materialized_view(name='silver_check_candidates',comment='All distinct source records, explicit structural selection status')
def silver_check_candidates():
    return check_candidates(spark.read.table(q('bronze_checks')),SNAPSHOT)


@dp.materialized_view(name='silver_quarantine',comment='Excluded records retain source, payload and a reason')
def silver_quarantine():
    return spark.read.table(q('silver_check_candidates')).filter("selection_status<>'included'").select('record_uid','source_file','source_row','selection_status','payload')


def included():
    return spark.read.table(q('silver_check_candidates')).filter("selection_status='included'")


@dp.materialized_view(name='silver_sold_items')
def silver_sold_items():
    df=included().select('record_uid',F.posexplode('c.ItemsSold').alias('line_index','i'))
    return df.select('record_uid','line_index',clean_id(F.col('i.ItemId')).alias('item_id'),
       clean_id(F.col('i.EmployeeId')).alias('employee_id'),F.col('i.Sequence').alias('sequence'),
       F.col('i.ParentSequence').alias('parent_sequence'),F.col('i.ItemPLU').alias('plu'),
       F.col('i.ItemSubCatId').alias('sub_category_id'),money('i.Quantity').alias('quantity'),
       money('i.GrossPrice').alias('gross_price'),money('i.SoldPrice').alias('sold_price'))


@dp.materialized_view(name='silver_discounts')
def silver_discounts():
    b=included()
    a=b.select('record_uid',F.lit('check').alias('level'),F.lit(None).cast('int').alias('line_index'),F.posexplode('c.Discounts').alias('event_index','d'))
    c=b.select('record_uid',F.posexplode('c.ItemsSold').alias('line_index','i')).select('record_uid',F.lit('item').alias('level'),'line_index',F.posexplode('i.Discounts').alias('event_index','d'))
    return a.unionByName(c).select('record_uid','level','line_index','event_index',F.col('d.DiscountId').alias('discount_id'),
       F.col('d.DiscountName').alias('discount_name'),money('d.Amount').alias('amount'),
       clean_id(F.col('d.EmployeeId')).alias('employee_id'),clean_id(F.col('d.ManagerId')).alias('manager_id'))


@dp.materialized_view(name='silver_payments')
def silver_payments():
    return included().select('record_uid',F.posexplode('c.Payments').alias('event_index','p')).select(
        'record_uid','event_index',F.col('p.PayCatName').alias('category_name'),money('p.Amount').alias('amount'),
        money('p.Gratuity').alias('gratuity'),clean_id(F.col('p.EmployeeId')).alias('employee_id'),
        F.expr('try_cast(p.IsCreditCard AS BOOLEAN)').alias('is_credit_card'))


@dp.materialized_view(name='silver_voids')
def silver_voids():
    return included().select('record_uid',F.posexplode('c.Voids').alias('event_index','v')).select(
        'record_uid','event_index',F.col('v.ItemId').alias('item_id'),money('v.Quantity').alias('quantity'),money('v.Amount').alias('amount'),
        F.col('v.VoidCategoryId').alias('category_id'),F.col('v.VoidCategoryName').alias('category_name'),
        clean_id(F.col('v.EmployeeId')).alias('employee_id'),clean_id(F.col('v.ManagerId')).alias('manager_id'))


@dp.materialized_view(name='silver_taxes')
def silver_taxes():
    return included().select('record_uid',F.posexplode('c.Tax').alias('event_index','t')).select(
        'record_uid','event_index',F.col('t.TaxCategoryId').alias('category_id'),money('t.Amount').alias('amount'))


@dp.materialized_view(name='silver_checks',cluster_by=['location_id','business_day'],comment='One row per valid logical check; line prices and recorded discounts; row-local reconciliation')
@dp.expect_or_fail('valid_account_key','check_id IS NOT NULL AND location_id IS NOT NULL AND business_day IS NOT NULL')
@dp.expect('reconciled_when_eligible','NOT money_eligible OR abs(reconciliation_residual)<=0.02')
@dp.expect('catalog_location','location_matched')
def silver_checks():
    # Every child is aggregated independently before joining, preventing fan-out.
    sql=f'''WITH i AS (SELECT record_uid,count(*) item_count,count(DISTINCT employee_id) seller_count,min(employee_id) only_seller,
       count_if(employee_id IS NULL) missing_seller,sum(gross_price) gross,sum(sold_price) sold,
       count_if(quantity IS NULL OR gross_price IS NULL OR sold_price IS NULL) missing_item_money,
       count_if(quantity<0 OR gross_price<0 OR sold_price<0) negative_items,count_if(quantity<>1) nonunit
       FROM {q('silver_sold_items')} GROUP BY record_uid),
      d AS (SELECT record_uid,count(*) discount_records,sum(amount) discount,count_if(amount>0) positive_discounts,
       sum(CASE WHEN level='check' THEN amount ELSE 0 END) check_discount,
       sum(CASE WHEN level='item' THEN amount ELSE 0 END) item_discount,
       count_if(amount IS NULL) missing_discounts,count_if(amount<0) negative_discounts
       FROM {q('silver_discounts')} GROUP BY record_uid),
      p AS (SELECT record_uid,count(*) payment_records,sum(amount) payment,sum(gratuity) gratuity,
       count_if(amount IS NULL OR gratuity IS NULL) missing_payments,count_if(amount<0) negative_payments,
       bool_or(upper(trim(category_name))='CASH' AND amount>0) has_cash FROM {q('silver_payments')} GROUP BY record_uid),
      t AS (SELECT record_uid,sum(amount) tax,count_if(amount IS NULL) missing_taxes FROM {q('silver_taxes')} GROUP BY record_uid),
      v AS (SELECT record_uid,count(*) void_records,sum(amount) void_amount FROM {q('silver_voids')} GROUP BY record_uid)
      SELECT b.record_uid,b.source_file,b.source_row,b.check_id,b.location_id,b.business_day,
       b.c.RevenueCenterId revenue_center_id,b.c.OpenTime open_time_raw,b.c.CloseTime close_time_raw,
       try_cast(b.c.ServiceCharge AS DECIMAL(20,6)) service_charge,try_cast(b.c.TrafficCount AS DOUBLE) traffic_count,
       l.time_zone,l.id IS NOT NULL location_matched,l.name location_name,
       CASE WHEN i.seller_count=1 AND i.missing_seller=0 THEN i.only_seller END seller_id,
       coalesce(i.item_count,0) item_count,coalesce(i.seller_count,0) seller_count,i.gross,i.sold,
       coalesce(i.missing_item_money,0) missing_item_money,coalesce(i.negative_items,0) negative_items,coalesce(i.nonunit,0) nonunit,
       coalesce(d.discount,0) discount,coalesce(d.check_discount,0) check_discount,coalesce(d.item_discount,0) item_discount,
       coalesce(d.discount_records,0) discount_records,coalesce(d.positive_discounts,0)>0 has_discount,
       coalesce(d.missing_discounts,0) missing_discounts,coalesce(d.negative_discounts,0) negative_discounts,
       coalesce(p.payment_records,0) payment_records,p.payment,coalesce(p.gratuity,0) gratuity,
       coalesce(p.missing_payments,0) missing_payments,coalesce(p.negative_payments,0) negative_payments,coalesce(p.has_cash,false) has_cash,
       coalesce(t.tax,0) tax,coalesce(t.missing_taxes,0) missing_taxes,
       coalesce(v.void_records,0) void_records,coalesce(v.void_amount,0) exported_void_amount
      FROM {q('silver_check_candidates')} b
      LEFT JOIN i USING(record_uid) LEFT JOIN d USING(record_uid) LEFT JOIN p USING(record_uid)
      LEFT JOIN t USING(record_uid) LEFT JOIN v USING(record_uid) LEFT JOIN {q('silver_locations')} l ON b.location_id=l.id
      WHERE b.selection_status='included' '''
    df=spark.sql(sql)
    tf=F.udf(local_time_info,T.StructType([T.StructField('local',T.StringType()),T.StructField('utc',T.StringType()),T.StructField('status',T.StringType())]))
    df=df.withColumn('_open',tf('open_time_raw','time_zone')).withColumn('_close',tf('close_time_raw','time_zone'))
    df=df.withColumn('open_time',F.col('_open.local')).withColumn('open_utc',F.col('_open.utc')).withColumn('open_status',F.col('_open.status'))
    df=df.withColumn('close_time',F.col('_close.local')).withColumn('close_utc',F.col('_close.utc')).withColumn('close_status',F.col('_close.status'))
    df=df.withColumn('week_start',F.to_date(F.date_trunc('week','business_day'))).withColumn('net_sales',F.col('gross')-F.col('discount'))
    df=df.withColumn('open_hour',F.hour(F.expr('try_cast(open_time AS TIMESTAMP_NTZ)')))
    df=df.withColumn('time_band',F.when(F.col('open_hour').isNull(),'unknown').when(F.col('open_hour')<11,'00-10').when(F.col('open_hour')<16,'11-15').when(F.col('open_hour')<21,'16-20').otherwise('21-23'))
    df=df.withColumn('reconciliation_residual',F.col('payment')-(F.col('net_sales')+F.col('tax')+F.col('service_charge')-F.col('gratuity')))
    df=df.withColumn('alternative_discount',F.col('gross')-F.col('sold')+F.col('check_discount'))
    df=df.withColumn('alternative_residual',F.col('payment')-(F.col('sold')-F.col('check_discount')+F.col('tax')+F.col('service_charge')-F.col('gratuity')))
    df=df.withColumn('money_evaluable',F.expr('''item_count>0 AND gross>0 AND payment_records>0 AND payment>=0 AND close_time_raw IS NOT NULL
       AND missing_item_money=0 AND missing_discounts=0 AND missing_payments=0 AND missing_taxes=0 AND service_charge IS NOT NULL
       AND negative_items=0 AND negative_discounts=0 AND negative_payments=0'''))
    df=df.withColumn('money_eligible',F.coalesce(F.expr('money_evaluable AND discount>=0 AND net_sales>=-0.02 AND abs(reconciliation_residual)<=0.02'),F.lit(False)))
    return df.drop('_open','_close')


@dp.materialized_view(name='silver_operational_events',comment='Discount/void event taxonomy retains provisional status and separate valuation')
@dp.expect('reason_mapped','operational_group IS NOT NULL')
def silver_operational_events():
    return spark.sql(f'''WITH e AS (
      SELECT 'discount' kind,record_uid,level,line_index,event_index,discount_id reason_id,discount_name reason_name,
       amount,employee_id,manager_id,NULL::STRING item_id FROM {q('silver_discounts')}
      UNION ALL SELECT 'void',record_uid,'void',NULL::INT,event_index,category_id,category_name,amount,employee_id,manager_id,item_id FROM {q('silver_voids')})
      SELECT e.*,a.location_id,a.business_day,a.week_start,a.money_eligible,
       coalesce(r.operational_group,'pending') operational_group,coalesce(r.review_status,'pending') review_status,
       coalesce(r.subgroup,'unknown') subgroup,m.id IS NOT NULL manager_matched
      FROM e JOIN {q('silver_checks')} a USING(record_uid)
      LEFT JOIN {q('silver_reason_rules')} r ON e.kind=r.kind AND a.location_id=r.location_id
        AND e.reason_id<=>nullif(r.reason_id,'') AND e.reason_name<=>nullif(r.reason_name,'')
      LEFT JOIN {q('silver_employees')} m ON e.manager_id=m.id''')


@dp.materialized_view(name='silver_account_operations')
def silver_account_operations():
    return spark.sql(f'''WITH e AS (SELECT record_uid,
       bool_or(operational_group='correction' AND (kind='void' OR amount>0)) has_correction,
       bool_or(kind='discount' AND operational_group='generic_authorization' AND amount>0) generic_discount,
       bool_or(kind='discount' AND operational_group='correction' AND amount>0) correction_discount,
       bool_or(kind='void' AND operational_group='correction') correction_void,
       sum(CASE WHEN kind='discount' AND operational_group='generic_authorization' AND amount>0 THEN amount ELSE 0 END) generic_amount,
       sum(CASE WHEN kind='discount' AND operational_group='correction' AND amount>0 THEN amount ELSE 0 END) correction_amount
       FROM {q('silver_operational_events')} GROUP BY record_uid)
      SELECT a.*,coalesce(e.has_correction,false) has_correction,coalesce(e.generic_discount,false) generic_discount,
       coalesce(e.correction_discount,false) correction_discount,coalesce(e.correction_void,false) correction_void,
       coalesce(e.generic_amount,0) generic_amount,coalesce(e.correction_amount,0) correction_amount
      FROM {q('silver_checks')} a LEFT JOIN e USING(record_uid)''')
