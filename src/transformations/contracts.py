"""Explicit payload contracts. Unknown source fields remain in Bronze payload."""
import re

TRANSPORT_DDL='record_uid STRING, entity STRING, source_file STRING, source_row BIGINT, source_id STRING, snapshot_id STRING, rules_version STRING, payload STRING, _rescued_data STRING, _corrupt_record STRING'
DISCOUNT_DDL='DiscountId STRING, DiscountName STRING, Amount STRING, EmployeeId STRING, ManagerId STRING'
SOLD_DDL=f'''ItemId STRING, EmployeeId STRING, RevenueCenterId STRING, Quantity STRING,
 GrossPrice STRING, SoldPrice STRING, OrderTime STRING, ItemPLU STRING,
 ItemMajorCatId STRING, ItemMajorCatName STRING, ItemSubCatId STRING, ItemSubCatName STRING,
 Sequence STRING, ParentSequence STRING, PosSourceId STRING, Discounts ARRAY<STRUCT<{DISCOUNT_DDL}>>'''
CHECK_DDL=f'''Id STRING, LocationId STRING, BusinessDate STRING, RevenueCenterId STRING,
 OpenTime STRING, CloseTime STRING, ServiceCharge STRING, TrafficCount STRING,
 ItemsSold ARRAY<STRUCT<{SOLD_DDL}>>, Discounts ARRAY<STRUCT<{DISCOUNT_DDL}>>,
 Payments ARRAY<STRUCT<PaymentCategoryId STRING, PayCatName STRING, Amount STRING, Gratuity STRING, EmployeeId STRING, PosSourceId STRING, IsCreditCard STRING>>,
 Voids ARRAY<STRUCT<ItemId STRING, EmployeeId STRING, ManagerId STRING, Quantity STRING, Amount STRING, VoidCategoryId STRING, VoidCategoryName STRING>>,
 Tax ARRAY<STRUCT<TaxCategoryId STRING, Amount STRING>>, _corrupt_record STRING'''
CATALOG_DDLS={
 'locations':'Id STRING, Name STRING, OpenDate STRING, ClosedDate STRING, TimeZoneName STRING',
 'revenue_centers':'Id STRING, Name STRING',
 'items':'Id STRING, Name STRING, Plu STRING, SalesCategoryId STRING, ExternalId STRING',
 'employees':'Id STRING, LocationId STRING, Deleted STRING, Borrowed STRING, HomeLocationId STRING, PrimaryJobID STRING, JobID ARRAY<STRING>',
 'reason_rules':'reason_key STRING, kind STRING, location_id STRING, reason_id STRING, reason_name STRING, operational_group STRING, subgroup STRING, review_status STRING, interpretation STRING, notes STRING',
}


def identifier(value):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',value):
        raise ValueError('Invalid SQL identifier')
    return value


def environment(spark):
    catalog=identifier(spark.conf.get('phase2.catalog'))
    schema=identifier(spark.conf.get('phase2.schema'))
    return catalog,schema,spark.conf.get('phase2.snapshot_id'),spark.conf.get('phase2.rules_version')


def qualified(spark,table):
    catalog,schema,_,_=environment(spark)
    return f'`{catalog}`.`{schema}`.`{identifier(table)}`'
