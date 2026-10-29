"""Business-ready materialized views. Discounts and void face values stay separate."""
import sys
sys.path.insert(0,spark.conf.get('phase2.source_root'))
from pyspark import pipelines as dp
from src.transformations.contracts import qualified
q=lambda name:qualified(spark,name)


@dp.materialized_view(name='gold_quality',comment='Run validation reads actual balances; expectations do not create a global transaction')
@dp.expect_or_fail('account_conservation','candidate_accounts=valid_accounts+quarantine_accounts')
@dp.expect_or_fail('no_account_fanout','valid_accounts=distinct_valid_accounts')
@dp.expect_or_fail('discount_conservation','abs(check_discount_amount-child_discount_amount)<=0.02')
def gold_quality():
    return spark.sql(f'''SELECT
      (SELECT count(*) FROM {q('bronze_checks')}) bronze_check_records,
      (SELECT count(DISTINCT record_uid) FROM {q('bronze_checks')}) bronze_distinct_records,
      (SELECT count(*) FROM {q('silver_check_candidates')}) candidate_accounts,
      (SELECT count(*) FROM {q('silver_checks')}) valid_accounts,
      (SELECT count(DISTINCT record_uid) FROM {q('silver_checks')}) distinct_valid_accounts,
      (SELECT count(*) FROM {q('silver_quarantine')}) quarantine_accounts,
      (SELECT count(*) FROM {q('silver_sold_items')}) sold_item_records,
      (SELECT count(*) FROM {q('silver_discounts')}) discount_records,
      (SELECT count(*) FROM {q('silver_payments')}) payment_records,
      (SELECT count(*) FROM {q('silver_voids')}) void_records,
      (SELECT count(*) FROM {q('silver_taxes')}) tax_records,
      (SELECT coalesce(sum(discount),0) FROM {q('silver_checks')}) check_discount_amount,
      (SELECT coalesce(sum(amount),0) FROM {q('silver_discounts')}) child_discount_amount,
      (SELECT count(*) FROM {q('silver_checks')} WHERE money_eligible) eligible_accounts,
      (SELECT sum(gross) FROM {q('silver_checks')} WHERE money_eligible) eligible_gross,
      (SELECT sum(discount) FROM {q('silver_checks')} WHERE money_eligible) eligible_discount,
      (SELECT count(*) FROM {q('silver_checks')} WHERE has_discount) accounts_with_discount,
      (SELECT count(*) FROM {q('silver_checks')} WHERE money_evaluable) evaluable_accounts,
      (SELECT count(*) FROM {q('silver_checks')} WHERE money_evaluable AND abs(reconciliation_residual)<=0.02) reconciled_evaluable_accounts,
      (SELECT count(*) FROM {q('silver_checks')} WHERE NOT location_matched) unmatched_locations,
      (SELECT count(*) FROM {q('silver_checks')} WHERE seller_id IS NULL) unattributed_accounts,
      (SELECT count(*) FROM {q('silver_operational_events')} WHERE NOT manager_matched) unmatched_manager_events,
      (SELECT count(*) FROM {q('silver_operational_events')} WHERE operational_group='pending') pending_reason_events,
      (SELECT count(*) FROM {q('silver_account_operations')} WHERE has_correction) correction_accounts,
      (SELECT count(*) FROM {q('silver_account_operations')} WHERE has_discount AND void_records>0) discount_void_accounts,
      (SELECT sum(generic_amount) FROM {q('silver_account_operations')} WHERE money_eligible) eligible_generic_discount,
      (SELECT count(DISTINCT business_day) FROM {q('silver_checks')}) source_days,
      (SELECT min(business_day) FROM {q('silver_checks')}) first_day,
      (SELECT max(business_day) FROM {q('silver_checks')}) last_day''')


@dp.materialized_view(name='gold_discount_reasons',comment='Counts per reason overlap across reasons; only eligible discount amount is monetary')
def gold_discount_reasons():
    return spark.sql(f'''SELECT location_id,week_start,kind,reason_id,reason_name,operational_group,review_status,
      count(*) events,count(DISTINCT record_uid) accounts,
      sum(amount) exported_amount,
      sum(CASE WHEN kind='discount' AND money_eligible THEN amount ELSE 0 END) reconciled_discount_amount,
      count_if(NOT manager_matched) unmatched_manager_events
      FROM {q('silver_operational_events')} GROUP BY ALL''')


@dp.materialized_view(name='gold_operations_weekly',comment='One account enters one local/week bucket before reason joins')
def gold_operations_weekly():
    return spark.sql(f'''SELECT location_id,week_start,count(*) accounts,count(DISTINCT business_day) observed_days,
      count_if(money_eligible) eligible_accounts,count_if(has_correction) correction_accounts,
      count_if(generic_discount) generic_accounts,count_if(has_correction OR generic_discount) broad_accounts,
      count_if(correction_discount) discount_correction_accounts,count_if(correction_void) void_correction_accounts,
      count_if(has_discount AND void_records>0) discount_void_accounts,
      1000.0*count_if(has_correction)/count(*) corrections_per_1000,
      1000.0*count_if(generic_discount)/count(*) generic_per_1000,
      sum(CASE WHEN money_eligible THEN gross ELSE 0 END) gross_sales,
      sum(CASE WHEN money_eligible THEN discount ELSE 0 END) discount_amount,
      sum(CASE WHEN money_eligible THEN net_sales ELSE 0 END) net_sales,
      sum(CASE WHEN money_eligible THEN generic_amount ELSE 0 END) generic_discount_amount,
      sum(exported_void_amount) exported_void_amount
      FROM {q('silver_account_operations')} GROUP BY location_id,week_start''')


@dp.materialized_view(name='gold_product_metrics',comment='Product presence, direct links, modifiers and workflow candidates; no attribution of full-check discount')
def gold_product_metrics():
    return spark.sql(f'''WITH p AS (
      SELECT record_uid,item_id,count(*) sold_lines,sum(gross_price) gross_line_amount,
       bool_or(parent_sequence IS NULL OR parent_sequence IN ('','0')) has_unlinked_line,
       bool_or(parent_sequence IS NOT NULL AND parent_sequence NOT IN ('','0')) has_linked_line
      FROM {q('silver_sold_items')} WHERE item_id IS NOT NULL AND quantity>0 GROUP BY record_uid,item_id),
      direct AS (
       SELECT DISTINCT d.record_uid,s.item_id FROM {q('silver_operational_events')} d
        JOIN {q('silver_sold_items')} s ON d.record_uid=s.record_uid AND d.line_index=s.line_index
        WHERE d.kind='discount' AND d.level='item' AND d.operational_group='correction' AND d.amount>0
       UNION SELECT DISTINCT record_uid,item_id FROM {q('silver_operational_events')} WHERE kind='void' AND operational_group='correction'),
      g AS (SELECT p.item_id,a.location_id,date_trunc('month',a.business_day)::DATE period_month,
       count(*) accounts_with_product,count_if(a.has_correction) correction_accounts,
       count_if(d.record_uid IS NOT NULL) directly_linked_accounts,
       count_if(p.has_unlinked_line) unlinked_accounts,count_if(p.has_linked_line) linked_accounts,
       count_if(p.gross_line_amount>0) positive_price_pairs
       FROM p JOIN {q('silver_account_operations')} a USING(record_uid)
       LEFT JOIN direct d ON p.record_uid=d.record_uid AND p.item_id=d.item_id
       WHERE a.money_eligible GROUP BY p.item_id,a.location_id,date_trunc('month',a.business_day)::DATE)
      SELECT g.*,i.name item_name,
       1000.0*correction_accounts/accounts_with_product corrections_per_1000,
       1000.0*directly_linked_accounts/accounts_with_product direct_per_1000,
       CASE WHEN upper(coalesce(i.name,'')) RLIKE 'DON.?T.*(MAKE|POUR)|PRIORITY[ ]*TCKT' THEN 'workflow_marker_candidate'
        WHEN positive_price_pairs=0 THEN 'nonpriced_item_candidate'
        WHEN unlinked_accounts*2<accounts_with_product THEN 'linked_component_candidate'
        ELSE 'unlinked_item_candidate' END item_role_candidate
      FROM g LEFT JOIN {q('silver_items')} i ON g.item_id=i.id''')
