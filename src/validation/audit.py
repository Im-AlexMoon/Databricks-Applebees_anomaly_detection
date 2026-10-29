"""Acceptance rules operate on observed metrics, never on intended success."""
from decimal import Decimal


def assess(metrics,manifest):
    results=[]
    def check(name,actual,expected,tolerance=0):
        if actual is None or expected is None:
            passed=False
        elif tolerance:
            passed=abs(Decimal(str(actual))-Decimal(str(expected)))<=Decimal(str(tolerance))
        else:
            passed=actual==expected
        results.append({'rule':name,'actual':str(actual),'expected':str(expected),'passed':passed,'tolerance':tolerance})
    counts=manifest['counts']
    check('raw_distinct_source_records',int(metrics['bronze_distinct_records']),counts['checks'])
    check('account_conservation',int(metrics['candidate_accounts']),int(metrics['valid_accounts'])+int(metrics['quarantine_accounts']))
    check('no_account_fanout',int(metrics['valid_accounts']),int(metrics['distinct_valid_accounts']))
    check('discount_conservation',metrics['check_discount_amount'],metrics['child_discount_amount'],.02)
    # Historical reference populations are explicit; QA data never enters these inputs.
    if int(metrics['quarantine_accounts'])==0:
        for source,field in [('sold_items','sold_item_records'),('discounts','discount_records'),('payments','payment_records'),('voids','void_records'),('taxes','tax_records')]:
            check('child_count_'+source,int(metrics[field]),counts.get(source,0))
    for field,expected in manifest.get('reference',{}).items():
        if field in metrics:
            check('reference_'+field,metrics[field],expected,.02 if field.startswith('eligible_') and field not in ('eligible_accounts',) else 0)
    return results
