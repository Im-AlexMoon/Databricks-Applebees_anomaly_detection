"""Compare accepted cumulative, incremental and repeat-run evidence without cloud access."""
import argparse
from decimal import Decimal
from pathlib import Path
import json


def load(path,name):
    return json.loads((path/name).read_text(encoding='utf-8'))


def verify(before,incremental,repeated,manifest):
    original=load(before,'validation.json');full=load(incremental,'validation.json');again=load(repeated,'validation.json')
    for evidence in (original,full,again):
        assert evidence['status']=='passed' and all(r['passed'] for r in evidence['rules']), 'Unaccepted evidence'
    for folder in (incremental,repeated):
        run=load(folder,'run.json');assert run['state']['result_state']=='SUCCESS','Unsuccessful job'
    a,b,c=original['metrics'],full['metrics'],again['metrics']
    assert full['snapshot_id']==again['snapshot_id']==original['snapshot_id']==manifest['snapshot_id']
    assert a['source_days']+1==b['source_days']==len(manifest['dates'])
    assert b['valid_accounts']==manifest['counts']['checks']
    assert b==c,'Metrics changed when repeating the same snapshot'
    delta=b['valid_accounts']-a['valid_accounts']
    last=max(manifest['dates'])
    assert delta==manifest['dates'][last],'Increment does not match the last source day'
    def canonical(rows):
        return sorted(json.dumps(r,sort_keys=True) for r in rows)
    first_data=load(incremental,'presentation_data.json')['datasets']
    second_data=load(repeated,'presentation_data.json')['datasets']
    for name in first_data:
        assert canonical(first_data[name])==canonical(second_data[name]),'Changed Gold aggregate: '+name
    return {'status':'passed','snapshot_id':full['snapshot_id'],'complete_source_days':b['source_days'],
        'before_accounts':a['valid_accounts'],'after_accounts':b['valid_accounts'],'incremental_accounts':delta,
        'eligible_accounts':b['eligible_accounts'],'recorded_discount_all_accounts':b['check_discount_amount'],
        'eligible_discount':b['eligible_discount'],'repeat_metrics_identical':True,
        'repeat_gold_exports_identical':True,'increment_run_id':load(incremental,'run.json')['run_id'],
        'repeat_run_id':load(repeated,'run.json')['run_id']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('before','incremental','repeated','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=verify(a.before,a.incremental,a.repeated, json.loads(a.manifest.read_text(encoding='utf-8')))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
