"""Cumulative batches use member receipts, retaining the same immutable snapshot."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import re

from .staging import _atomic_json


def through_day(manifest_path,day):
    root=Path(manifest_path).parent
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    selected=[]
    for path in sorted((root/'members').glob('*.json')):
        member=json.loads(path.read_text(encoding='utf-8'))
        if member['entity']=='checks':
            dates=[k[5:] for k in member['counts'] if k.startswith('date:')]
            if not dates or max(d[:10] for d in dates)>day:
                continue
        selected.append(member)
    counts=Counter();dates=Counter();files=[]
    for member in selected:
        files.extend(member['files'])
        for key,n in member['counts'].items():
            if key.startswith('date:'):
                dates[key[5:]]+=n
            else:
                counts[key]+=n
    if not counts.get('checks'):
        raise ValueError('No check records selected')
    result=deepcopy(manifest)
    result.update(counts=dict(counts),dates=dict(sorted(dates.items())),files=sorted(files,key=lambda f:f['relative_path']),
                  source_count=sum(m['entity']=='checks' for m in selected))
    result['complete_history']=result['source_count']==result['expected_source_count']
    target=root/('manifest_through_'+day+'.json');_atomic_json(target,result)
    return target


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('--through',required=True)
    a=p.parse_args()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',a.through):
        p.error('--through must be YYYY-MM-DD')
    print(through_day(a.manifest,a.through))
