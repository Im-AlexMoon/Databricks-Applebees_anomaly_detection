"""Lossless JSON transport to immutable, compressed NDJSON chunks.

No analytical filtering occurs here. Each wrapper keeps its original payload,
source-relative member, row and stable source fingerprint. Employee personal
fields that have no analytical purpose are projected out before cloud transfer.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
import zipfile

import simplejson

from applebees_eda.pipeline import records,sources,signature

VERSION="phase2-transport-v1"
ENTITIES={'revenue_centers':'revenueCenters.json','locations':'locations.json','items':'items.json','employees':'employees.json'}
EMPLOYEE_FIELDS={'Id','LocationId','Deleted','Borrowed','HomeLocationId','PrimaryJobID','JobID'}


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _json(value):
    return simplejson.dumps(value,use_decimal=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def _atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    tmp.replace(path)


def _stage_one(task):
    source,entity,source_info,snapshot,root,max_bytes,prefix,rules_version=task
    source_name=source.split('::',1)[-1] if '::' in source else Path(source).name
    source_id=hashlib.sha256(_json([source_name,source_info]).encode()).hexdigest()[:24]
    folder=Path(root);state=folder/'members'/(entity+'-'+source_id+'.json')
    if state.exists():
        prior=json.loads(state.read_text(encoding='utf-8'))
        if all((folder/x['relative_path']).exists() and sha256_file(folder/x['relative_path'])==x['sha256'] for x in prior['files']):
            return prior
    target=folder/('checks' if entity=='checks' else 'catalogs');target.mkdir(parents=True,exist_ok=True)
    files=[];counts=Counter();stream=None;raw=None;part=0;size=0;part_records=0;temp=None;final=None

    def close_part():
        nonlocal stream,raw
        if stream is not None:
            stream.close();raw.close();temp.replace(final)
            files.append({'relative_path':final.relative_to(folder).as_posix(),'sha256':sha256_file(final),
                          'bytes':final.stat().st_size,'uncompressed_bytes':size,'records':part_records,
                          'entity':entity,'source_id':source_id,'source_file':source_name})
            stream=raw=None
    try:
        for row_number,obj in enumerate(records(source,prefix),1):
            if entity=='employees':
                obj={key:value for key,value in obj.items() if key in EMPLOYEE_FIELDS}
            payload=_json(obj)
            wrapper={'record_uid':f'{source_id}:{row_number}','entity':entity,'source_file':source_name,
                     'source_row':row_number,'source_id':source_id,'snapshot_id':snapshot,
                     'rules_version':rules_version,'payload':payload}
            line=(_json(wrapper)+'\n').encode('utf-8')
            if stream is None or (size+len(line)>max_bytes and part_records):
                close_part();part+=1;size=0;part_records=0
                final=target/f'{entity}-{source_id}-{part:04d}.jsonl.gz';temp=final.with_suffix(final.suffix+'.tmp')
                raw=temp.open('wb');stream=gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=1)
            stream.write(line);size+=len(line);part_records+=1;counts[entity]+=1
            if entity=='checks':
                for key,child in [('ItemsSold','sold_items'),('Payments','payments'),('Voids','voids'),('Tax','taxes'),('Discounts','discounts')]:
                    values=obj.get(key)
                    if isinstance(values,list):
                        counts[child]+=len(values)
                        if key=='ItemsSold':
                            for sold in values:
                                if isinstance(sold,dict) and isinstance(sold.get('Discounts'),list):
                                    counts['discounts']+=len(sold['Discounts'])
                if isinstance(obj.get('BusinessDate'),str):
                    counts['date:'+obj['BusinessDate']]+=1
        close_part()
    finally:
        if stream is not None:
            stream.close()
        if raw is not None and not raw.closed:
            raw.close()
    result={'entity':entity,'source_id':source_id,'source_file':source_name,'counts':dict(counts),'files':files}
    state.parent.mkdir(exist_ok=True);_atomic_json(state,result)
    return result


def prepare_staging(config,root,rule_dictionary,workers=4,max_bytes=128*1024**2,max_sources=None,progress=print):
    files=sources(config.checks)
    if not files:
        raise FileNotFoundError('No check sources match')
    descriptions=[]
    for f in files:
        s=signature(f)
        descriptions.append({'name':f.split('::',1)[-1] if '::' in f else Path(f).name,
                             'size':s['size'],'crc':s.get('crc'),'sha256':None if '::' in f else sha256_file(f)})
    catalog_info={entity:{'name':Path(path).name,'sha256':sha256_file(path)} for entity,path in config.catalogs.items() if Path(path).is_file()}
    snapshot=hashlib.sha256(_json({'sources':descriptions,'catalogs':catalog_info,'transport':VERSION}).encode()).hexdigest()[:20]
    root=Path(root)/snapshot;root.mkdir(parents=True,exist_ok=True);(root/'members').mkdir(exist_ok=True)
    rules_version=sha256_file(rule_dictionary)[:20]
    # CSV interpretation remains versioned separately from the historical data.
    import csv
    rule_records=root/'rules.source.json'
    with Path(rule_dictionary).open(encoding='utf-8-sig',newline='') as f:
        rules=list(csv.DictReader(f))
    rule_records.write_text(_json(rules),encoding='utf-8')
    tasks=[]
    selected=files if max_sources is None else files[:max_sources]
    for source,desc in zip(selected,descriptions):
        tasks.append((source,'checks',desc,snapshot,str(root),max_bytes,config.json_prefixes.get('checks'),rules_version))
    for entity,path in config.catalogs.items():
        if entity in catalog_info:
            tasks.append((path,entity,catalog_info[entity],snapshot,str(root),max_bytes,config.json_prefixes.get(entity),rules_version))
    tasks.append((str(rule_records),'reason_rules',{'sha256':sha256_file(rule_records)},snapshot,str(root),max_bytes,None,rules_version))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(_stage_one,t):t[1] for t in tasks}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            progress(f"Transporte {len(results)}/{len(tasks)}: {result['entity']}, {result['counts'].get(result['entity'],0):,} registros")
    all_counts=Counter();outputs=[];dates=Counter()
    for result in results:
        for key,n in result['counts'].items():
            if key.startswith('date:'):
                dates[key[5:]]+=n
            else:
                all_counts[key]+=n
        outputs.extend(result['files'])
    outputs.sort(key=lambda x:x['relative_path'])
    manifest={'version':VERSION,'snapshot_id':snapshot,'rules_version':rules_version,'counts':dict(all_counts),
              'dates':dict(sorted(dates.items())),'files':outputs,'complete_history':len(selected)==len(files),
              'source_count':len(selected),'expected_source_count':len(files),
              'privacy_projection':{'employees':sorted(EMPLOYEE_FIELDS)},
              'created_utc':datetime.now(timezone.utc).isoformat()}
    _atomic_json(root/('manifest.json' if manifest['complete_history'] else 'manifest_smoke.json'),manifest)
    return root,manifest
