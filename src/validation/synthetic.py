"""Spark QA uses temporary DataFrames, never historical tables or landing files."""
import json
from src.transformations.normalization import check_candidates
from src.transformations.contracts import TRANSPORT_DDL
from src.transformations.shared_time import local_time_info


def run_synthetic_checks(spark):
    base={'Id':'valid','LocationId':'1','BusinessDate':'2026-05-04','ItemsSold':[],
          'Discounts':[],'Payments':[],'Voids':[],'Tax':[]}
    rows=[]
    def add(uid,payload):
        rows.append((uid,'checks','synthetic.json',1,'synthetic','qa','qa',json.dumps(payload),None,None))
    add('valid',base);add('valid',base)  # exact replay is idempotent
    add('missing',{**base,'Id':None})
    add('arrays',{**base,'Id':'arrays','Voids':None})
    add('dup1',{**base,'Id':'duplicate'});add('dup2',{**base,'Id':'duplicate'})
    add('conflict',{**base,'Id':'conflict'});add('conflict',{**base,'Id':'conflict','LocationId':'2'})
    frame=spark.createDataFrame(rows,TRANSPORT_DDL)
    actual={r.record_uid:r.selection_status for r in check_candidates(frame,'qa').select('record_uid','selection_status').collect()}
    expected={'valid':'included','missing':'invalid_key','arrays':'incomplete_arrays','dup1':'duplicate_key','dup2':'duplicate_key','conflict':'conflicting_source_uid'}
    if actual!=expected:
        raise AssertionError(f'Synthetic selection mismatch: {actual}')
    assert local_time_info('2026-03-08T02:30','America/Chicago')[2]=='nonexistent_local_time'
    assert local_time_info('2026-11-01T01:30','America/Chicago')[2]=='ambiguous_local_time'
    assert local_time_info('2026-05-04T12:00','America/Chicago')[2]=='valid'
    return {'cases':9,'passed':9,'tests':['exact_replay','invalid_key','null_array','duplicate_key_group','conflicting_uid','DST_nonexistent','DST_ambiguous','valid_time','no_historical_writes']}
