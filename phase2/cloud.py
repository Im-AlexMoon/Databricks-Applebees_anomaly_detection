"""Official SDK/CLI access. OAuth secrets stay in the operating system secure store."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse
import json
import os
import subprocess
import sys

from .staging import sha256_file,_atomic_json

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_HOST='https://dbc-8785bcd0-4e22.cloud.databricks.com'


def cli_path():
    candidate=ROOT/'.cache'/'tools'/'databricks'/'1.22.1'/'databricks.exe'
    return os.environ.get('DATABRICKS_CLI_PATH',str(candidate))


def client():
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.core import Config
    config=ROOT/'.cache'/'databricks'/'config'
    os.environ['DATABRICKS_CONFIG_FILE']=str(config)
    return WorkspaceClient(config=Config(host=os.environ.get('DATABRICKS_HOST',DEFAULT_HOST),profile='APPLEBEES_PHASE2',
        config_file=str(config),databricks_cli_path=Path(cli_path()).as_posix(),auth_type='databricks-cli'))


def bootstrap(w,catalog='workspace',schema='applebees_phase2'):
    from databricks.sdk.errors import NotFound,ResourceDoesNotExist
    from databricks.sdk.service.catalog import VolumeType
    try:
        w.schemas.get(f'{catalog}.{schema}')
    except (NotFound,ResourceDoesNotExist):
        w.schemas.create(schema,catalog,comment='Applebees Phase 2 educational delivery. Provisional business semantics.')
    try:
        w.volumes.read(f'{catalog}.{schema}.landing')
    except (NotFound,ResourceDoesNotExist):
        w.volumes.create(catalog,schema,'landing',VolumeType.MANAGED,comment='Private immutable source snapshots and execution evidence')
    w.workspace.mkdirs('/Shared/applebees_phase2')
    return f'/Volumes/{catalog}/{schema}/landing'


def reference_for(manifest):
    """Compare exact source members with the established local account population."""
    import duckdb
    base=ROOT/'outputs'/'eda_iterativo'/'inputs-f64a99f8-0e49808848'/'accounts.parquet'
    if not base.exists():
        return {}
    c=duckdb.connect();c.execute("SET memory_limit='1GB'")
    c.read_parquet(str(base)).create_view('a')
    members=sorted({f['source_file'] for f in manifest['files'] if f['entity']=='checks'})
    if not members:
        return {}
    c.execute('CREATE TEMP TABLE source_members(name VARCHAR)')
    c.executemany('INSERT INTO source_members VALUES (?)',[(n,) for n in members])
    result=c.execute('''SELECT count(*) valid_accounts,count(*) FILTER(WHERE eligible) eligible_accounts,
       count(*) FILTER(WHERE has_discount) accounts_with_discount,
       sum(gross) FILTER(WHERE eligible) eligible_gross,sum(discount) FILTER(WHERE eligible) eligible_discount,
       count(DISTINCT business_day) source_days FROM a WHERE EXISTS(SELECT 1 FROM source_members s WHERE ends_with(a.source_file,s.name))''').fetchdf().iloc[0].to_dict()
    c.close()
    return {k:int(v) if k in ('valid_accounts','eligible_accounts','accounts_with_discount','source_days') else float(v) for k,v in result.items()}


def upload_manifest(w,manifest_path,catalog='workspace',schema='applebees_phase2',workers=4,progress=print):
    from databricks.sdk.errors import NotFound,ResourceDoesNotExist
    manifest_path=Path(manifest_path).resolve();root=manifest_path.parent
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    volume=bootstrap(w,catalog,schema)
    landing=f"{volume}/snapshots/{manifest['snapshot_id']}"
    for directory in ('checks','catalogs','manifests','evidence'):
        w.files.create_directory(landing+'/'+directory)
    receipt_path=root/'upload_receipts.json'
    receipts=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    def send(entry):
        local=root/entry['relative_path'];remote=landing+'/'+entry['relative_path']
        if sha256_file(local)!=entry['sha256']:
            raise AssertionError('Local staged file hash changed')
        try:
            metadata=w.files.get_metadata(remote)
            if metadata.content_length!=entry['bytes']:
                raise AssertionError('Immutable remote file has conflicting length: '+entry['relative_path'])
            if receipts.get(remote)!=entry['sha256']:
                # Unknown pre-existing file must be checked before trusting it.
                import hashlib
                h=hashlib.sha256();response=w.files.download(remote)
                for chunk in iter(lambda:response.contents.read(4*1024*1024),b''):
                    h.update(chunk)
                response.contents.close()
                if h.hexdigest()!=entry['sha256']:
                    raise AssertionError('Immutable remote file hash conflict')
        except (NotFound,ResourceDoesNotExist):
            with local.open('rb') as f:
                w.files.upload(remote,f,overwrite=False)
            if w.files.get_metadata(remote).content_length!=entry['bytes']:
                raise AssertionError('Upload length mismatch')
        return remote,entry['sha256']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(send,e) for e in manifest['files']]
        for i,future in enumerate(as_completed(futures),1):
            remote,digest=future.result();receipts[remote]=digest
            _atomic_json(receipt_path,receipts)
            progress(f'Upload {i}/{len(futures)} verified')
    manifest['landing_root']=landing
    manifest['reference']=reference_for(manifest)
    batch=__import__('hashlib').sha256(json.dumps([(f['relative_path'],f['sha256']) for f in manifest['files']]).encode()).hexdigest()[:16]
    manifest['batch_version']=batch
    manifest['evidence_root']=landing+'/evidence/'+batch
    w.files.create_directory(manifest['evidence_root'])
    remote_manifest=landing+'/manifests/'+batch+'.json'
    import io
    w.files.upload(remote_manifest,io.BytesIO(json.dumps(manifest,indent=2).encode()),overwrite=True)
    variables={'catalog':catalog,'schema':schema,'snapshot_id':manifest['snapshot_id'],'rules_version':manifest['rules_version'],
               'landing_root':landing,'manifest_path':remote_manifest}
    state={'host':w.config.host,'variables':variables,'local_manifest':str(manifest_path),'batch_version':batch}
    state_path=ROOT/'.cache'/'databricks'/'deployment.json';state_path.parent.mkdir(exist_ok=True)
    _atomic_json(state_path,state)
    progress('Sources ready; reference counts: '+json.dumps(manifest['reference']))
    return state


def bundle_command(command,state,extra=(),capture=False):
    env=os.environ.copy();env['DATABRICKS_CONFIG_FILE']=str(ROOT/'.cache'/'databricks'/'config')
    for key,value in state['variables'].items():
        env['BUNDLE_VAR_'+key]=str(value)
    result=subprocess.run([cli_path(),'bundle',*command,'--profile','APPLEBEES_PHASE2',*extra],cwd=ROOT,env=env,text=True,capture_output=capture)
    if result.returncode:
        raise RuntimeError('Databricks bundle command failed')
    return result.stdout


def resource_ids(state):
    summary=json.loads(bundle_command(['summary'],state,extra=('-o','json'),capture=True))
    return {'job_id':int(summary['resources']['jobs']['applebees_job']['id']),
            'pipeline_id':summary['resources']['pipelines']['applebees_pipeline']['id']}


def run_status(w,run_id,state):
    run=w.jobs.get_run(run_id)
    result={'run_id':run_id,'run_page_url':run.run_page_url,'state':run.state.as_dict() if run.state else {},'tasks':[]}
    for task in run.tasks or []:
        info={'task_key':task.task_key,'run_id':task.run_id,'state':task.state.as_dict() if task.state else {}}
        if task.state and task.state.result_state and str(task.state.result_state.value)!='SUCCESS':
            output=w.jobs.get_run_output(task.run_id)
            info['error']=output.error;info['error_trace']=output.error_trace
        result['tasks'].append(info)
    output_dir=ROOT/'outputs'/'phase2_cloud'/state['batch_version'];output_dir.mkdir(parents=True,exist_ok=True)
    _atomic_json(output_dir/f'run-{run_id}.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    up=sub.add_parser('upload');up.add_argument('manifest',type=Path);up.add_argument('--workers',type=int,default=4)
    sub.add_parser('inspect')
    sub.add_parser('validate')
    sub.add_parser('deploy')
    sub.add_parser('run')
    status=sub.add_parser('status');status.add_argument('run_id',type=int)
    sub.add_parser('evidence')
    sub.add_parser('events')
    args=p.parse_args()
    if args.command=='inspect':
        w=client();print(json.dumps({'active':w.current_user.me().active,'catalogs':[c.name for c in w.catalogs.list()]},indent=2));return
    if args.command=='upload':
        upload_manifest(client(),args.manifest,workers=args.workers,progress=lambda m:print(m,flush=True));return
    state=json.loads((ROOT/'.cache'/'databricks'/'deployment.json').read_text())
    if args.command=='validate':
        bundle_command(['validate'],state)
    elif args.command=='deploy':
        bundle_command(['deploy'],state)
        state.update(resource_ids(state));_atomic_json(ROOT/'.cache/databricks/deployment.json',state)
        print(json.dumps({k:state[k] for k in ('job_id','pipeline_id')}))
    elif args.command=='run':
        if 'job_id' not in state:
            state.update(resource_ids(state))
        run_id=client().jobs.run_now(state['job_id']).response.run_id
        state['last_run_id']=run_id;_atomic_json(ROOT/'.cache/databricks/deployment.json',state)
        print(json.dumps({'run_id':run_id,'batch_version':state['batch_version']}))
    elif args.command=='status':
        print(json.dumps(run_status(client(),args.run_id,state),indent=2))
    elif args.command=='evidence':
        w=client();root=ROOT/'outputs/phase2_cloud'/state['batch_version'];root.mkdir(parents=True,exist_ok=True)
        for name in ('validation.json','presentation_data.json'):
            remote=state['variables']['landing_root']+'/evidence/'+state['batch_version']+'/'+name
            response=w.files.download(remote)
            with (root/name).open('wb') as f:
                for chunk in iter(lambda:response.contents.read(4*1024*1024),b''):
                    f.write(chunk)
            response.contents.close()
        print(str(root))
    elif args.command=='events':
        if 'pipeline_id' not in state:
            state.update(resource_ids(state))
        events=[]
        for e in client().pipelines.list_pipeline_events(state['pipeline_id'],max_results=50):
            events.append(e.as_dict())
            if len(events)>=50:
                break
        out=ROOT/'outputs/phase2_cloud'/state['batch_version'];out.mkdir(parents=True,exist_ok=True)
        _atomic_json(out/'pipeline_events.json',events)
        print(json.dumps([{'timestamp':e.get('timestamp'),'level':e.get('level'),'event_type':e.get('event_type'),'message':e.get('message'),'error':e.get('error')} for e in events[:20]],indent=2))


if __name__=='__main__':
    main()
