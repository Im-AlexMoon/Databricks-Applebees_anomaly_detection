# Databricks notebook source
"""Verify immutable source files and exercise shared Spark rules before running Lakeflow."""
import json,hashlib,sys
from pathlib import Path,PurePosixPath
for name in ('catalog','schema','manifest_path','source_root'):
    dbutils.widgets.text(name,'')
sys.path.insert(0,dbutils.widgets.get('source_root'))
from src.validation.synthetic import run_synthetic_checks

manifest_path=Path(dbutils.widgets.get('manifest_path'))
manifest=json.loads(manifest_path.read_text())
landing=Path(manifest['landing_root'])
if not manifest.get('files'):
    raise ValueError('Manifest has no source files')
for entry in manifest['files']:
    relative=PurePosixPath(entry['relative_path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe manifest path')
    path=landing.joinpath(*relative.parts)
    if not path.is_file() or path.stat().st_size!=entry['bytes']:
        raise AssertionError('Missing or incomplete upload: '+entry['relative_path'])
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            digest.update(block)
    if digest.hexdigest()!=entry['sha256']:
        raise AssertionError('Source hash mismatch: '+entry['relative_path'])
qa=run_synthetic_checks(spark)
print(json.dumps({'status':'passed','verified_files':len(manifest['files']),'snapshot_id':manifest['snapshot_id'],'synthetic_qa':qa}))
dbutils.jobs.taskValues.set(key='snapshot_id',value=manifest['snapshot_id'])
