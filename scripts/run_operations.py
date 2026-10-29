"""Execute the operational phase from existing Parquet; keep JSON and results local."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from applebees_eda import Config,open_database
from applebees_eda.operations import run_operations
from applebees_eda.operations_data import OperationsConfig
from applebees_eda.operations_reporting import write_findings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checks',default=str(Path.home()/'Downloads'/'facts.zip')+'::facts/checks/*/checks.json')
    p.add_argument('--raw',type=Path,default=Path.home()/'Downloads')
    p.add_argument('--snapshot',type=Path,default=ROOT/'outputs'/'eda')
    p.add_argument('--output',type=Path,default=ROOT/'outputs'/'operacion_comercial')
    p.add_argument('--sample-accounts',type=int,default=40000)
    p.add_argument('--bootstrap-runs',type=int,default=500)
    p.add_argument('--memory-limit',default='4GB',help='Memoria de trabajo DuckDB; usa 2GB en equipos con poca RAM')
    p.add_argument('--skip-json-review',action='store_true')
    args=p.parse_args()
    config=Config(checks=(args.checks,),output_dir=args.snapshot,memory_limit=args.memory_limit,catalogs={key:str(args.raw/name) for key,name in {
        'revenue_centers':'revenueCenters.json','locations':'locations.json','items':'items.json','employees':'employees.json'}.items()})
    cfg=OperationsConfig(sample_accounts=args.sample_accounts,bootstrap_runs=args.bootstrap_runs)
    con=open_database(config)
    try:
        run=run_operations(con,config,cfg,args.output,review_json=not args.skip_json_review,
                           progress=lambda m:print(m,flush=True))
        write_findings(run,con)
        print(run,flush=True)
    finally:
        con.close()


if __name__=='__main__':
    main()
