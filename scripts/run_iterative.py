"""Run the same experiment as the notebook, using an existing Parquet snapshot."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from applebees_eda import Config, open_database
from applebees_eda.iterative_data import ExperimentConfig
from applebees_eda.iterative import run_experiment
from applebees_eda.iterative_reporting import compare_models


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks',default=str(Path.home()/'Downloads'/'facts.zip')+'::facts/checks/*/checks.json')
    parser.add_argument('--raw',type=Path,default=Path.home()/'Downloads')
    parser.add_argument('--snapshot',type=Path,default=ROOT/'outputs'/'eda')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/'eda_iterativo')
    parser.add_argument('--bootstrap-runs',type=int,default=10)
    parser.add_argument('--monetary-mode',choices=['records_sum','price_difference_plus_check'],default='records_sum')
    parser.add_argument('--skip-json-review',action='store_true',help='Solo desarrollo; la entrega debe contrastar las fuentes')
    args=parser.parse_args()
    config=Config(checks=(args.checks,),catalogs={key:str(args.raw/name) for key,name in {
        'revenue_centers':'revenueCenters.json','locations':'locations.json','items':'items.json','employees':'employees.json'}.items()},
        output_dir=args.snapshot)
    con=open_database(config)
    try:
        output=run_experiment(con,config,ExperimentConfig(bootstrap_runs=args.bootstrap_runs),args.output,
                              progress=lambda message:print(message,flush=True),review_json=not args.skip_json_review,monetary_mode=args.monetary_mode)
        compare_models(output)
        print(output,flush=True)
    finally:
        con.close()


if __name__=='__main__':
    main()
