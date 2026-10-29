"""Prepare immutable NDJSON.gz transport locally; no cloud credentials needed."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from applebees_eda import Config
from phase2.staging import prepare_staging


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checks',default=str(Path.home()/'Downloads'/'facts.zip')+'::facts/checks/*/checks.json')
    p.add_argument('--raw',type=Path,default=Path.home()/'Downloads')
    p.add_argument('--rules',type=Path,default=ROOT/'outputs'/'operacion_comercial'/'reason_dictionary.csv')
    p.add_argument('--output',type=Path,default=ROOT/'data'/'staging'/'phase2')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--max-sources',type=int)
    a=p.parse_args()
    config=Config(checks=(a.checks,),catalogs={k:str(a.raw/n) for k,n in {
        'revenue_centers':'revenueCenters.json','locations':'locations.json','items':'items.json','employees':'employees.json'}.items()})
    root,manifest=prepare_staging(config,a.output,a.rules,a.workers,max_sources=a.max_sources,progress=lambda m:print(m,flush=True))
    print(root,flush=True);print(manifest['counts'],flush=True)


if __name__=='__main__':
    main()
