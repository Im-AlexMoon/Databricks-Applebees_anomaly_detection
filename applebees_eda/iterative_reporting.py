"""Post-fit comparisons; these reports never feed back into model training."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from .iterative_data import KEY


def _top_overlap(a,b,left_score,right_score):
    left=set(map(tuple,a.nlargest(50,left_score)[KEY].to_numpy()))
    right=set(map(tuple,b.nlargest(50,right_score)[KEY].to_numpy()))
    return len(left&right)/max(1,min(50,len(left),len(right)))


def compare_models(run):
    run=Path(run)
    scores=pd.read_parquet(run/'weekly_scores.parquet')
    rows=[]
    if not scores.empty:
        for (period,week),all_rows in scores.groupby(['score_period','week_start']):
            group=all_rows.loc[all_rows.model_eligible]
            if group.empty:
                continue
            rows.append({'period':period,'week_start':week,'weekly_rows':len(all_rows),'eligible_rows':len(group),
                'eligible_coverage':len(group)/len(all_rows),'weak_label_coverage':float((group.pseudo_label>=0).mean()),
                'elevated_rows':int((group.pseudo_label==1).sum()),'detector_disagreement':float(group.detector_disagreement.mean()),
                'gmm_if_spearman':float(spearmanr(group.gmm_anomaly,group.if_anomaly).statistic) if len(group)>2 else np.nan,
                'gmm_if_top50_overlap':_top_overlap(group,group,'gmm_anomaly','if_anomaly'),
                'interpretation':'anomaly_stability_not_fraud_precision'})
    teachers=pd.DataFrame(rows)
    teachers.to_csv(run/'teacher_comparison.csv',index=False)
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    convention_rows=[]
    match=None
    for other in sorted(run.parent.glob('weak-eda-v1-*')):
        if other==run or not (other/'complete.json').exists():
            continue
        m=json.loads((other/'manifest.json').read_text(encoding='utf-8'))
        if (m.get('monetary_mode') in ('records_sum','price_difference_plus_check') and
            m['monetary_mode']!=manifest['monetary_mode'] and
            all(m.get(key)==manifest[key] for key in ('code_sha256','config','input_fingerprint','review_json'))):
            match=other
            break
    if match is not None and not scores.empty:
        other=pd.read_parquet(match/'weekly_scores.parquet')
        if other.empty:
            other=scores.iloc[:0].copy()
        joined=scores.merge(other,on=KEY,suffixes=('_a','_b'),validate='one_to_one')
        for (period,week),group in joined.groupby(['score_period_a','week_start']):
            a=scores.loc[(scores.week_start==week)&scores.model_eligible]
            b=other.loc[(other.week_start==week)&other.model_eligible]
            valid=group.model_eligible_a&group.model_eligible_b
            convention_rows.append({'period':period,'week_start':week,'common_rows':len(group),
                'label_agreement':float((group.pseudo_label_a==group.pseudo_label_b).mean()),
                'cluster_ari':adjusted_rand_score(group.loc[valid,'cluster_id_a'],group.loc[valid,'cluster_id_b']) if valid.sum()>1 else np.nan,
                'teacher_top50_overlap':_top_overlap(a,b,'rank_score','rank_score'),
                'discount_difference':float(group.discount_b.sum()-group.discount_a.sum()),
                'convention_a':manifest['monetary_mode'],'convention_b':json.loads((match/'manifest.json').read_text())['monetary_mode']})
    conventions=pd.DataFrame(convention_rows)
    conventions.to_csv(run/'convention_model_comparison.csv',index=False)
    (run/'comparison_manifest.json').write_text(json.dumps({'report_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'run':str(run),'matching_alternative_run':str(match) if match else None,
        'interpretation':'stability and imitation only; no fraud accuracy'},indent=2),encoding='utf-8')
    return teachers,conventions
