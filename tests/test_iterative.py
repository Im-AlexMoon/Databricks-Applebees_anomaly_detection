"""Synthetic checks for temporal isolation, attribution and weak-label behavior."""
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import pandas as pd

from test_eda import check, write_dataset
from applebees_eda import ingest, open_database
from applebees_eda.iterative_data import (ExperimentConfig, PeerReference, weekly_dataset, build_inputs,
                                         phase_for, SUM_COLUMNS, KEY)
from applebees_eda.weak_supervision import Teacher, FEATURES, FORBIDDEN, label_scores, run_teachers, run_students
from applebees_eda.iterative_statistics import descriptive_reports, adjust_families
from applebees_eda.iterative import run_experiment
from applebees_eda.iterative_reporting import compare_models


def synthetic_data(weeks=8, employees=18, seed=42):
    rng=np.random.default_rng(seed)
    rows=[]
    for week in pd.date_range('2026-05-04',periods=weeks,freq='7D'):
        for employee in range(employees):
            n=60
            d=int(rng.binomial(n,.12 if employee else .65))
            row=dict(seller_id=str(employee),location_id='1',week_start=week,revenue_center_id='1',time_band='11-15',weekday=1,
                     n=n,discounted=d,cash=int(rng.binomial(n,.15)),voids=int(rng.binomial(n,.02)),money_n=n,
                     gross=1200.,discount=d*8.,ratio_sum=d*.4,full_n=int(d*.05),unexplained_n=0,
                     alternative_discount=d*8.,alternative_n=n,alternative_gross=1200.,check_discount=d*8.,item_discount=0.,
                     time_valid_n=n,seller_matched_n=n)
            rows.append(row)
    contexts=pd.DataFrame(rows)
    weeks_frame=pd.DataFrame({'week_start':contexts.week_start.unique(),'source_days':7,'evaluable_n':employees*60,
                             'reconciled_n':employees*60,'quantity_n':0,'quantity_reconciled_n':0,
                             'discounted_n':100,'discounted_reconciled_n':100})
    pairs=contexts[KEY].assign(manager_id='99',manager_matched=True,checks=1,records=1,discount=8.)
    motives=contexts[KEY].assign(discount_id='3',checks=1,records=1,discount=8.)
    cfg=ExperimentConfig(min_peers=2,min_class_employees=2,bootstrap_runs=2,min_class_rows=2,bootstrap_ci_runs=10)
    weekly=weekly_dataset(contexts,weeks_frame,pairs,motives,cfg)
    return contexts,weekly,cfg


class IterativeTests(unittest.TestCase):
    def setUp(self):
        self.scratch=Path(__file__).resolve().parents[1]/'.cache'/'test_runs'
        self.scratch.mkdir(parents=True,exist_ok=True)
        self.root=self.scratch/uuid.uuid4().hex
        self.root.mkdir()
        self.connections=[]

    def tearDown(self):
        for con in self.connections:
            con.close()
        self.root.resolve().relative_to(self.scratch.resolve())
        shutil.rmtree(self.root)

    def test_account_money_roles_cache_and_unknown_managers(self):
        rows=[check(i+1,'2026-05-'+str(4+i).zfill(2),discount=2) for i in range(7)]
        config=write_dataset(self.root,rows)
        ingest(config)
        con=open_database(config); self.connections.append(con)
        output=self.root/'iterative'
        first=build_inputs(con,config,output)
        second=build_inputs(con,config,output)
        self.assertEqual(first,second)
        self.assertEqual(con.execute('SELECT count(*),sum(discount),min(seller_id),max(seller_id) FROM iterative_accounts').fetchone(),(7,14.,'1','1'))
        contexts=con.execute('SELECT * FROM iterative_contexts').fetchdf()
        weeks=con.execute('SELECT * FROM iterative_weeks').fetchdf()
        pairs=con.execute('SELECT * FROM iterative_pairs').fetchdf()
        motives=con.execute('SELECT * FROM iterative_motives').fetchdf()
        weekly=weekly_dataset(contexts,weeks,pairs,motives,ExperimentConfig())
        self.assertEqual(weekly.source_days.iloc[0],7)
        self.assertEqual(weekly.manager_coverage.iloc[0],0)
        self.assertFalse(weekly.quality_ok.iloc[0])
        self.assertEqual(weekly.manager_hhi.iloc[0],0)

    def test_leave_employee_out_and_low_exposure_shrinkage(self):
        c,w,cfg=synthetic_data()
        ref=PeerReference(cfg).fit(c)
        a=ref.transform(c,w)
        # An extreme employee does not inflate their own expected frequency.
        own=c.copy(); own.loc[own.seller_id=='0','discounted']=own.loc[own.seller_id=='0','n']
        b=PeerReference(cfg).fit(own).transform(c,w)
        np.testing.assert_allclose(a.loc[a.seller_id=='0','peer_rate'],b.loc[b.seller_id=='0','peer_rate'])
        tiny=c.iloc[:1].copy(); tiny['n']=1; tiny['discounted']=1; tiny['money_n']=1
        tw=w.iloc[:1].copy(); tw['n']=1; tw['money_n']=1; tw['discounted']=1; tw['enough_exposure']=False
        result=ref.transform(tiny,tw)
        self.assertFalse(result.model_eligible.iloc[0])
        self.assertLess(result.smoothed_rate.iloc[0],.3)

    def test_persistence_requires_calendar_and_location_and_no_low_support_alert(self):
        x=pd.DataFrame({'seller_id':['1']*4,'location_id':['1','1','1','2'],
                        'week_start':pd.to_datetime(['2026-05-04','2026-05-11','2026-05-25','2026-06-01']),
                        'excess_evidence':[True]*4,'model_eligible':[True,True,False,True],
                        'both_high':[True]*4,'both_low':[False]*4,'detector_disagreement':[False]*4})
        labeled=label_scores(x)
        self.assertEqual(labeled.pseudo_label.tolist(),[-1,1,-1,-1])
        self.assertTrue(labeled.fraud_confirmed.isna().all())

    def test_future_changes_do_not_change_oof_scores_labels_or_weights(self):
        c,w,cfg=synthetic_data(weeks=6)
        past=run_teachers(c,w,cfg,progress=lambda _:None)['scores']
        self.assertFalse(past.empty)
        future=c.loc[c.week_start==c.week_start.max()].copy()
        future['week_start']=pd.Timestamp('2026-08-03'); future['discounted']=future.n; future['discount']=future.gross
        fw=w.loc[w.week_start==w.week_start.max()].copy(); fw['week_start']=pd.Timestamp('2026-08-03'); fw['phase']='test'
        combined=run_teachers(pd.concat([c,future],ignore_index=True),pd.concat([w,fw],ignore_index=True),cfg,progress=lambda _:None)['scores']
        got=combined.loc[combined.score_period=='oof'].reset_index(drop=True)
        pd.testing.assert_frame_equal(past.reset_index(drop=True),got)
        self.assertTrue((got.reference_end<got.week_start).all())

    def test_individual_signal_and_shared_promotion_have_distinct_peer_excess(self):
        c,w,cfg=synthetic_data()
        result=PeerReference(cfg).fit(c).transform(c,w)
        self.assertGreater(result.loc[result.seller_id=='0','excess_rate'].median(),.25)
        # A promotion present across peers is learned as context, not a unique employee excess.
        promotion=c.copy(); promotion['discounted']=36
        wp=w.copy(); wp['discounted']=36; wp['discount_rate']=.6
        promo=PeerReference(cfg).fit(promotion).transform(promotion,wp)
        self.assertLess(promo.excess_rate.abs().max(),1e-10)

    def test_one_component_is_valid_and_students_skip_without_two_classes(self):
        c,w,cfg=synthetic_data()
        features=PeerReference(cfg).fit(c).transform(c,w)
        model=Teacher(cfg).fit(features,architecture=(1,'diag'))
        scored=model.score(features)
        self.assertTrue((scored.p_cluster==1).all())
        scored['score_period']='oof'; scored['pseudo_label']=0; scored['pseudo_weight']=1.
        self.assertEqual(run_students(scored,cfg)['status']['status'],'skipped')
        self.assertFalse(set(FEATURES)&FORBIDDEN)

    def test_both_weighted_students_train_and_report_only_imitation_metrics(self):
        c,w,cfg=synthetic_data()
        features=PeerReference(cfg).fit(c).transform(c,w)
        scored=Teacher(cfg).fit(features,architecture=(1,'diag')).score(features)
        scored['score_period']=np.where(scored.week_start<pd.Timestamp('2026-06-15'),'oof','validation')
        scored['pseudo_label']=scored.seller_id.isin(['0','1']).astype(int)
        scored['pseudo_weight']=.8
        result=run_students(scored,cfg)
        self.assertEqual(result['status']['status'],'fit')
        self.assertEqual(set(result['metrics'].model),{'GaussianNB','LogisticRegression'})
        self.assertTrue(result['metrics'].interpretation.str.contains('not_fraud_accuracy').all())

    def test_small_experiment_exports_quality_and_skips_without_forcing_models(self):
        config=write_dataset(self.root,[check(i+1,'2026-05-'+str(4+i).zfill(2)) for i in range(7)])
        ingest(config)
        con=open_database(config); self.connections.append(con)
        cfg=ExperimentConfig(bootstrap_runs=2,bootstrap_ci_runs=10)
        out=run_experiment(con,config,cfg,self.root/'experiment',progress=lambda _:None,review_json=False)
        self.assertTrue((out/'complete.json').exists())
        self.assertTrue((out/'motive_tests.parquet').exists())
        self.assertEqual(json.loads((out/'student_status.json').read_text())['status'],'skipped')
        alternative=run_experiment(con,config,cfg,self.root/'experiment',progress=lambda _:None,review_json=False,
                                   monetary_mode='price_difference_plus_check')
        self.assertNotEqual(out,alternative)
        self.assertEqual(pd.read_parquet(alternative/'weekly_dataset.parquet').discount.sum(),14.)
        legacy=out.parent/'weak-eda-v1-legacy'
        legacy.mkdir()
        (legacy/'complete.json').write_text('{}')
        (legacy/'manifest.json').write_text('{}')
        teachers,conventions=compare_models(out)
        self.assertTrue(teachers.empty)
        self.assertTrue((out/'comparison_manifest.json').exists())

    def test_boundaries_and_multiple_comparisons(self):
        cfg=ExperimentConfig()
        weeks=pd.Series(pd.to_datetime(['2026-06-22','2026-06-29','2026-07-06','2026-07-27','2026-08-17']))
        self.assertEqual(phase_for(weeks,cfg).tolist(),['discovery','descriptive','validation','descriptive','descriptive'])
        result=adjust_families(pd.DataFrame({'family':['a','a','b'],'p_value':[.01,.02,.01]}))
        self.assertTrue((result.q_by>=result.p_value).all())
        with self.assertRaises(ValueError):
            ExperimentConfig(discovery_start='2026-05-05')


if __name__=='__main__':
    unittest.main()
