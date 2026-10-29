"""Behavioral tests for operational attribution, dictionary edits and denominators."""
from pathlib import Path
import json
import shutil
import unittest
import uuid

import numpy as np
import pandas as pd

from test_eda import check,write_dataset
from applebees_eda import ingest,open_database
from applebees_eda.operations_data import build_operational_inputs,OperationsConfig
from applebees_eda.operations_analysis import product_comparisons,account_reports,activity_models
from applebees_eda.operations import run_operations,select_review,verify_review
from applebees_eda.operations_reporting import write_findings


def correction(c,name='SERVERERR'):
    c['Discounts'][0]['DiscountName']=name
    return c


class OperationsTests(unittest.TestCase):
    def setUp(self):
        self.scratch=Path(__file__).resolve().parents[1]/'.cache'/'test_runs'
        self.scratch.mkdir(parents=True,exist_ok=True)
        self.root=self.scratch/uuid.uuid4().hex;self.root.mkdir()
        self.connections=[]
        self.cfg=OperationsConfig(bootstrap_runs=10,minimum_product_checks=2,review_accounts=12)

    def tearDown(self):
        for c in self.connections:
            c.close()
        self.root.resolve().relative_to(self.scratch.resolve())
        shutil.rmtree(self.root)

    def load(self,rows,item_name=None):
        config=write_dataset(self.root,rows)
        if item_name is not None:
            path=Path(config.catalogs['items'])
            catalog=json.loads(path.read_text())
            catalog[0]['Name']=item_name
            path.write_text(json.dumps(catalog),encoding='utf-8')
        ingest(config)
        con=open_database(config);self.connections.append(con)
        result=build_operational_inputs(con,config,self.root/'operations',self.cfg,progress=lambda _:None)
        return con,config,result

    def test_check_corrections_do_not_become_direct_item_faults_or_multiply_amounts(self):
        c=correction(check())
        extra=dict(c['ItemsSold'][0],Sequence=2,ParentSequence=1)
        c['ItemsSold'].append(extra)
        c['Payments'][0]['Amount']=39
        c['Voids']=[{'ItemId':11,'Amount':5,'VoidCategoryId':1,'VoidCategoryName':'WRONG ITEM'}]
        con,config,_=self.load([c])
        row=con.execute('SELECT count(*),sum(discount),sum(exported_void_amount),bool_and(has_correction) FROM ops_accounts').fetchone()
        self.assertEqual(row,(1,2.,5.,True))
        self.assertEqual(con.execute('SELECT count(*),max(sold_lines),bool_or(directly_linked_correction),bool_or(has_linked_line) FROM ops_products').fetchone(),(1,2,False,True))
        reports=account_reports(con)
        self.assertEqual(reports['quality'].both_correction_forms.iloc[0],1)

    def test_line_discount_maps_only_to_its_item_and_voids_map_independently(self):
        c=check(discount=0)
        c['ItemsSold'][0]['Discounts']=[{'DiscountId':3,'DiscountName':'KITCHNERR','Amount':2,'EmployeeId':1,'ManagerId':99}]
        c['Payments'][0]['Amount']=19
        con,_,_=self.load([c])
        self.assertTrue(con.execute('SELECT directly_linked_correction FROM ops_products').fetchone()[0])
        self.assertEqual(con.execute('SELECT discount FROM ops_accounts').fetchone()[0],2.)

    def test_dictionary_pending_and_edits_create_new_version_preserve_notes(self):
        con,config,first=self.load([check()])
        self.assertEqual(first[2].operational_group.iloc[0],'pending')
        path=self.root/'operations'/'reason_dictionary.csv'
        edits=pd.read_csv(path,dtype=str,keep_default_na=False)
        edits['operational_group']='correction';edits['notes']='human note';edits.to_csv(path,index=False)
        second=build_operational_inputs(con,config,self.root/'operations',self.cfg,progress=lambda _:None)
        self.assertNotEqual(first[0],second[0])
        self.assertEqual(second[2].notes.iloc[0],'human note')
        self.assertTrue(con.execute('SELECT has_correction FROM ops_accounts').fetchone()[0])

    def test_common_products_have_no_supported_absence_comparison(self):
        rows=[correction(check(i+1,'2026-05-04')) if i<10 else check(i+1,'2026-05-04',discount=0) for i in range(40)]
        con,_,_=self.load(rows)
        products,_=product_comparisons(con,self.cfg)
        self.assertEqual(products.accounts_with_product.iloc[0],40)
        self.assertEqual(products.peer_coverage.iloc[0],0)
        self.assertFalse(products.sufficient_support.iloc[0])

    def test_full_small_phase_finishes_without_inventing_model_evidence(self):
        rows=[correction(check(i+1,'2026-05-'+str(4+i).zfill(2))) for i in range(7)]
        con,config,_=self.load(rows)
        run=run_operations(con,config,self.cfg,self.root/'operations',progress=lambda _:None)
        summary=json.loads((run/'complete.json').read_text())
        self.assertEqual(len(summary['hypotheses']),4)
        self.assertEqual(json.loads((run/'model_status.json').read_text())['status'],'skipped')
        review=pd.read_csv(run/'reviewed_accounts.csv')
        self.assertTrue(review.identity_verified.all())
        self.assertTrue(review.fraud_confirmed.isna().all())
        self.assertFalse(review.policy_verified.any())
        briefs,priorities=write_findings(run,con)
        self.assertEqual(len(priorities),3)
        self.assertTrue((run/'findings.md').exists())
        notes=self.root/'operations'/'analyst_notes.csv'
        notes.write_text('hypothesis_id,reviewer_notes,business_validation\nH1,Keep this note,pending\n',encoding='utf-8')
        again=run_operations(con,config,self.cfg,self.root/'operations',progress=lambda _:None)
        self.assertEqual(run,again)
        self.assertIn('Keep this note',notes.read_text())

    def test_same_code_with_different_names_is_not_silently_merged(self):
        con,_,result=self.load([correction(check(1)),correction(check(2),'MANAGER')])
        self.assertEqual(len(result[2]),2)
        self.assertEqual(set(result[2].operational_group),{'correction','generic_authorization'})
        self.assertEqual(len(account_reports(con)['reason_id_name_conflicts']),1)

    def test_popularity_alone_does_not_create_product_excess(self):
        rows=[]
        for i in range(100):
            c=check(i+1,discount=2 if i%5==0 else 0)
            if c['Discounts']:
                c['Discounts'][0]['DiscountName']='SERVERERR'
            c['ItemsSold'][0]['ItemId']=10 if i<80 else 11
            rows.append(c)
        con,_,_=self.load(rows)
        products,_=product_comparisons(con,self.cfg)
        np.testing.assert_allclose(products.excess_per_1000,0,atol=1e-10)

    def test_explanatory_models_fit_both_outcomes_with_sufficient_clusters(self):
        rows=[]
        for i in range(400):
            day='2026-05-'+str(4+i%20).zfill(2)
            c=check(i+1,day,seller=1+(i//20)%10,discount=2 if i%3 else 0,cash=(i%4==0))
            if c['Discounts']:
                c['Discounts'][0]['DiscountName']='SERVERERR' if (i//3)%2 else 'MANAGER'
            hour=12 if i%5 else 18
            c['OpenTime']=day+f'T{hour}:00';c['CloseTime']=day+f'T{hour+1}:00'
            rows.append(c)
        con,_,_=self.load(rows)
        estimates,status,_=activity_models(con,self.cfg)
        self.assertEqual(status['sample_n'],400)
        self.assertEqual({m['target'] for m in status['models'] if m['status']=='fit'},{'correction','generic'})
        self.assertTrue(set(estimates.scale).issubset({'odds_ratio','percentage_points'}))

    def test_nonempty_case_briefs_preserve_csv_identifiers_and_marker_warning(self):
        rows=[]
        for i in range(160):
            c=check(i+1,'2026-05-04',seller=1+i%10,discount=2 if i<80 and i%2==0 else 0)
            if c['Discounts']:
                c['Discounts'][0]['DiscountName']='SERVERERR'
            c['ItemsSold'][0]['ItemId']=10 if i<80 else 11
            rows.append(c)
        con,config,_=self.load(rows,item_name='DON T MAKE FOOD')
        run=run_operations(con,config,self.cfg,self.root/'operations',review_json=False,progress=lambda _:None)
        briefs,_=write_findings(run,con)
        self.assertGreater(len(briefs),0)
        self.assertIn('10',briefs.item_id.tolist())
        self.assertTrue(briefs.workflow_marker_candidate.any())
        self.assertTrue(briefs.loc[briefs.workflow_marker_candidate,'next_action'].str.contains('secuencia').all())
        self.assertTrue((run/'commercial_candidate_shortlist.csv').exists())


if __name__=='__main__':
    unittest.main()
