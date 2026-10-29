"""Transport and cloud acceptance contracts; Spark-specific QA also runs in preflight."""
from collections import Counter
from decimal import Decimal
from pathlib import Path
from contextlib import contextmanager
import shutil
import uuid
import gzip
import json
import unittest

import simplejson
import yaml

from phase2.staging import _stage_one,sha256_file
from phase2.batches import through_day
from src.validation.audit import assess
from src.transformations.contracts import identifier


ROOT=Path(__file__).resolve().parents[1]


@contextmanager
def scratch_directory():
    parent=ROOT/'.cache/test_runs';parent.mkdir(parents=True,exist_ok=True)
    root=parent/uuid.uuid4().hex;root.mkdir()
    try:
        yield root
    finally:
        root.resolve().relative_to(parent.resolve())
        shutil.rmtree(root)



class TransportTests(unittest.TestCase):
    def stage(self,root,rows,entity='checks',max_bytes=800):
        source=root/'input.json';source.write_text(simplejson.dumps(rows,use_decimal=True),encoding='utf-8')
        return _stage_one((str(source),entity,{'sha256':sha256_file(source)},'snapshot',str(root),max_bytes,None,'rules'))

    def test_decimal_roundtrip_chunking_and_source_trace(self):
        with scratch_directory() as folder:
            root=Path(folder)
            source=[{'Id':n,'Amount':Decimal('0.1234567890123456789'),'ItemsSold':[],
                     'Discounts':[{'Amount':Decimal('0.01')}],'BusinessDate':'2026-05-04'} for n in range(4)]
            receipt=self.stage(root,source)
            wrappers=[]
            for entry in receipt['files']:
                with gzip.open(root/entry['relative_path'],'rt',encoding='utf-8') as f:
                    wrappers.extend(json.loads(line) for line in f)
            recovered=[simplejson.loads(w['payload'],use_decimal=True) for w in wrappers]
            self.assertEqual(recovered,source)
            self.assertGreater(len(receipt['files']),1)
            self.assertEqual([w['source_row'] for w in wrappers],[1,2,3,4])
            self.assertEqual(receipt['counts']['discounts'],4)
            self.assertEqual(len({w['record_uid'] for w in wrappers}),4)

    def test_resume_is_identical_and_corrupt_chunk_is_rebuilt(self):
        with scratch_directory() as folder:
            root=Path(folder);rows=[{'Id':1}]
            first=self.stage(root,rows);self.assertEqual(first,self.stage(root,rows))
            output=root/first['files'][0]['relative_path'];original=output.read_bytes();output.write_bytes(b'corrupt')
            self.assertEqual(first,self.stage(root,rows));self.assertEqual(output.read_bytes(),original)

    def test_employee_projection_retains_deleted_jobs_and_removes_personal_data(self):
        with scratch_directory() as folder:
            root=Path(folder)
            row={'Id':1,'Name':'Sensitive','DateOfBirth':'1900-01-01','Deleted':True,'JobID':[6,9],'HomeLocationId':0}
            result=self.stage(root,[row],'employees')
            with gzip.open(root/result['files'][0]['relative_path'],'rt',encoding='utf-8') as f:
                record=json.loads(json.loads(f.readline())['payload'])
            self.assertEqual(record,{'Id':1,'Deleted':True,'JobID':[6,9],'HomeLocationId':0})

    def test_cumulative_batch_conserves_children_and_does_not_include_late_sources(self):
        with scratch_directory() as folder:
            root=Path(folder);(root/'members').mkdir()
            for n,day in enumerate(('2026-08-19','2026-08-20')):
                member={'entity':'checks','counts':{'checks':2,'discounts':n+1,'date:'+day:2},
                        'files':[{'relative_path':f'checks/{n}.gz'}]}
                (root/f'members/{n}.json').write_text(json.dumps(member))
            original={'expected_source_count':2,'snapshot_id':'fixed','counts':{'checks':4},'files':[]}
            manifest=root/'manifest.json';manifest.write_text(json.dumps(original))
            batch=json.loads(through_day(manifest,'2026-08-19').read_text())
            self.assertEqual(batch['snapshot_id'],'fixed');self.assertEqual(batch['counts'],{'checks':2,'discounts':1})
            self.assertFalse(batch['complete_history']);self.assertEqual(batch['source_count'],1)


class AcceptanceTests(unittest.TestCase):
    def sample(self):
        metrics={'bronze_distinct_records':2,'candidate_accounts':2,'valid_accounts':2,'quarantine_accounts':0,
                 'distinct_valid_accounts':2,'check_discount_amount':'1.000000','child_discount_amount':'1.000000',
                 'sold_item_records':4,'discount_records':1,'payment_records':2,'void_records':0,'tax_records':2,
                 'eligible_discount':Decimal('1.000000')}
        manifest={'counts':{'checks':2,'sold_items':4,'discounts':1,'payments':2,'voids':0,'taxes':2},
                  'reference':{'eligible_discount':1}}
        return metrics,manifest

    def test_accepts_balanced_results(self):
        self.assertTrue(all(r['passed'] for r in assess(*self.sample())))

    def test_rejects_fanout_amount_loss_and_wrong_historical_population(self):
        for field,value in [('valid_accounts',3),('child_discount_amount','1.03'),('bronze_distinct_records',1),('eligible_discount',Decimal('0.9'))]:
            with self.subTest(field=field):
                metrics,manifest=self.sample();metrics[field]=value
                self.assertTrue(any(not r['passed'] for r in assess(metrics,manifest)))

    def test_identifier_guard(self):
        self.assertEqual(identifier('safe_table'),'safe_table')
        with self.assertRaises(ValueError):identifier('unsafe;drop')

    def test_job_has_gated_exports_paused_schedule_and_bounded_concurrency(self):
        doc=yaml.safe_load((ROOT/'resources/jobs/daily_orchestration_job.yml').read_text())
        job=doc['resources']['jobs']['applebees_job'];tasks={t['task_key']:t for t in job['tasks']}
        self.assertEqual(job['max_concurrent_runs'],1);self.assertEqual(job['schedule']['pause_status'],'PAUSED')
        self.assertEqual(tasks['presentation_outputs']['depends_on'],[{'task_key':'validate_results'}])
        self.assertFalse(tasks['update_pipeline']['pipeline_task']['full_refresh'])


if __name__=='__main__':unittest.main()
