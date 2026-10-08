"""Behavioral tests for nested accounting, joins, DST and temporal leakage."""
import json
from pathlib import Path
import shutil
import uuid
import unittest

import numpy as np
import pandas as pd

from applebees_eda.pipeline import Config, ingest, open_database, prepare_facts, quality_report, monetary_gate, local_time_info
from applebees_eda.analysis import (seller_contexts, model_features, explore_anomalies, review_checks,
    baseline, discount_types, actor_roles, employee_manager_pairs, weekly_trends, associations)


def check(check_id=1, day="2026-04-28", seller=1, discount=2, cash=False):
    return {"Id": check_id, "LocationId": 1, "BusinessDate": day, "RevenueCenterId": 1,
        "OpenTime": day+"T12:00", "CloseTime": day+"T13:00", "ServiceCharge": 0,
        "TrafficCount": 2, "Tax": [{"TaxCategoryId": 1, "Amount": 1}],
        "ItemsSold": [{"ItemId": 10, "Quantity": 1, "GrossPrice": 20, "SoldPrice": 20,
            "EmployeeId": seller, "RevenueCenterId": 1, "ItemPLU": 100, "ItemSubCatId": 224,
            "Sequence": 1, "ParentSequence": 0, "Discounts": []}],
        "Discounts": [] if not discount else [{"DiscountId": 3, "DiscountName": "TEST", "Amount": discount,
            "EmployeeId": seller, "ManagerId": 99}],
        "Payments": [{"PaymentCategoryId": 1, "PayCatName": "CASH" if cash else "VISA",
            "Amount": 21-discount, "Gratuity": 0, "EmployeeId": 99, "IsCreditCard": not cash}], "Voids": []}


def write_dataset(root, checks, employees=None, duplicate_location=False):
    employees = employees or [{"Id": 1, "Name": "Never exported", "DateOfBirth": "1900-01-01",
        "LocationId": 1, "LocationEmployeeId": "001", "PayrollEmployeeId": "001", "Deleted": True,
        "Borrowed": False, "HomeLocationId": 0, "PrimaryJobID": 6, "JobID": [6,9,16,21]}]
    catalogs = {
        "locations": [{"Id": 1, "Name": "TEST", "OpenDate": "1990-01-01", "ClosedDate": None,
            "TimeZoneName": "America/Chicago"}],
        "revenue_centers": [{"Id": 1, "Name": "In-Store"}],
        "items": [{"Id": 10, "Name": "TEST ITEM", "Plu": 100, "SalesCategoryId": 224}],
        "employees": employees,
    }
    if duplicate_location:
        catalogs["locations"].append(catalogs["locations"][0].copy())
    paths = {}
    for name, rows in catalogs.items():
        path = root/(name+".json")
        path.write_text(json.dumps(rows), encoding="utf-8")
        paths[name] = str(path)
    path = root/"checks.json"
    path.write_text(json.dumps(checks), encoding="utf-8")
    return Config(checks=(str(path),), catalogs=paths, output_dir=root/"out", batch_checks=500,
        price_mode="line", discount_mode="records_sum", gratuity_mode="ignore")


class EDATests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1]/".cache"/"test_runs"
        scratch.mkdir(parents=True,exist_ok=True)
        self.scratch = scratch.resolve()
        self.root = self.scratch/uuid.uuid4().hex
        self.root.mkdir()
        self.connections = []

    def tearDown(self):
        for con in self.connections:
            con.close()
        self.root.resolve().relative_to(self.scratch)
        shutil.rmtree(self.root)

    def load(self, rows, **kwargs):
        config = write_dataset(self.root, rows, **kwargs)
        ingest(config)
        con = open_database(config)
        self.connections.append(con)
        prepare_facts(con, config)
        return config, con

    def test_nested_rows_do_not_multiply_money_or_employee_jobs(self):
        c = check()
        c["Discounts"][0]["Amount"] = 1
        c["Discounts"].append(dict(c["Discounts"][0], DiscountId=4))
        c["Payments"][0]["Amount"] = 10
        c["Payments"].append(dict(c["Payments"][0], Amount=9))
        c["Voids"] = [{"ItemId": 10, "Amount": 2}, {"ItemId": 10, "Amount": 3}]
        config, con = self.load([c])
        self.assertEqual(con.execute("SELECT count(*),sum(recorded_discount),sum(payment_amount),sum(void_amount) FROM facts").fetchone(), (1,2.,19.,5.))
        self.assertEqual(con.execute("SELECT count(*) FROM employee_jobs").fetchone()[0],4)
        self.assertTrue(con.execute("SELECT seller_deleted FROM facts").fetchone()[0])
        self.assertEqual(con.execute("SELECT seller_id FROM facts").fetchone()[0],"1")  # Payment employee is 99.
        self.assertNotIn("name", [r[0] for r in con.execute("DESCRIBE employees").fetchall()])
        self.assertNotIn("date_of_birth", [r[0] for r in con.execute("DESCRIBE employees").fetchall()])
        self.assertTrue(monetary_gate(con, config)[0])

    def test_duplicate_keys_excluded_and_catalog_ambiguity_no_fanout(self):
        config, con = self.load([check(),check(),check(2)], duplicate_location=True)
        self.assertEqual(con.execute("SELECT count(*) FROM facts").fetchone()[0],1)
        self.assertEqual(con.execute("SELECT count(*) FROM check_base WHERE selection_status='duplicate_key'").fetchone()[0],2)
        self.assertFalse(con.execute("SELECT location_matched FROM facts").fetchone()[0])
        self.assertEqual(quality_report(con,config)["catalogs"].query("catalog=='locations'").ambiguous_rows.iloc[0],2)

    def test_quantity_convention_and_zero_gross(self):
        c = check(discount=0)
        c["ItemsSold"][0]["Quantity"] = 2
        # Prices are LINE totals in this fixture; multiplying would invent sales.
        config, con = self.load([c])
        qa = quality_report(con,config)["reconciliation"]
        self.assertEqual(qa.query("price_mode=='line' and discount_mode=='records_sum' and gratuity_mode=='ignore'").match_fraction.iloc[0],1)
        self.assertEqual(qa.query("price_mode=='unit' and discount_mode=='records_sum' and gratuity_mode=='ignore'").match_fraction.iloc[0],0)
        config.price_mode = "unit"
        prepare_facts(con,config)
        self.assertFalse(monetary_gate(con,config)[0])

    def test_unresolved_money_skips_model_but_preserves_incidence(self):
        config, con = self.load([check()])
        config.price_mode = "unresolved"
        prepare_facts(con,config)
        self.assertIsNone(con.execute("SELECT discount_ratio FROM facts").fetchone()[0])
        self.assertEqual(explore_anomalies(con,config)["status"],"skipped")
        self.assertEqual(len(seller_contexts(con,config)),1)
        self.assertEqual(len(review_checks(con,config)),1)

    def test_multiseller_and_negative_records_are_not_money_eligible(self):
        c = check()
        c["ItemsSold"].append(dict(c["ItemsSold"][0], EmployeeId=2, GrossPrice=-1, SoldPrice=-1))
        config, con = self.load([c])
        self.assertIsNone(con.execute("SELECT seller_id FROM facts").fetchone()[0])
        self.assertFalse(con.execute("SELECT money_eligible FROM facts").fetchone()[0])
        self.assertTrue(seller_contexts(con,config).empty)

    def test_missing_money_and_zero_discount_not_silently_normalized(self):
        c = check()
        c["Discounts"][0]["Amount"] = 0
        c["ItemsSold"][0]["Quantity"] = None
        config, con = self.load([c])
        self.assertEqual(con.execute("SELECT has_discount,money_eligible,missing_item_money FROM facts").fetchone(),(False,False,1))

    def test_missing_catalogs_and_invalid_keys_reported(self):
        rows = [check(),dict(check(2),Id=None),check(3,day="2026-09-01")]
        config = write_dataset(self.root,rows)
        config.catalogs = {}
        ingest(config)
        con = open_database(config)
        self.connections.append(con)
        prepare_facts(con,config)
        self.assertEqual(con.execute("SELECT count(*) FROM facts").fetchone()[0],1)
        self.assertFalse(con.execute("SELECT seller_matched FROM facts").fetchone()[0])
        self.assertEqual(con.execute("SELECT open_time_status FROM facts").fetchone()[0],"missing_or_invalid_zone")
        self.assertEqual(len(quality_report(con,config)["selection"]),3)

    def test_dst_and_aware_time(self):
        self.assertEqual(local_time_info("2026-11-01T01:30","America/Chicago")[2],"ambiguous_local_time")
        self.assertEqual(local_time_info("2026-03-08T02:30","America/Chicago")[2],"nonexistent_local_time")
        local, utc, status = local_time_info("2026-04-28T17:00Z","America/Chicago")
        self.assertEqual(local,"2026-04-28T12:00:00")
        self.assertEqual(utc,"2026-04-28T17:00:00+00:00")
        self.assertEqual(status,"valid")

    def test_failed_ingest_leaves_no_partial_snapshot(self):
        c = check()
        c["Payments"] = {"Amount": 1}
        config = write_dataset(self.root,[c])
        with self.assertRaises(ValueError):
            ingest(config)
        self.assertFalse((config.output_dir/"parquet").exists())
        self.assertFalse(list(config.output_dir.glob(".ingest-*")))

    def test_jsonl_and_bom_and_no_overwrite(self):
        config = write_dataset(self.root,[check()])
        p = self.root/"checks.jsonl"
        p.write_text(json.dumps(check())+"\n"+json.dumps(check(2)),encoding="utf-8-sig")
        config.checks = (str(p),)
        manifest = ingest(config)
        self.assertEqual(manifest["counts"]["checks"],2)
        with self.assertRaises(FileExistsError):
            ingest(config)

    def test_all_eda_queries_and_two_discount_levels(self):
        c = check(discount=2)
        c["ItemsSold"][0]["SoldPrice"] = 17
        c["ItemsSold"][0]["Discounts"] = [{"DiscountId": 4,"DiscountName": "ITEM TEST",
            "Amount": 3,"EmployeeId": 1,"ManagerId": 99}]
        c["Payments"][0]["Amount"] = 16
        config,con = self.load([c])
        self.assertEqual(con.execute("SELECT recorded_discount,discount_amount,money_eligible FROM facts").fetchone(),(5.,5.,True))
        self.assertEqual(len(discount_types(con)),2)
        self.assertEqual(len(actor_roles(con)),2)
        self.assertEqual(employee_manager_pairs(con).pair_checks.iloc[0],1)
        self.assertEqual(len(baseline(con)),1)
        self.assertEqual(len(weekly_trends(con)),1)
        self.assertEqual(len(associations(con)["cash_void"]),1)

    def test_zero_gross_and_missing_arrays_are_reported(self):
        c = check(discount=0)
        c["ItemsSold"][0].update(GrossPrice=0,SoldPrice=0)
        c["Payments"][0]["Amount"] = 1
        missing = check(2)
        del missing["Discounts"]
        config,con = self.load([c,missing])
        self.assertIsNone(con.execute("SELECT discount_ratio FROM facts").fetchone()[0])
        self.assertFalse(con.execute("SELECT money_eligible FROM facts").fetchone()[0])
        self.assertEqual(con.execute("SELECT count(*) FROM check_base WHERE selection_status='incomplete_arrays'").fetchone()[0],1)

    def test_reconciliation_subset_blocks_misleading_overall_score(self):
        rows = [check(i,discount=0) for i in range(100)]
        bad = check(100,discount=0)
        bad["ItemsSold"][0]["Quantity"] = 2
        rows.append(bad)
        config,con = self.load(rows)
        config.price_mode = "unit"
        prepare_facts(con,config)
        self.assertFalse(monetary_gate(con,config)[0])
        self.assertEqual(con.execute("SELECT count(*) FROM facts WHERE money_eligible").fetchone()[0],0)
        self.assertTrue(pd.isna(baseline(con).discount_to_gross.iloc[0]))

    def test_snapshot_detects_changed_sources(self):
        config,con = self.load([check()])
        path = Path(config.checks[0])
        path.write_text(path.read_text(encoding="utf-8")+" ",encoding="utf-8")
        with self.assertRaises(ValueError):
            open_database(config)

    def test_temporal_features_and_model_do_not_learn_from_august(self):
        rows = []
        rng = np.random.default_rng(42)
        days = list(pd.date_range("2026-04-06","2026-07-27",freq="7D"))+[pd.Timestamp("2026-08-03")]
        for week, day in enumerate(days):
            for seller in range(1,7):
                for n in range(35):
                    amount = 2 if rng.random() < .1+seller*.025 else 0
                    rows.append(check(len(rows)+1,day.strftime("%Y-%m-%d"),seller,amount,bool(rng.random()<.2+week*.005)))
        employees = [{"Id": i,"LocationId": 1,"JobID": [6]} for i in range(1,7)]
        config, con = self.load(rows,employees=employees)
        first, _ = model_features(con,config)
        first_model = explore_anomalies(con,config)
        self.assertEqual(first_model["status"],"completed")
        self.assertTrue((first_model["reference_features"].period_end<pd.Timestamp("2026-08-01")).all())
        self.assertTrue((first_model["scored_august"].period_start>=pd.Timestamp("2026-08-01")).all())
        con.execute("UPDATE facts SET has_discount=true,discount_amount=10 WHERE business_day>=DATE '2026-08-01'")
        second, _ = model_features(con,config)
        cols = ["location_id","seller_id","week_start"]
        pd.testing.assert_frame_equal(first.query("phase=='reference'").sort_values(cols).reset_index(drop=True),
            second.query("phase=='reference'").sort_values(cols).reset_index(drop=True))
        # Changing August observations changes scored features, not expectations.
        self.assertFalse(first.query("phase=='august'").discount_rate_excess.equals(second.query("phase=='august'").discount_rate_excess))


if __name__ == "__main__":
    unittest.main()
