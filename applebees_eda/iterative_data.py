"""Versioned, account-conserving inputs for the iterative experiment.

Monetary eligibility is row-local here. A future period must never turn off
eligibility in an earlier period through the baseline's global monetary gate.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .pipeline import prepare_facts

VERSION = "weak-eda-v1"
KEY = ["seller_id", "location_id", "week_start"]
CONTEXT = ["location_id", "revenue_center_id", "time_band", "weekday"]


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    min_checks: int = 30
    min_peers: int = 3
    prior_checks: float = 30.0
    min_coverage: float = .8
    tail: float = .95
    reference_quantile: float = .5
    bootstrap_runs: int = 10
    bootstrap_ci_runs: int = 300
    min_training_rows: int = 50
    min_class_rows: int = 20
    min_class_employees: int = 5
    statistical_sample: int = 40000
    discovery_start: str = "2026-05-04"
    discovery_end: str = "2026-06-28"
    validation_start: str = "2026-07-06"
    validation_end: str = "2026-07-26"
    test_start: str = "2026-08-03"
    test_end: str = "2026-08-16"

    def __post_init__(self):
        if self.min_checks < 1 or self.prior_checks <= 0 or self.min_peers < 1:
            raise ValueError("Exposición y soporte de pares deben ser positivos")
        if not 0 < self.reference_quantile < self.tail < 1 or not 0 < self.min_coverage <= 1:
            raise ValueError("Umbrales inválidos")
        if self.bootstrap_runs < 2 or self.bootstrap_ci_runs < 2:
            raise ValueError("Se requieren al menos dos remuestreos")
        windows = []
        for phase in ("discovery", "validation", "test"):
            start, end = pd.Timestamp(getattr(self, phase+"_start")), pd.Timestamp(getattr(self, phase+"_end"))
            if start.weekday() != 0 or end.weekday() != 6 or end < start:
                raise ValueError("Los períodos deben contener semanas lunes–domingo completas")
            windows.append((start, end))
        if any(a[1] >= b[0] for a, b in zip(windows, windows[1:])):
            raise ValueError("Los períodos deben ser sucesivos y no solaparse")


def phase_for(weeks, cfg):
    weeks = pd.to_datetime(weeks)
    result = pd.Series("descriptive", index=weeks.index)
    for phase in ("discovery", "validation", "test"):
        result.loc[(weeks >= getattr(cfg, phase+"_start")) &
                   (weeks + pd.Timedelta(days=6) <= getattr(cfg, phase+"_end"))] = phase
    return result


def _quote(path):
    return str(Path(path).resolve()).replace("'", "''")


def _drop_input(con, name):
    kind = con.execute("SELECT table_type FROM information_schema.tables WHERE table_name=?", ["iterative_"+name]).fetchone()
    if kind:
        con.execute(f"DROP {'VIEW' if kind[0]=='VIEW' else 'TABLE'} iterative_{name}")


def build_inputs(con, config, output):
    """Build once, reuse the account/context Parquet on subsequent iterations."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = config.output_dir/"parquet"/"manifest.json"
    # Cache tracks the actual builder source, monetary tolerance, and input snapshot.
    digest = hashlib.sha256(source_manifest.read_bytes() + Path(__file__).read_bytes() +
                            str(config.money_tolerance).encode()).hexdigest()
    marker = output/"inputs.json"
    files = [output/(name+".parquet") for name in ("accounts", "contexts", "weeks", "pairs", "motives")]
    if marker.exists() and json.loads(marker.read_text())["fingerprint"] == digest and all(p.exists() for p in files):
        attach_inputs(con, output)
        return json.loads(marker.read_text())
    if marker.exists():
        raise ValueError("El snapshot iterativo cambió; usa otra carpeta de entradas")
    for name in ("accounts", "contexts", "weeks", "pairs", "motives"):
        _drop_input(con,name)

    prepare_facts(con, replace(config, price_mode="unresolved", discount_mode="unresolved", gratuity_mode="unresolved"))
    tol = float(config.money_tolerance)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE iterative_accounts AS
      WITH money AS (SELECT *, gross_line gross,
        recorded_discount discount,
        gross_line-sold_line+check_discount alternative_discount,
        payment_amount-(gross_line-recorded_discount+tax_amount+service_charge-gratuity_amount) residual,
        payment_amount-(sold_line-check_discount+tax_amount+service_charge-gratuity_amount) alternative_residual,
        (item_count>0 AND gross_line>0 AND payment_records>0 AND payment_amount>=0
         AND close_time IS NOT NULL AND missing_item_money=0 AND missing_discount_amounts=0
         AND missing_payment_money=0 AND missing_tax_amounts=0 AND service_charge IS NOT NULL
         AND negative_items=0 AND negative_discounts=0 AND negative_payments=0) evaluable
        FROM facts)
      SELECT uid,source_file,source_row,check_id,seller_id,location_id,business_day,week_start,
        revenue_center_id,time_band,weekday,item_count,traffic_count,seller_matched,
        has_discount, has_cash, (void_records>0) has_void, gross,discount,alternative_discount,
        check_discount,item_discount,residual,alternative_residual,evaluable,quantities_not_one,
        (evaluable AND discount>=0 AND gross-discount>=-{tol} AND abs(residual)<={tol}) eligible,
        (evaluable AND alternative_discount>=0 AND gross-alternative_discount>=-{tol}
          AND abs(alternative_residual)<={tol}) alternative_eligible,
        (abs(gross-discount)<={tol} AND gross>0) full_discount,
        (gross_line-sold_line-item_discount) unexplained_reduction,
        open_time_status,close_time_status
      FROM money""")
    before = con.execute("SELECT count(*),sum(recorded_discount) FROM facts").fetchone()
    after = con.execute("SELECT count(*),sum(discount) FROM iterative_accounts").fetchone()
    if before[0] != after[0] or not np.isclose(before[1], after[1], atol=tol, rtol=0):
        raise AssertionError("La construcción semanal alteró cuentas o importes")

    # Sales composition is per account; all sold lines are reduced before joining.
    con.execute("""CREATE OR REPLACE TEMP TABLE composition AS
      WITH categories AS (SELECT uid,coalesce(sub_category_id,'unknown') category,
                         sum(greatest(gross_price,0)) amount FROM sold_items GROUP BY 1,2)
      SELECT uid,arg_max(category,amount) sales_category FROM categories GROUP BY uid""")
    con.execute("ALTER TABLE iterative_accounts ADD COLUMN sales_category VARCHAR")
    con.execute("UPDATE iterative_accounts a SET sales_category=c.sales_category FROM composition c WHERE a.uid=c.uid")
    con.execute("""CREATE OR REPLACE TEMP TABLE iterative_weeks AS
      SELECT week_start,count(DISTINCT business_day) source_days,
        count(*) accounts,count(*) FILTER(WHERE seller_id IS NULL) unattributed_accounts,
        count(*) FILTER(WHERE eligible) eligible_accounts,
        sum(discount) recorded_discount,sum(discount) FILTER(WHERE eligible) eligible_discount,
        count(*) FILTER(WHERE evaluable) evaluable_n,
        count(*) FILTER(WHERE evaluable AND abs(residual)<=0.02) reconciled_n,
        count(*) FILTER(WHERE evaluable AND quantities_not_one>0) quantity_n,
        count(*) FILTER(WHERE evaluable AND quantities_not_one>0 AND abs(residual)<=0.02) quantity_reconciled_n,
        count(*) FILTER(WHERE evaluable AND has_discount) discounted_n,
        count(*) FILTER(WHERE evaluable AND has_discount AND abs(residual)<=0.02) discounted_reconciled_n,
        avg((abs(residual)<=0.02)::DOUBLE) FILTER(WHERE evaluable) reconciliation
      FROM iterative_accounts GROUP BY 1""")
    con.execute("""CREATE OR REPLACE TEMP TABLE iterative_contexts AS
      SELECT seller_id,location_id,week_start,revenue_center_id,time_band,weekday,
        count(*) n,sum(has_discount::INT) discounted,sum(has_cash::INT) cash,sum(has_void::INT) voids,
        count(*) FILTER(WHERE eligible) money_n,
        sum(gross) FILTER(WHERE eligible) gross,sum(discount) FILTER(WHERE eligible) discount,
        sum(discount/gross) FILTER(WHERE eligible) ratio_sum,
        sum(full_discount::INT) FILTER(WHERE eligible) full_n,
        sum(abs(unexplained_reduction)>0.02) FILTER(WHERE eligible) unexplained_n,
        sum(alternative_discount) FILTER(WHERE alternative_eligible) alternative_discount,
        count(*) FILTER(WHERE alternative_eligible) alternative_n,
        sum(gross) FILTER(WHERE alternative_eligible) alternative_gross,
        sum(check_discount) FILTER(WHERE eligible) check_discount,
        sum(item_discount) FILTER(WHERE eligible) item_discount,
        sum((open_time_status='valid' AND close_time_status='valid')::INT) time_valid_n,
        sum(seller_matched::INT) seller_matched_n
      FROM iterative_accounts WHERE seller_id IS NOT NULL GROUP BY ALL""")
    con.execute("""CREATE OR REPLACE TEMP TABLE iterative_pairs AS
      SELECT a.seller_id,a.location_id,a.week_start,d.manager_id,
        (m.id IS NOT NULL) manager_matched,
        count(DISTINCT a.uid) checks,count(*) records,sum(d.amount) discount
      FROM discounts d JOIN iterative_accounts a USING(uid)
      LEFT JOIN unique_employees m ON d.manager_id=m.id
      WHERE d.amount>0 AND a.seller_id IS NOT NULL
      GROUP BY ALL""")
    con.execute("""CREATE OR REPLACE TEMP TABLE iterative_motives AS
      SELECT a.seller_id,a.location_id,a.week_start,coalesce(d.discount_id,'unknown') discount_id,
        count(DISTINCT a.uid) checks,count(*) records,sum(d.amount) discount
      FROM discounts d JOIN iterative_accounts a USING(uid)
      WHERE d.amount>0 AND a.seller_id IS NOT NULL GROUP BY ALL""")
    for name, path in zip(("accounts", "contexts", "weeks", "pairs", "motives"), files):
        con.execute(f"COPY iterative_{name} TO '{_quote(path)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    info = {"version": VERSION, "fingerprint": digest, "accounts": after[0], "recorded_discount": after[1],
            "money_convention": "line / records_sum / subtract; provisional, row-local",
            "alternative": "line / price_difference_plus_check / subtract"}
    marker.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def attach_inputs(con, output):
    for name in ("accounts", "contexts", "weeks", "pairs", "motives"):
        _drop_input(con,name)
        con.execute(f"CREATE OR REPLACE TEMP VIEW iterative_{name} AS SELECT * FROM read_parquet('{_quote(Path(output)/(name+'.parquet'))}')")


SUM_COLUMNS = ["n", "discounted", "cash", "voids", "money_n", "gross", "discount", "ratio_sum", "full_n",
               "unexplained_n", "alternative_discount", "alternative_n", "alternative_gross", "check_discount",
               "item_discount", "time_valid_n", "seller_matched_n"]


def weekly_dataset(contexts, weeks, pairs, motives, cfg):
    weekly = contexts.groupby(KEY, as_index=False)[SUM_COLUMNS].sum()
    weeks = weeks.copy()
    weeks["week_start"] = pd.to_datetime(weeks.week_start)
    weekly["week_start"] = pd.to_datetime(weekly.week_start)
    weekly = weekly.merge(weeks[["week_start", "source_days","evaluable_n","reconciled_n","quantity_n",
                                 "quantity_reconciled_n","discounted_n","discounted_reconciled_n"]], on="week_start", validate="many_to_one")
    weekly["phase"] = phase_for(weekly.week_start, cfg)
    weekly.loc[weekly.source_days != 7, "phase"] = "descriptive"
    weekly["money_coverage"] = weekly.money_n/weekly.n
    weekly["time_coverage"] = weekly.time_valid_n/weekly.n
    weekly["seller_coverage"] = weekly.seller_matched_n/weekly.n
    for num, denom, name in [("discounted","n","discount_rate"),("discount","gross","discount_share"),
                              ("ratio_sum","money_n","mean_ratio"),("cash","n","cash_rate"),
                              ("voids","n","void_rate"),("full_n","money_n","full_rate"),
                              ("unexplained_n","money_n","unexplained_rate")]:
        weekly[name] = weekly[num]/weekly[denom].replace(0, np.nan)
    for frame, prefix in ((motives, "motive"), (pairs.loc[pairs.manager_matched], "manager")):
        frame = frame.copy()
        frame["week_start"] = pd.to_datetime(frame.week_start)
        # Concentration is defined over positive discount RECORDS, not account exposure.
        denom = frame.groupby(KEY).records.transform("sum")
        frame["squared_share"] = (frame.records/denom)**2
        concentration = frame.groupby(KEY).agg(**{prefix+"_hhi": ("squared_share", "sum"),
                                                  prefix+"_records": ("records", "sum")}).reset_index()
        weekly = weekly.merge(concentration, on=KEY, how="left", validate="one_to_one")
    known = pairs.loc[pairs.manager_matched].groupby(KEY).records.sum()
    all_records = pairs.groupby(KEY).records.sum()
    coverage = (known/all_records).fillna(0).rename("manager_coverage").reset_index()
    coverage["week_start"] = pd.to_datetime(coverage.week_start)
    weekly = weekly.merge(coverage, on=KEY, how="left", validate="one_to_one")
    weekly[["manager_hhi","motive_hhi"]] = weekly[["manager_hhi","motive_hhi"]].fillna(0)
    weekly["manager_coverage"] = weekly.manager_coverage.fillna(1)  # no discount: no required authorization
    weekly["quality_ok"] = ((weekly.money_coverage >= cfg.min_coverage) & (weekly.time_coverage >= cfg.min_coverage)
                            & (weekly.seller_coverage >= cfg.min_coverage) & (weekly.manager_coverage >= cfg.min_coverage))
    weekly["enough_exposure"] = weekly.n >= cfg.min_checks
    weekly["dataset_version"] = VERSION
    return weekly.sort_values(KEY).reset_index(drop=True)


class PeerReference:
    """Leave-employee-out empirical Bayes baseline, fitted on past contexts only.

    Fallback: local activity -> global activity. Context support excludes the
    target employee; sparse groups shrink toward that local reference.
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, contexts):
        self.history = contexts.copy()
        self.history["week_start"] = pd.to_datetime(self.history.week_start)
        if self.history.empty:
            raise ValueError("Referencia de pares vacía")
        self.cutoff = self.history.week_start.max()
        return self

    def transform(self, contexts, weekly):
        x = contexts.copy()
        hist = self.history
        # Employee-out aggregates for the context, location, and global fallback.
        for cols, prefix in ((CONTEXT,"ctx"),(["location_id"],"loc"),([],"all")):
            values = ["n","discounted","money_n","ratio_sum","cash","voids"]
            if cols:
                total = hist.groupby(cols, dropna=False)[values].sum().add_prefix(prefix+"_total_").reset_index()
                own = hist.groupby(cols+["seller_id"], dropna=False)[values].sum().add_prefix(prefix+"_own_").reset_index()
                peers = hist.groupby(cols, dropna=False).seller_id.nunique().rename(prefix+"_people").reset_index()
                x = x.merge(total, on=cols, how="left", validate="many_to_one").merge(own, on=cols+["seller_id"], how="left", validate="many_to_one")
                x = x.merge(peers, on=cols, how="left", validate="many_to_one")
            else:
                for v in values:
                    x[prefix+"_total_"+v] = hist[v].sum()
                own = hist.groupby("seller_id")[values].sum().add_prefix(prefix+"_own_").reset_index()
                x = x.merge(own,on="seller_id",how="left",validate="many_to_one")
                x[prefix+"_people"] = hist.seller_id.nunique()
            for v in values:
                x[prefix+"_"+v] = x[prefix+"_total_"+v].fillna(0)-x[prefix+"_own_"+v].fillna(0)
            x[prefix+"_people"] = x[prefix+"_people"].fillna(0)-(x[prefix+"_own_n"].notna()).astype(int)
        for numerator, denom, name in (("discounted","n","rate"),("ratio_sum","money_n","ratio"),
                                        ("cash","n","cash"),("voids","n","void")):
            global_mean = (x["all_"+numerator]/x["all_"+denom].replace(0,np.nan)).fillna(0)
            local_mean = ((x["loc_"+numerator]+self.cfg.prior_checks*global_mean)/
                          (x["loc_"+denom]+self.cfg.prior_checks))
            x["peer_"+name] = ((x["ctx_"+numerator]+self.cfg.prior_checks*local_mean)/
                               (x["ctx_"+denom]+self.cfg.prior_checks))
            x["expected_"+name] = x["peer_"+name]*x["n" if name != "ratio" else "money_n"]
        x["supported_n"] = x.n*((x.ctx_people>=self.cfg.min_peers)&(x.ctx_n>=self.cfg.min_checks))
        agg = x.groupby(KEY)[["expected_rate","expected_ratio","expected_cash","expected_void","supported_n"]].sum().reset_index()
        result = weekly.merge(agg,on=KEY,how="left",validate="one_to_one")
        for name, denom in (("rate","n"),("ratio","money_n"),("cash","n"),("void","n")):
            result["peer_"+name] = result["expected_"+name]/result[denom].replace(0,np.nan)
        strength = self.cfg.prior_checks
        result["smoothed_rate"] = (result.discounted+strength*result.peer_rate)/(result.n+strength)
        result["excess_rate"] = result.smoothed_rate-result.peer_rate
        result["smoothed_ratio"] = (result.ratio_sum+strength*result.peer_ratio)/(result.money_n+strength)
        result["excess_ratio"] = result.smoothed_ratio-result.peer_ratio
        result["excess_cash"] = result.cash_rate-result.peer_cash
        result["excess_void"] = result.void_rate-result.peer_void
        result["peer_coverage"] = result.supported_n/result.n
        result["model_eligible"] = (result.quality_ok & result.enough_exposure &
                                      (result.peer_coverage>=self.cfg.min_coverage) & (result.source_days==7))
        # Beta posterior intervals describe sampling uncertainty conditional on this reference.
        from scipy.stats import beta
        alpha = result.discounted+strength*result.peer_rate.clip(.00001,.99999)
        beta_param = result.n-result.discounted+strength*(1-result.peer_rate.clip(.00001,.99999))
        result["rate_low"] = beta.ppf(.025,alpha,beta_param)
        result["rate_high"] = beta.ppf(.975,alpha,beta_param)
        result["excess_evidence"] = (result.rate_low>result.peer_rate)&(result.excess_rate>0)
        result["reference_end"] = self.cutoff
        return result
