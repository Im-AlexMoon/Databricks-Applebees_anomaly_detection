"""Aggregated EDA and chronological, peer-adjusted anomaly exploration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest
from statsmodels.stats.proportion import proportion_confint

from .pipeline import Config, monetary_gate


def aggregate_frame(con, query, parameters=None, maximum_rows=200_000):
    """Guard the pandas boundary, including high-cardinality employee aggregates."""
    result = con.execute(f"SELECT * FROM ({query}) q LIMIT {maximum_rows+1}", parameters or []).fetchdf()
    if len(result) > maximum_rows:
        raise ValueError("Agregado demasiado grande para pandas; reduce el período configurado.")
    return result


def baseline(con, group="location_id"):
    allowed = {"location_id", "revenue_center_id", "business_day", "weekday", "time_band", "week_start"}
    if group not in allowed:
        raise ValueError("Agrupación no permitida")
    return aggregate_frame(con, f"""SELECT {group},count(*) checks,sum(has_discount::INTEGER) discounted_checks,
        avg(has_discount::DOUBLE) discount_rate, count(*) FILTER (WHERE money_eligible) reconciled_checks,
        sum(gross_sales) FILTER (WHERE money_eligible) gross_sales_reconciled,
        sum(discount_amount) FILTER (WHERE money_eligible) discount_amount_reconciled,
        sum(discount_amount) FILTER (WHERE money_eligible)/nullif(sum(gross_sales) FILTER (WHERE money_eligible),0) discount_to_gross,
        median(discount_ratio) FILTER (WHERE money_eligible) median_check_discount_ratio,
        quantile_cont(discount_ratio,0.95) FILTER (WHERE money_eligible) p95_check_discount_ratio,
        avg(has_cash::DOUBLE) cash_check_rate, avg((void_records>0)::DOUBLE) void_check_rate
        FROM facts GROUP BY 1 ORDER BY checks DESC""")


def discount_types(con, limit=100):
    # Codes and labels remain local to a location; no global meaning is inferred.
    return aggregate_frame(con, """SELECT f.location_id,d.discount_id,d.discount_name,d.level,
        count(*) records,count(DISTINCT d.uid) checks,sum(d.amount) recorded_amount_provisional,
        count(*) FILTER (WHERE d.amount>0) positive_records,
        count(*) FILTER (WHERE d.amount=0) zero_records,
        count(*) FILTER (WHERE d.amount<0) negative_records
        FROM discounts d JOIN facts f USING(uid)
        GROUP BY 1,2,3,4 ORDER BY checks DESC LIMIT ?""", [limit])


def check_sample(con, limit=10_000, seed=42, monetary=False):
    condition = "money_eligible" if monetary else "true"
    return con.execute(f"""SELECT uid,location_id,business_day,seller_id,discount_records,has_discount,
        recorded_discount,discount_ratio,duration_minutes,has_cash,void_records,gross_sales,discount_amount
        FROM facts WHERE {condition} ORDER BY hash(uid,?) LIMIT ?""", [seed, limit]).fetchdf()


def seller_contexts(con, config: Config):
    frame = aggregate_frame(con, """SELECT location_id,revenue_center_id,time_band,seller_id,
        count(*) checks,sum(has_discount::INTEGER) discounted_checks,
        count(*) FILTER (WHERE money_eligible) reconciled_checks,
        sum(gross_sales) FILTER (WHERE money_eligible) gross_sales_reconciled,
        sum(discount_amount) FILTER (WHERE money_eligible) discount_amount_reconciled,
        sum(discount_amount) FILTER (WHERE money_eligible)/nullif(sum(gross_sales) FILTER (WHERE money_eligible),0) discount_to_gross,
        sum((void_records>0)::INTEGER) void_checks,sum(has_cash::INTEGER) cash_checks,
        bool_or(seller_deleted) seller_deleted,bool_or(seller_borrowed) seller_borrowed
        FROM facts WHERE seller_id IS NOT NULL
        GROUP BY 1,2,3,4""")
    if frame.empty:
        return frame
    frame["discount_rate"] = frame.discounted_checks / frame.checks
    low, high = proportion_confint(frame.discounted_checks.to_numpy(), frame.checks.to_numpy(), method="wilson")
    frame["wilson_low"], frame["wilson_high"] = low, high
    frame["limited_sample"] = frame.checks < config.minimum_peer_checks
    frame["peer_count"] = 0
    frame["peer_median_rate"] = np.nan
    frame["peer_percentile"] = np.nan
    frame["robust_rate_distance"] = np.nan
    for key, group in frame.groupby(["location_id", "revenue_center_id", "time_band"], dropna=False):
        if any(pd.isna(v) for v in key) or key[2]=="unknown":
            continue
        for index in group.index:
            others = group[(group.index != index) & ~group.limited_sample]
            frame.loc[index, "peer_count"] = len(others)
            if len(others) < config.minimum_peer_employees or frame.loc[index, "limited_sample"]:
                continue
            values = others.discount_rate.to_numpy()
            value = frame.loc[index, "discount_rate"]
            median = np.median(values)
            frame.loc[index, "peer_median_rate"] = median
            # Mid-rank ties: equal rates are not all placed in the highest percentile.
            frame.loc[index, "peer_percentile"] = (np.sum(values < value) + .5*np.sum(values == value))/len(values)
            mad = median_abs_deviation(values, scale="normal")
            if mad > 0:
                frame.loc[index, "robust_rate_distance"] = (value-median)/mad
    frame["rate_excess"] = frame.discount_rate-frame.peer_median_rate
    frame["review_signal"] = (frame.peer_percentile >= .95) & (frame.rate_excess > 0)
    return frame.sort_values(["review_signal", "rate_excess", "checks"], ascending=[False, False, False])


def actor_roles(con):
    """Separate exposures: applicants/approvers do not inherit seller denominators."""
    return aggregate_frame(con, """WITH events AS (
        SELECT f.location_id,d.uid,d.employee_id actor_id,'discount_applicant' actor_role,d.amount
          FROM discounts d JOIN facts f USING(uid) WHERE d.amount>0
        UNION ALL
        SELECT f.location_id,d.uid,d.manager_id,'discount_manager',d.amount
          FROM discounts d JOIN facts f USING(uid) WHERE d.amount>0)
        SELECT location_id,actor_id,actor_role,count(*) positive_records,count(DISTINCT uid) discounted_checks,
          sum(amount) recorded_amount_provisional FROM events GROUP BY 1,2,3 ORDER BY discounted_checks DESC""")


def employee_manager_pairs(con, maximum=5_000):
    return aggregate_frame(con, """WITH pairs AS (
        SELECT f.location_id,d.employee_id,d.manager_id,count(DISTINCT d.uid) pair_checks,
            count(*) positive_records,sum(d.amount) recorded_amount_provisional
        FROM discounts d JOIN facts f USING(uid)
        WHERE d.amount>0 AND d.employee_id IS NOT NULL AND d.manager_id IS NOT NULL
        GROUP BY 1,2,3
    ), counts AS (
        SELECT *,sum(positive_records) OVER (PARTITION BY location_id,employee_id) employee_records,
            sum(positive_records) OVER (PARTITION BY location_id,manager_id) manager_records,
            sum(positive_records) OVER (PARTITION BY location_id) location_records FROM pairs
    ) SELECT *,positive_records::DOUBLE/employee_records employee_share,
        positive_records::DOUBLE/manager_records manager_share,
        positive_records::DOUBLE*location_records/(employee_records*manager_records) observed_expected_ratio
    FROM counts ORDER BY pair_checks DESC LIMIT ?""", [maximum])


def weekly_trends(con):
    frame = aggregate_frame(con, """SELECT location_id,seller_id,week_start,count(*) checks,
        sum(has_discount::INTEGER) discounted_checks,
        avg(has_discount::DOUBLE) discount_rate,
        count(DISTINCT business_day) observed_business_days,
        count(*) FILTER (WHERE money_eligible) reconciled_checks,
        sum(discount_amount) FILTER (WHERE money_eligible)/nullif(sum(gross_sales) FILTER (WHERE money_eligible),0) discount_to_gross
        FROM facts WHERE seller_id IS NOT NULL GROUP BY 1,2,3 ORDER BY 1,2,3""")
    if not frame.empty:
        # A gap starts a new rolling segment; missing weeks are not zero activity.
        frame["segment"] = frame.groupby(["location_id", "seller_id"]).week_start.transform(
            lambda s: s.diff().dt.days.ne(7).cumsum())
        frame["rolling_median_4_observed_weeks"] = frame.groupby(
            ["location_id", "seller_id", "segment"]).discount_rate.transform(lambda s: s.rolling(4, min_periods=2).median())
    return frame


def associations(con):
    """Within-context descriptive rates; categories include zero-price modifiers."""
    cash_void = aggregate_frame(con, """SELECT location_id,revenue_center_id,time_band,weekday,
        has_cash,void_records>0 has_void,count(*) checks,sum(has_discount::INTEGER) discounted_checks,
        avg(has_discount::DOUBLE) discount_rate,
        count(*) FILTER (WHERE money_eligible AND full_discount) full_discount_checks,
        count(*) FILTER (WHERE money_eligible) reconciled_checks
        FROM facts GROUP BY 1,2,3,4,5,6""")
    categories = aggregate_frame(con, """WITH presence AS (
        SELECT DISTINCT uid,major_category_id,major_category_name,sub_category_id,sub_category_name
        FROM sold_items)
        SELECT f.location_id,f.revenue_center_id,f.time_band,p.major_category_id,p.major_category_name,
            p.sub_category_id,p.sub_category_name,count(*) checks,sum(f.has_discount::INTEGER) discounted_checks,
            avg(f.has_discount::DOUBLE) discount_rate
        FROM presence p JOIN facts f USING(uid) GROUP BY 1,2,3,4,5,6,7""")
    # Explicit rate differences against the other payment group, preserving context.
    payment = aggregate_frame(con, """SELECT location_id,revenue_center_id,time_band,weekday,has_cash,
        count(*) checks,avg(has_discount::DOUBLE) discount_rate
        FROM facts GROUP BY 1,2,3,4,5""")
    if not payment.empty:
        pivot = payment.pivot(index=["location_id", "revenue_center_id", "time_band", "weekday"],
            columns="has_cash", values=["checks", "discount_rate"])
        pivot = pivot.reindex(columns=pd.MultiIndex.from_product([["checks", "discount_rate"], [False, True]]))
        pivot.columns = [f"{metric}_{'cash' if value else 'no_cash'}" for metric,value in pivot.columns]
        pivot["rate_difference_cash_minus_no_cash"] = pivot.discount_rate_cash-pivot.discount_rate_no_cash
        payment = pivot.reset_index()
    return {"cash_void": cash_void, "categories": categories, "payment_rate_differences": payment}


MODEL_FEATURES = ["discount_rate_excess", "discount_ratio_excess", "void_rate_excess", "cash_rate_excess"]
CONTEXT = ["location_id", "revenue_center_id", "time_band"]


def model_features(con, config):
    """Peer expectations from April-July only; exclude the seller being scored."""
    frame = aggregate_frame(con, """SELECT location_id,revenue_center_id,time_band,seller_id,week_start,
        CASE WHEN business_day<DATE '2026-08-01' THEN 'reference' ELSE 'august' END phase,
        min(business_day) period_start,max(business_day) period_end,
        count(*) checks,sum(has_discount::INTEGER) discounted_checks,
        sum((void_records>0)::INTEGER) void_checks,sum(has_cash::INTEGER) cash_checks,
        count(*) FILTER (WHERE money_eligible) money_checks,
        sum(gross_sales) FILTER (WHERE money_eligible) gross,
        sum(discount_amount) FILTER (WHERE money_eligible) discount
        FROM facts WHERE seller_id IS NOT NULL AND business_day BETWEEN DATE '2026-04-01' AND DATE '2026-08-31'
        GROUP BY 1,2,3,4,5,6""")
    if frame.empty:
        return frame, {"context_rows": 0, "covered_context_rows": 0, "weekly_rows": 0}
    reference = frame[frame.phase=="reference"]
    peers = reference.groupby(CONTEXT+["seller_id"], dropna=False)[
        ["checks", "discounted_checks", "void_checks", "cash_checks", "money_checks", "gross", "discount"]].sum(min_count=1).reset_index()
    peers = peers[(peers.checks>=config.minimum_peer_checks) & (peers.money_checks>=config.minimum_peer_checks) & (peers.gross>0)].copy()
    for dest, numerator, denominator in [("discount_rate", "discounted_checks", "checks"),
            ("void_rate", "void_checks", "checks"), ("cash_rate", "cash_checks", "checks"),
            ("discount_ratio", "discount", "gross")]:
        peers[dest] = peers[numerator]/peers[denominator]
    lookup = {key: group for key,group in peers.groupby(CONTEXT, dropna=False)}
    # Contexts with unknown codes/time bands are not comparable.
    for metric in ["discount_rate", "discount_ratio", "void_rate", "cash_rate"]:
        frame[f"expected_{metric}"] = np.nan
    for index, row in frame.iterrows():
        key = tuple(row[k] for k in CONTEXT)
        if any(pd.isna(v) for v in key) or row.time_band=="unknown":
            continue
        group = lookup.get(key)
        if group is None:
            continue
        others = group[group.seller_id!=row.seller_id]
        if len(others) < config.minimum_peer_employees:
            continue
        for metric in ["discount_rate", "discount_ratio", "void_rate", "cash_rate"]:
            frame.loc[index, f"expected_{metric}"] = others[metric].median()
    frame["covered"] = frame.expected_discount_rate.notna() & frame.expected_discount_ratio.notna()
    total = frame.groupby(["location_id", "seller_id", "week_start", "phase"], dropna=False).checks.sum().rename("all_checks")
    usable = frame[frame.covered].copy()
    if usable.empty:
        return usable, {"context_rows": len(frame), "covered_context_rows": 0, "weekly_rows": 0}
    for metric in ("discount_rate", "void_rate", "cash_rate"):
        usable[f"expected_{metric}_count"] = usable[f"expected_{metric}"]*usable.checks
    usable["expected_discount"] = usable.expected_discount_ratio*usable.gross
    group_keys = ["location_id", "seller_id", "week_start", "phase"]
    weekly = usable.groupby(group_keys, dropna=False).agg(
        checks=("checks", "sum"), discounted_checks=("discounted_checks", "sum"),
        void_checks=("void_checks", "sum"), cash_checks=("cash_checks", "sum"),
        money_checks=("money_checks", "sum"), gross=("gross", "sum"), discount=("discount", "sum"),
        expected_discount_count=("expected_discount_rate_count", "sum"),
        expected_void_count=("expected_void_rate_count", "sum"), expected_cash_count=("expected_cash_rate_count", "sum"),
        expected_discount=("expected_discount", "sum"), period_start=("period_start", "min"), period_end=("period_end", "max"))
    weekly = weekly.join(total).reset_index()
    weekly["context_coverage"] = weekly.checks/weekly.all_checks
    weekly["money_coverage"] = weekly.money_checks/weekly.checks
    weekly["discount_rate_excess"] = (weekly.discounted_checks-weekly.expected_discount_count)/weekly.checks
    weekly["void_rate_excess"] = (weekly.void_checks-weekly.expected_void_count)/weekly.checks
    weekly["cash_rate_excess"] = (weekly.cash_checks-weekly.expected_cash_count)/weekly.checks
    weekly["discount_ratio_excess"] = (weekly.discount-weekly.expected_discount)/weekly.gross
    eligible = weekly[(weekly.context_coverage>=.8) & (weekly.money_coverage>=.8) &
        (weekly.checks>=config.minimum_peer_checks) & (weekly.money_checks>=config.minimum_peer_checks)]
    eligible = eligible.replace([np.inf,-np.inf], np.nan).dropna(subset=MODEL_FEATURES)
    return eligible, {"context_rows": len(frame), "covered_context_rows": int(frame.covered.sum()),
        "weekly_rows": len(weekly), "eligible_weekly_rows": len(eligible)}


def explore_anomalies(con, config):
    okay, message = monetary_gate(con, config)
    result = {"status": "skipped", "message": message, "features": pd.DataFrame(), "candidates": pd.DataFrame(), "diagnostics": {}}
    if not okay:
        return result
    frame, diagnostics = model_features(con, config)
    result.update(features=frame, diagnostics=diagnostics)
    if frame.empty:
        result["message"] = "Sin semanas con volumen, conciliación y pares suficientes."
        return result
    train = frame[frame.phase=="reference"].copy()
    august = frame[frame.phase=="august"].copy()
    diagnostics.update(training_rows=len(train), august_rows=len(august), training_sellers=train.seller_id.nunique())
    if len(train)<config.minimum_training_rows or train.seller_id.nunique()<config.minimum_training_sellers or august.empty:
        result["message"] = "Histórico o agosto insuficientes: se necesitan al menos " + str(config.minimum_training_rows) + " semanas de referencia y " + str(config.minimum_training_sellers) + " vendedores."
        return result
    if (train[MODEL_FEATURES].nunique()<=1).all():
        result["message"] = "Las variables de referencia no tienen variación suficiente."
        return result
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=config.random_state, n_jobs=1)
    model.fit(train[MODEL_FEATURES])
    reference_scores = -model.score_samples(train[MODEL_FEATURES])
    august["anomaly_score"] = -model.score_samples(august[MODEL_FEATURES])
    ordered = np.sort(reference_scores)
    august["reference_score_percentile"] = [float(np.searchsorted(ordered, s, side="right")/len(ordered)) for s in august.anomaly_score]
    centers = train[MODEL_FEATURES].median()
    scales = train[MODEL_FEATURES].apply(lambda s: median_abs_deviation(s, scale="normal"))
    explanations = []
    for _,row in august.iterrows():
        signals = []
        for feature in MODEL_FEATURES:
            delta = row[feature]-centers[feature]
            if (scales[feature]>0 and abs(delta)/scales[feature]>=2.5) or (scales[feature]==0 and abs(delta)>1e-9):
                signals.append(f"{feature}={row[feature]:.3f}; mediana referencia={centers[feature]:.3f}")
        explanations.append("; ".join(signals) or "Combinación de variables inusual; revisar los valores adjuntos")
    august["observed_signals"] = explanations
    august["interpretation"] = "Anomalía exploratoria; no probabilidad ni etiqueta de fraude"
    # Keep all scores in features; the review list focuses on weeks with discounts.
    candidates = august[august.discounted_checks>0].sort_values("anomaly_score", ascending=False).head(25)
    result.update(status="completed", message="Referencia abril-julio; puntuación de agosto sin etiquetas de fraude.",
        candidates=candidates, scored_august=august, reference_features=train)
    return result


def review_checks(con, config, sellers=None, limit=100):
    """Explainable frequency signals; amount signals require reconciliation."""
    okay, _ = monetary_gate(con, config)
    where = ["has_discount"]
    parameters = []
    if sellers is not None:
        if sellers.empty:
            return pd.DataFrame()
        # Restrict to the anomalous seller, location AND week/phase boundaries.
        clauses = []
        for _,row in sellers.iterrows():
            clauses.append("(seller_id=? AND location_id=? AND business_day BETWEEN ?::DATE AND ?::DATE)")
            parameters.extend([str(row.seller_id), str(row.location_id), str(row.period_start.date()), str(row.period_end.date())])
        where.append("(" + " OR ".join(clauses) + ")")
    monetary_columns = "discount_amount,gross_sales,discount_ratio,full_discount," if okay else ""
    frame = con.execute(f"""SELECT uid,check_id,location_id,business_day,seller_id,seller_count,
        {monetary_columns} recorded_discount AS recorded_amount_provisional,
        check_discount_records,item_discount_records,has_cash,void_records,
        duration_minutes,money_eligible,source_file,source_row,
        CASE WHEN check_discount_records>0 AND item_discount_records>0 THEN 'Descuentos en ambos niveles; revisar conciliación'
             WHEN void_records>0 THEN 'Descuento y anulación en la misma cuenta'
             WHEN has_cash THEN 'Descuento en cuenta con efectivo'
             ELSE 'Cuenta con descuento positivo registrado' END observed_signal
        FROM facts WHERE {' AND '.join(where)}
        ORDER BY {'full_discount DESC,money_eligible DESC,discount_ratio DESC NULLS LAST,' if okay else ''}
            (void_records>0) DESC,has_cash DESC,discount_records DESC,uid LIMIT ?""", parameters+[limit]).fetchdf()
    if not frame.empty:
        frame["interpretation"] = "Caso para revisión; importe registrado no equivale a pérdida demostrada"
    return frame


def export_reports(config, quality, seller_frame, pair_frame, cases, model):
    root = config.output_dir / "reports"
    root.mkdir(parents=True, exist_ok=True)
    for name,frame in quality.items():
        frame.to_csv(root/f"quality_{name}.csv", index=False)
    seller_frame.to_csv(root/"seller_contexts.csv", index=False)
    pair_frame.to_csv(root/"employee_manager_pairs.csv", index=False)
    cases.to_csv(root/"review_checks.csv", index=False)
    model["candidates"].to_csv(root/"model_candidates_august.csv", index=False)
    import json
    (root/"model_status.json").write_text(json.dumps({"status": model["status"], "message": model["message"],
        "diagnostics": model["diagnostics"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return root
