"""Reproducible local orchestration; private artifacts remain under outputs/."""
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd

from .iterative_data import VERSION, KEY, ExperimentConfig, build_inputs, weekly_dataset
from .iterative_statistics import (binomial_context_model, intensity_bootstrap, full_pair_analysis,
                                   descriptive_reports, hypothesis_log)
from .weak_supervision import FEATURES, run_teachers, run_students
from .pipeline import records


def json_write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding="utf-8")


def analysis_inputs(con, monetary_mode):
    """Switch conventions without changing or re-ingesting the immutable cache."""
    if monetary_mode not in ("records_sum","price_difference_plus_check"):
        raise ValueError("Convención experimental desconocida")
    if monetary_mode=="records_sum":
        con.execute("CREATE OR REPLACE TEMP VIEW analysis_accounts AS SELECT * FROM iterative_accounts")
        contexts=con.execute("SELECT * FROM iterative_contexts").fetchdf()
        weeks=con.execute("SELECT * FROM iterative_weeks").fetchdf()
    else:
        con.execute("""CREATE OR REPLACE TEMP VIEW analysis_accounts AS SELECT * REPLACE(
          alternative_discount AS discount,alternative_residual AS residual,alternative_eligible AS eligible,
          (gross>0 AND abs(gross-alternative_discount)<=0.02) AS full_discount) FROM iterative_accounts""")
        contexts=con.execute("""SELECT seller_id,location_id,week_start,revenue_center_id,time_band,weekday,
          count(*) n,sum(has_discount::INT) discounted,sum(has_cash::INT) cash,sum(has_void::INT) voids,
          count(*) FILTER(WHERE eligible) money_n,sum(gross) FILTER(WHERE eligible) gross,
          sum(discount) FILTER(WHERE eligible) discount,sum(discount/gross) FILTER(WHERE eligible) ratio_sum,
          sum(full_discount::INT) FILTER(WHERE eligible) full_n,
          sum(abs(unexplained_reduction)>0.02) FILTER(WHERE eligible) unexplained_n,
          sum(alternative_discount) FILTER(WHERE alternative_eligible) alternative_discount,
          count(*) FILTER(WHERE alternative_eligible) alternative_n,
          sum(gross) FILTER(WHERE alternative_eligible) alternative_gross,
          sum(check_discount) FILTER(WHERE eligible) check_discount,sum(item_discount) FILTER(WHERE eligible) item_discount,
          sum((open_time_status='valid' AND close_time_status='valid')::INT) time_valid_n,
          sum(seller_matched::INT) seller_matched_n
          FROM analysis_accounts WHERE seller_id IS NOT NULL GROUP BY ALL""").fetchdf()
        weeks=con.execute("""SELECT week_start,count(DISTINCT business_day) source_days,count(*) accounts,
          count(*) FILTER(WHERE seller_id IS NULL) unattributed_accounts,count(*) FILTER(WHERE eligible) eligible_accounts,
          sum(discount) recorded_discount,sum(discount) FILTER(WHERE eligible) eligible_discount,
          count(*) FILTER(WHERE evaluable) evaluable_n,
          count(*) FILTER(WHERE evaluable AND abs(residual)<=0.02) reconciled_n,
          count(*) FILTER(WHERE evaluable AND quantities_not_one>0) quantity_n,
          count(*) FILTER(WHERE evaluable AND quantities_not_one>0 AND abs(residual)<=0.02) quantity_reconciled_n,
          count(*) FILTER(WHERE evaluable AND has_discount) discounted_n,
          count(*) FILTER(WHERE evaluable AND has_discount AND abs(residual)<=0.02) discounted_reconciled_n,
          avg((abs(residual)<=0.02)::DOUBLE) FILTER(WHERE evaluable) reconciliation
          FROM analysis_accounts GROUP BY ALL""").fetchdf()
    contexts["week_start"]=pd.to_datetime(contexts.week_start)
    return contexts,weeks


def review_sample(con, cases, output, cfg, per_group=3):
    """Read selected source rows from JSON/ZIP, never infer confirmed fraud."""
    if cases.empty:
        pd.DataFrame(columns=["uid","sample_group","fraud_confirmed"]).to_csv(output/"json_review.csv",index=False)
        return
    rng=np.random.default_rng(cfg.seed)
    selected=[]
    groups={"alert":cases.loc[cases.pseudo_label==1],
            "disagreement":cases.loc[cases.detector_disagreement],
            "random_control":cases.loc[cases.pseudo_label==0]}
    seen=set()
    for name,frame in groups.items():
        if frame.empty:
            continue
        indices=rng.permutation(frame.index)[:per_group]
        for index in indices:
            row=frame.loc[index]
            candidate=con.execute("""SELECT uid,source_file,source_row,check_id,location_id,discount,gross,
                has_discount,has_cash,has_void FROM iterative_accounts
                WHERE seller_id=? AND location_id=? AND week_start=?
                ORDER BY has_discount DESC,hash(uid,?) LIMIT 1""",
                [row.seller_id,row.location_id,row.week_start.to_pydatetime(),cfg.seed]).fetchdf()
            if len(candidate) and candidate.uid.iloc[0] not in seen:
                item=candidate.iloc[0].to_dict()
                item.update(sample_group=name,seller_id=row.seller_id,week_start=str(row.week_start),
                            weekly_excess_pp=float(row.excess_rate)*100,weekly_discount=float(row.discount))
                selected.append(item); seen.add(item["uid"])
    # Group by source so each selected daily member is scanned only once.
    for source in sorted({r["source_file"] for r in selected}):
        targets={int(r["source_row"]):r for r in selected if r["source_file"]==source}
        for row_number,raw in enumerate(records(source),1):
            if row_number in targets:
                item=targets[row_number]
                all_discounts=list(raw.get("Discounts") or [])
                for sold in raw.get("ItemsSold") or []:
                    all_discounts.extend(sold.get("Discounts") or [])
                raw_total=sum(float(d.get("Amount") or 0) for d in all_discounts)
                identity=(str(raw.get("Id"))==str(item["check_id"]) and str(raw.get("LocationId"))==str(item["location_id"]))
                if not identity or not np.isclose(raw_total,item["discount"],atol=.02,rtol=0):
                    raise AssertionError("La revisión JSON no coincide con Parquet")
                item.update(raw_identity_matches=True,raw_discount_matches=True,
                            recorded_reasons=json.dumps(sorted({str(d.get("DiscountName")) for d in all_discounts if d.get("DiscountName")}),ensure_ascii=False),
                            evidence="Identidad e importe contrastados con JSON; ver señales semanales y motivos registrados",
                            alternatives="Promoción o autorización legítima posible; no hay reglas ni evidencia operativa",
                            missing_information="Políticas, turnos, hora de aplicación y revisión operativa independiente",
                            decision="reformular" if item["sample_group"]=="random_control" else "continuar",fraud_confirmed=None)
            if row_number>=max(targets):
                break
    if selected and not all(r.get("raw_identity_matches") for r in selected):
        raise AssertionError("No se encontraron todas las filas fuente seleccionadas")
    pd.DataFrame(selected).to_csv(output/"json_review.csv",index=False)


def run_experiment(con, config, cfg=None, output_root=None, progress=print, review_json=True, monetary_mode="records_sum"):
    cfg=cfg or ExperimentConfig()
    root=Path(output_root or config.output_dir.parent/"eda_iterativo")
    root.mkdir(parents=True,exist_ok=True)
    modules=["iterative_data.py","iterative_statistics.py","weak_supervision.py","iterative.py"]
    code_hash=hashlib.sha256(b"".join((Path(__file__).parent/name).read_bytes() for name in modules)).hexdigest()
    data_hash=hashlib.sha256((config.output_dir/"parquet"/"manifest.json").read_bytes()).hexdigest()
    input_hash=hashlib.sha256((Path(__file__).parent/"iterative_data.py").read_bytes()+str(config.money_tolerance).encode()).hexdigest()[:10]
    inputs=root/("inputs-"+data_hash[:8]+"-"+input_hash)
    progress("Preparando/reutilizando entradas semanales desde Parquet")
    info=build_inputs(con,config,inputs)
    experiment_hash=hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()+code_hash.encode()+info["fingerprint"].encode()+str(review_json).encode()+monetary_mode.encode()).hexdigest()[:12]
    output=root/(VERSION+"-"+experiment_hash)
    if (output/"complete.json").exists():
        progress("Reutilizando experimento completo: "+str(output))
        return output
    output.mkdir(parents=True,exist_ok=True)
    manifest={"version":VERSION,"config":asdict(cfg),"code_sha256":code_hash,"input_fingerprint":info["fingerprint"],
              "input_directory":str(inputs.resolve()),"started_utc":datetime.now(timezone.utc).isoformat(),
              "python":platform.python_version(),"packages":{p:version(p) for p in ("duckdb","pandas","numpy","scipy","statsmodels","scikit-learn")},
              "features":FEATURES,"labels":"Anomaly hypotheses, no confirmed fraud", "review_json":review_json,"monetary_mode":monetary_mode,
              "prospective_validation":False,"august_interpretation":"retrospective, previously explored",
              "bootstrap_note":"Employee-cluster resampling, selected GMM architecture fixed per fold",
              "threshold_note":"Frozen 95th/50th training quantiles; not fraud prevalence"}
    json_write(output/"manifest.json",manifest)
    contexts,weeks=analysis_inputs(con,monetary_mode)
    pairs=con.execute("SELECT * FROM iterative_pairs").fetchdf()
    motives=con.execute("SELECT * FROM iterative_motives").fetchdf()
    weekly=weekly_dataset(contexts,weeks,pairs,motives,cfg)
    # Do not discard precisely the unexplained reductions being investigated:
    # their denominator includes evaluable accounts that fail reconciliation.
    reductions=con.execute("""SELECT seller_id,location_id,week_start,count(*) FILTER(WHERE evaluable) reduction_evaluable_n,
      sum((abs(unexplained_reduction)>0.02)::INT) FILTER(WHERE evaluable) unexplained_all_n
      FROM analysis_accounts WHERE seller_id IS NOT NULL GROUP BY ALL""").fetchdf()
    weekly=weekly.merge(reductions,on=KEY,validate="one_to_one")
    weekly["unexplained_rate"]=weekly.unexplained_all_n/weekly.reduction_evaluable_n.replace(0,np.nan)
    weekly["unexplained_rate"]=weekly.unexplained_rate.fillna(0)
    weekly["monetary_mode"]=monetary_mode
    con.execute("""SELECT uid,seller_id,location_id,business_day,source_file,source_row,gross,discount,
       alternative_discount,unexplained_reduction,residual,alternative_residual,eligible,alternative_eligible
       FROM analysis_accounts WHERE evaluable AND abs(unexplained_reduction)>0.02""").fetchdf().to_csv(output/"unexplained_reductions.csv",index=False)
    weekly.to_parquet(output/"weekly_dataset.parquet",index=False)
    weeks.to_csv(output/"quality_weekly.csv",index=False)
    progress("Estadística: regresión binomial, bootstrap y todas las parejas")
    glm,glm_status=binomial_context_model(con,cfg)
    glm.to_csv(output/"context_glm.csv",index=False)
    json_write(output/"context_glm_status.json",glm_status)
    intensity=intensity_bootstrap(con,cfg)
    intensity.to_csv(output/"intensity_bootstrap.csv",index=False)
    pair_stats=full_pair_analysis(pairs,cfg)
    pair_stats.to_parquet(output/"pair_tests.parquet",index=False)
    motive_stats=full_pair_analysis(motives.rename(columns={"discount_id":"manager_id"}).assign(manager_matched=True),cfg)
    if not motive_stats.empty:
        motive_stats=motive_stats.rename(columns={"manager_id":"discount_id","manager_total":"motive_total"}).drop(columns="manager_matched")
        motive_stats["family"]="motive_concentration"
    motive_stats.to_parquet(output/"motive_tests.parquet",index=False)
    for name,frame in descriptive_reports(con,weekly).items():
        frame.to_csv(output/(name+".csv"),index=False)
    hypothesis_log(glm,intensity,pair_stats,glm_status).to_csv(output/"hypothesis_log_template.csv",index=False)
    # The human iteration log is editable and never overwritten by re-execution.
    if not (root/"hypothesis_log.csv").exists():
        hypothesis_log(glm,intensity,pair_stats,glm_status).to_csv(root/"hypothesis_log.csv",index=False)
    result=run_teachers(contexts,weekly,cfg,progress)
    for name in ("gmm_selection","bootstrap_stability"):
        result[name].to_csv(output/(name+".csv"),index=False)
    json_write(output/"teacher_status.json",result["teacher_status"])
    scores=result["scores"]
    scores.to_parquet(output/"weekly_scores.parquet",index=False)
    progress("Segunda pasada: comprobando soporte de pseudoetiquetas")
    students=run_students(scores,cfg)
    json_write(output/"student_status.json",students["status"])
    students["metrics"].to_csv(output/"student_metrics.csv",index=False)
    students["predictions"].to_parquet(output/"student_predictions.parquet",index=False)
    if not scores.empty:
        scores.groupby(["score_period","pseudo_label"]).agg(rows=("seller_id","size"),employees=("seller_id","nunique"),
            discount=("discount","sum"),mean_weight=("pseudo_weight","mean")).reset_index().to_csv(output/"label_coverage.csv",index=False)
        # Prioritization contains signal AND material amount; rank is not a fraud score.
        candidates=scores.loc[scores.model_eligible].copy()
        candidates["priority_amount"]=candidates.discount*candidates.excess_rate.clip(lower=0)
        candidates=candidates.sort_values(["week_start","rank_score","priority_amount"],ascending=[True,False,False])
        top=candidates.groupby("week_start",group_keys=False).head(50)
        top.to_csv(output/"weekly_top50.csv",index=False)
        con.register("scored_keys",scores[KEY])
        trace=output/"account_trace.parquet"
        escaped=str(trace.resolve()).replace("'","''")
        con.execute(f"""COPY (SELECT a.* FROM iterative_accounts a JOIN scored_keys s
            USING(seller_id,location_id,week_start)) TO '{escaped}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
        con.unregister("scored_keys")
        if review_json:
            progress("Contrastando muestra de alertas, desacuerdos y controles con JSON originales")
            review_sample(con,scores,output,cfg)
    summary={"output":str(output.resolve()),"weekly_rows":len(weekly),"scored_rows":len(scores),
             "teacher_folds":result["teacher_status"],"student":students["status"],"glm":glm_status,
             "completed_utc":datetime.now(timezone.utc).isoformat()}
    json_write(output/"complete.json",summary)
    (root/"latest.txt").write_text(str(output.resolve()),encoding="utf-8")
    return output


def read_csv_optional(path):
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError,FileNotFoundError):
        return pd.DataFrame()
