"""Exploratory inference with explicit populations, effects and uncertainty."""
import json

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, fisher_exact
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
import patsy

from .iterative_data import KEY


def adjust_families(frame):
    result = frame.copy()
    result["q_by"] = np.nan
    for _, group in result.groupby("family"):
        valid = group.p_value.notna() & np.isfinite(group.p_value)
        if valid.any():
            result.loc[group.index[valid],"q_by"] = multipletests(group.loc[valid,"p_value"],method="fdr_by")[1]
    return result


def binomial_context_model(con, cfg):
    """Uniform deterministic discovery sample; no inference on all 2M accounts.

    Account-level GLM controls location, center, opening band, weekday, week,
    gross, items, party size, and dominant sales category. Sandwich covariance
    clusters simultaneously by employee and operating date.
    """
    population = con.execute("""SELECT count(*) FROM analysis_accounts
      WHERE business_day BETWEEN ?::DATE AND ?::DATE AND seller_id IS NOT NULL AND eligible""",
      [cfg.discovery_start,cfg.discovery_end]).fetchone()[0]
    sample = con.execute("""SELECT seller_id,location_id,revenue_center_id,time_band,weekday,week_start,
       business_day,sales_category,item_count,traffic_count,gross,has_cash::INT cash,has_void::INT voided,
       has_discount::INT discounted FROM analysis_accounts
       WHERE business_day BETWEEN ?::DATE AND ?::DATE AND seller_id IS NOT NULL AND eligible
       ORDER BY hash(uid,?) LIMIT ?""",[cfg.discovery_start,cfg.discovery_end,cfg.seed,cfg.statistical_sample]).fetchdf()
    status = {"population":"discovery, one seller, reconciled positive accounts", "population_n":population,
              "sample_n":len(sample),"sample_method":"uniform account hash ranking, no outcome stratification",
              "sampling_fraction":len(sample)/max(1,population),"clusters":["seller_id","business_day"],
              "composition":"dominant subcategory + gross + item count + party size",
              "limitation":"Exploratory, sampled inference; category composition is summarized, not full basket"}
    if len(sample)<200 or sample.discounted.nunique()<2 or sample.seller_id.nunique()<10 or sample.business_day.nunique()<10:
        return pd.DataFrame(),dict(status,status="skipped",reason="Soporte insuficiente para regresión y dos agrupamientos")
    # Rare levels are pooled using training data only, reducing unstable columns.
    for col in ("revenue_center_id","sales_category","time_band"):
        counts = sample[col].fillna("unknown").value_counts()
        keep = counts.loc[counts>=max(30,len(sample)*.005)].index
        sample[col] = sample[col].fillna("unknown").where(sample[col].fillna("unknown").isin(keep),"other")
    sample["log_gross"] = np.log1p(sample.gross.clip(lower=0))
    sample["log_items"] = np.log1p(sample.item_count.clip(lower=0))
    sample["log_party"] = np.log1p(sample.traffic_count.fillna(0).clip(lower=0))
    sample["party_missing"] = sample.traffic_count.isna().astype(int)
    formula = ("discounted ~ cash * voided + C(location_id) + C(revenue_center_id) + C(time_band) + "
               "C(weekday) + C(week_start) + C(sales_category) + log_gross + log_items + log_party + party_missing")
    y,x = patsy.dmatrices(formula,sample,return_type="dataframe")
    # Remove exactly constant zero columns, e.g. a fully observed party size.
    x = x.loc[:,(x.nunique()>1)|(x.columns=="Intercept")]
    try:
        with threadpool_limits(limits=1):
            model = sm.GLM(y,x,family=sm.families.Binomial()).fit(maxiter=100)
            covariance,_,_ = cov_cluster_2groups(model,pd.factorize(sample.loc[x.index,"seller_id"])[0],
                                                 pd.factorize(sample.loc[x.index,"business_day"])[0])
        if not model.converged:
            return pd.DataFrame(),dict(status,status="skipped",reason="GLM no convergió")
        rows=[]
        for term in ("cash","voided","cash:voided"):
            if term not in x.columns:
                continue
            i = list(x.columns).index(term)
            variance = covariance[i,i]
            se = np.sqrt(variance) if variance>0 else np.nan
            coefficient = float(model.params.iloc[i])
            rows.append({"family":"adjusted_cash_void","contrast":term,"effect_scale":"odds_ratio",
                         "estimate":np.exp(np.clip(coefficient,-30,30)),
                         "low":np.exp(np.clip(coefficient-1.96*se,-30,30)),"high":np.exp(np.clip(coefficient+1.96*se,-30,30)),
                         "p_value":2*norm.sf(abs(coefficient/se)) if np.isfinite(se) else np.nan,
                         "n":len(x),"covariance_valid":bool(variance>0)})
        # Standardized association: toggle one variable and its interaction,
        # average probabilities over observed covariates, delta-method interval.
        for variable,other in (("cash","voided"),("voided","cash")):
            if variable not in x.columns:
                continue
            a,b = x.copy(),x.copy()
            a[variable],b[variable] = 1.,0.
            if "cash:voided" in x.columns:
                a["cash:voided"],b["cash:voided"] = x[other],0.
            pa,pb = expit(a.to_numpy()@model.params.to_numpy()),expit(b.to_numpy()@model.params.to_numpy())
            effect = float(np.mean(pa-pb))*100
            gradient = ((a.to_numpy()*(pa*(1-pa))[:,None]).mean(axis=0)-
                        (b.to_numpy()*(pb*(1-pb))[:,None]).mean(axis=0))*100
            variance = float(gradient@covariance@gradient)
            se = np.sqrt(variance) if variance>0 else np.nan
            rows.append({"family":"adjusted_cash_void","contrast":variable,"effect_scale":"percentage_points",
                         "estimate":effect,"low":effect-1.96*se,"high":effect+1.96*se,
                         "p_value":2*norm.sf(abs(effect/se)) if np.isfinite(se) else np.nan,
                         "n":len(x),"covariance_valid":bool(variance>0)})
        return adjust_families(pd.DataFrame(rows)),dict(status,status="fit",parameters=len(x.columns),formula=formula,
              covariance_warning="Negative two-way covariance variances yield missing intervals; not silently replaced")
    except (ValueError,np.linalg.LinAlgError) as exc:
        return pd.DataFrame(),dict(status,status="skipped",reason=str(exc))


def intensity_bootstrap(con, cfg):
    frames=[]
    for phase in ("discovery","validation","test"):
        start,end = getattr(cfg,phase+"_start"),getattr(cfg,phase+"_end")
        data = con.execute("""SELECT seller_id,has_cash,count(*) FILTER(WHERE eligible) n,
            sum(discount/gross) FILTER(WHERE eligible) ratio_sum,
            sum((abs(unexplained_reduction)>0.02)::INT) FILTER(WHERE evaluable) unexplained_n,
            count(*) FILTER(WHERE evaluable) reduction_n
            FROM analysis_accounts WHERE seller_id IS NOT NULL
            AND business_day BETWEEN ?::DATE AND ?::DATE GROUP BY ALL""",[start,end]).fetchdf()
        if data.empty:
            continue
        ids=np.sort(data.seller_id.unique())
        rng=np.random.default_rng(cfg.seed)
        for numerator,name in (("ratio_sum","discount_intensity"),("unexplained_n","unexplained_reduction_rate")):
            denominator="reduction_n" if numerator=="unexplained_n" else "n"
            piv=data.pivot(index="seller_id",columns="has_cash",values=[denominator,numerator]).reindex(index=ids).fillna(0)
            if not {False,True}.issubset(set(data.has_cash)):
                continue
            a,b=piv[(numerator,True)].to_numpy(),piv[(numerator,False)].to_numpy()
            na,nb=piv[(denominator,True)].to_numpy(),piv[(denominator,False)].to_numpy()
            boot=[]
            for _ in range(cfg.bootstrap_ci_runs):
                draw=rng.integers(0,len(ids),len(ids))
                if na[draw].sum() and nb[draw].sum():
                    boot.append(100*(a[draw].sum()/na[draw].sum()-b[draw].sum()/nb[draw].sum()))
            estimate=100*(a.sum()/na.sum()-b.sum()/nb.sum())
            low,high=np.quantile(boot,[.025,.975])
            frames.append({"phase":phase,"contrast":"cash_minus_non_cash","outcome":name,
                           "estimate_pp":estimate,"low_pp":low,"high_pp":high,"employees":len(ids),
                           "n_cash":int(na.sum()),"n_non_cash":int(nb.sum()),"replicates":len(boot),
                           "interpretation":"descriptive association; employee-cluster bootstrap, not causal"})
    return pd.DataFrame(frames)


def full_pair_analysis(pairs, cfg):
    """All known pairs, not the baseline's top-5000 CSV. Unknowns stay incomplete.

    Hypergeometric/Fisher comparison uses discount RECORDS and active weekly
    marginals within location; it does not assume every manager worked every shift.
    """
    data=pairs.loc[pairs.manager_matched].copy()
    if data.empty:
        return pd.DataFrame()
    grouping=["location_id","week_start"]
    data["employee_total"]=data.groupby(grouping+["seller_id"]).records.transform("sum")
    data["manager_total"]=data.groupby(grouping+["manager_id"]).records.transform("sum")
    data["total"]=data.groupby(grouping).records.transform("sum")
    data["expected_records"]=data.employee_total*data.manager_total/data.total
    data["lift"]=data.records/data.expected_records
    rows=[]
    for row in data.itertuples():
        a=int(row.records); b=int(row.employee_total-a); c=int(row.manager_total-a)
        d=int(row.total-a-b-c)
        p=fisher_exact([[a,b],[c,d]],alternative="greater").pvalue if min(a,b,c,d)>=0 else np.nan
        rows.append(p)
    data["p_value"]=rows
    data["family"]="pair_concentration"
    data=adjust_families(data)
    data["sufficient_support"]=(data.records>=5)&(data.expected_records>=5)
    # Consecutive calendar persistence, no bridging missing weeks.
    data["week_start"]=pd.to_datetime(data.week_start)
    elevated=data.loc[data.sufficient_support&(data.lift>1)&(data.q_by<.05),
                      ["seller_id","location_id","manager_id","week_start"]].copy()
    elevated["week_start"]+=pd.Timedelta(days=7)
    elevated["previous_elevated"]=True
    data=data.merge(elevated,on=["seller_id","location_id","manager_id","week_start"],how="left",validate="one_to_one")
    data["persistent"]=data.previous_elevated.eq(True)&data.sufficient_support&(data.lift>1)&(data.q_by<.05)
    return data.drop(columns="previous_elevated")


def descriptive_reports(con, weekly):
    distribution=con.execute("""SELECT week_start,location_id,count(*) n,
        count(*) FILTER(WHERE eligible) money_n,avg(has_discount::DOUBLE) discount_rate,
        median(discount/gross) FILTER(WHERE eligible) median_ratio,
        mad(discount/gross) FILTER(WHERE eligible) mad_ratio,
        quantile_cont(discount/gross,0.95) FILTER(WHERE eligible) p95_ratio,
        avg(full_discount::DOUBLE) FILTER(WHERE eligible) full_discount_rate,
        sum(discount) FILTER(WHERE eligible) discount,
        sum(alternative_discount) FILTER(WHERE alternative_eligible) alternative_discount,
        sum(check_discount) FILTER(WHERE eligible) check_discount,
        sum(item_discount) FILTER(WHERE eligible) item_discount,
        avg((abs(residual)<=0.02)::DOUBLE) FILTER(WHERE evaluable) reconciliation,
        avg((abs(alternative_residual)<=0.02)::DOUBLE) FILTER(WHERE evaluable) alternative_reconciliation
        FROM analysis_accounts GROUP BY ALL""").fetchdf()
    # A local synchronized rise suggests a shared operational explanation.
    shifts=weekly.groupby(["location_id","week_start"]).agg(employees=("seller_id","nunique"),
        median_rate=("discount_rate","median"),discount=("discount","sum")).reset_index().sort_values(["location_id","week_start"])
    shifts["rate_change_pp"]=shifts.groupby("location_id").median_rate.diff()*100
    shifts["shared_shift_candidate"]=(shifts.employees>=5)&(shifts.rate_change_pp.abs()>=5)
    loc_tests=con.execute("""SELECT location_id,has_cash,count(*) n,sum(has_discount::INT) discounted
       FROM analysis_accounts WHERE seller_id IS NOT NULL GROUP BY ALL""").fetchdf()
    rows=[]
    for location,group in loc_tests.groupby("location_id"):
        if len(group)!=2:
            continue
        a=group.loc[group.has_cash].iloc[0]; b=group.loc[~group.has_cash].iloc[0]
        effect=100*(a.discounted/a.n-b.discounted/b.n)
        p=fisher_exact([[int(a.discounted),int(a.n-a.discounted)],[int(b.discounted),int(b.n-b.discounted)]]).pvalue
        rows.append({"family":"cash_location_sensitivity","location_id":location,"n":int(a.n+b.n),
                     "estimate_pp":effect,"p_value":p,
                     "limitation":"Unadjusted sensitivity, dependent accounts; see clustered GLM for inference"})
    sensitivity=adjust_families(pd.DataFrame(rows)) if rows else pd.DataFrame()
    return {"weekly_location_distribution":distribution,"shared_shifts":shifts,"location_sensitivity":sensitivity}


def hypothesis_log(glm, intensity, pair_stats, status):
    records=[]
    specifications=[
        ("H1","Contexto explica frecuencia","Cuentas conciliadas de exploración",status.get("sample_n",0),
         "GLM binomial; errores agrupados empleado y fecha","Promociones, turnos y reglas no observados", "context_glm.csv"),
        ("H2","Efectivo, anulaciones e interacción","Misma muestra de H1",status.get("sample_n",0),
         "OR y diferencias ajustadas en puntos porcentuales; BY por familia","Asociación no implica causalidad", "context_glm.csv"),
        ("H3","Intensidad y reducción no documentada","Cuentas positivas conciliadas, por período",None,
         "Percentiles, MAD y bootstrap por empleado","Convención de precios y propinas provisional", "intensity_bootstrap.csv"),
        ("H4","Parejas y motivos persistentes","Todos los registros positivos con gerente conocido",None,
         "Marginales semanales, Fisher y BY; HHI por motivo","No hay turnos ni exposición real del gerente; dependencia de registros", "pair_tests.parquet"),
        ("H5","Robustez a especificaciones","Localización, período y convención alternativa",None,
         "Sensibilidad de importes y tasas; comparación temporal","Agosto ya se exploró y es retrospectivo", "weekly_location_distribution.csv")]
    for hid,hyp,pop,n,method,alternatives,evidence in specifications:
        effects = (glm.to_dict("records") if hid in ("H1","H2") else
                   intensity.to_dict("records") if hid=="H3" else
                   {"pair_weeks":len(pair_stats),"persistent_pair_weeks":int(pair_stats.persistent.sum()) if len(pair_stats) else 0} if hid=="H4" else
                   {"review":"weekly_location_distribution.csv; account and item discount totals; alternative convention"})
        records.append({"hypothesis_id":hid,"hypothesis":hyp,"population":pop,"denominator":n,
                        "method":method,"effect_and_interval_artifact":evidence,"alternative_explanations":alternatives,
                        "effect_and_interval":json.dumps(effects,default=str,ensure_ascii=False),
                        "decision":"reformular" if hid=="H1" and status.get("status")!="fit" else "continuar",
                        "decision_reason":"Revisar efectos, cobertura y sensibilidad; sin confirmación operativa",
                        "reviewer_notes":"","fraud_confirmed":None})
    return pd.DataFrame(records)
