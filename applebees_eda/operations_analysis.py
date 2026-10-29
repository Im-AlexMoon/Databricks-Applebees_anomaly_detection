"""Question-driven operational comparisons. All effects are exploratory associations."""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
import patsy
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits


def by_correction(frame, family="family"):
    out=frame.copy();out["q_by"]=np.nan
    if out.empty or "p_value" not in out:
        return out
    for _,g in out.groupby(family):
        valid=g.p_value.notna()&np.isfinite(g.p_value)
        if valid.any():
            out.loc[g.index[valid],"q_by"]=multipletests(g.loc[valid,"p_value"],method="fdr_by")[1]
    return out


def account_reports(con):
    quality=con.execute("""SELECT count(*) accounts,count(DISTINCT uid) distinct_accounts,
       min(business_day) first_day,max(business_day) last_day,count(DISTINCT business_day) source_days,
       count(*) FILTER(WHERE eligible) eligible_accounts,sum(gross) FILTER(WHERE eligible) gross,
       sum(discount) FILTER(WHERE eligible) discount,sum(gross-discount) FILTER(WHERE eligible) net_sales,
       count(*) FILTER(WHERE seller_id IS NULL) unattributed_accounts,
       count(*) FILTER(WHERE open_hour IS NULL OR open_time_status<>'valid') invalid_open_times,
       count(*) FILTER(WHERE has_correction) correction_accounts,
       count(*) FILTER(WHERE correction_discount AND correction_void) both_correction_forms,
       count(*) FILTER(WHERE has_discount AND has_void) discount_and_void_accounts,
       count(*) FILTER(WHERE discount<0 OR gross<0) negative_money_accounts,
       count(*) FILTER(WHERE gross=0) zero_gross_accounts,
       count(*) FILTER(WHERE duration_minutes<0) reversed_times,
       count(*) FILTER(WHERE traffic_count IS NULL) missing_traffic,
       count(*) FILTER(WHERE traffic_count<0) negative_traffic
       FROM ops_accounts""").fetchdf()
    money=con.execute("""SELECT e.kind,e.operational_group,e.review_status,count(*) records,
       count(DISTINCT e.uid) accounts,sum(e.amount) exported_amount,
       sum(e.amount) FILTER(WHERE e.kind='discount' AND a.eligible) reconciled_discount_amount,
       sum(e.amount) FILTER(WHERE e.amount>0 AND a.eligible) positive_amount_eligible,
       count(*) FILTER(WHERE e.amount IS NULL) missing_amounts,
       count(*) FILTER(WHERE e.amount<0) negative_records,
       count(*) FILTER(WHERE NOT e.manager_matched) unknown_manager_records
       FROM ops_events e JOIN ops_accounts a USING(uid) GROUP BY ALL""").fetchdf()
    weekly=con.execute("""SELECT location_id,week_start,count(*) accounts,
       count(DISTINCT business_day) observed_days,
       count(*) FILTER(WHERE eligible) eligible_accounts,sum(has_correction::INT) corrections,
       sum(correction_discount::INT) discount_corrections,sum(correction_void::INT) void_corrections,
       sum(generic_discount::INT) generic_accounts,
       sum(discount) FILTER(WHERE eligible) discount_amount,
       sum(correction_discount_amount) FILTER(WHERE eligible) correction_discount_amount,
       sum(generic_discount_amount) FILTER(WHERE eligible) generic_discount_amount,
       sum(exported_void_amount) exported_void_amount,
       sum(gross-discount) FILTER(WHERE eligible) net_sales,
       avg(has_correction::DOUBLE) correction_rate,avg(generic_discount::DOUBLE) generic_rate,
       avg((has_correction OR generic_discount)::DOUBLE) broad_rate,
       median(duration_minutes) FILTER(WHERE duration_minutes>=0) median_account_minutes,
       median(traffic_count) median_traffic FROM ops_accounts GROUP BY ALL ORDER BY location_id,week_start""").fetchdf()
    rates=con.execute("""SELECT location_id,revenue_center_id,time_band,weekday,count(*) accounts,
       sum(has_correction::INT) corrections,sum(generic_discount::INT) generic_accounts,
       1000*avg(has_correction::DOUBLE) corrections_per_1000,
       1000*avg(generic_discount::DOUBLE) generic_per_1000 FROM ops_accounts GROUP BY ALL""").fetchdf()
    mapping=con.execute("""SELECT kind,reason_id,count(DISTINCT reason_name) distinct_names,
       count(DISTINCT location_id) locations,count(*) records
       FROM ops_events GROUP BY 1,2 HAVING count(DISTINCT reason_name)>1 ORDER BY records DESC""").fetchdf()
    parents=con.execute("""SELECT count(*) account_product_pairs,count(*) FILTER(WHERE has_linked_line) linked_pairs,
       count(*) FILTER(WHERE missing_parent) missing_parent_pairs,
       count(*) FILTER(WHERE has_unlinked_line) unlinked_pairs,
       count(*) FILTER(WHERE directly_linked_correction) direct_correction_pairs FROM ops_products""").fetchdf()
    return {"quality":quality,"amount_by_reason_group":money,"weekly_locations":weekly,
            "context_rates":rates,"reason_id_name_conflicts":mapping,"product_structure":parents}


def product_comparisons(con,cfg):
    """Account-product co-occurrence; item-level linkage is an independent signal.

    Expected rate excludes the product from each context and local/month prior.
    Many products may share an account. Amounts are never summed over products.
    """
    con.execute("""CREATE OR REPLACE TEMP TABLE ops_account_context AS
       SELECT location_id,date_trunc('month',business_day)::DATE period_month,revenue_center_id,time_band,weekday,
        count(*) n,sum(has_correction::INT) y FROM ops_accounts WHERE eligible GROUP BY ALL""")
    con.execute("""CREATE OR REPLACE TEMP TABLE ops_product_context AS
       SELECT p.item_id,min(p.item_name) item_name,a.location_id,date_trunc('month',a.business_day)::DATE period_month,
        a.revenue_center_id,a.time_band,a.weekday,count(*) n,sum(a.has_correction::INT) y,
        sum(p.directly_linked_correction::INT) direct_y,
        sum(p.has_unlinked_line::INT) unlinked_n,
        sum((p.has_unlinked_line AND a.has_correction)::INT) unlinked_y
       FROM ops_products p JOIN ops_accounts a USING(uid) WHERE a.eligible GROUP BY ALL""")
    strength=float(cfg.peer_strength)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE ops_product_expected AS
      WITH totals AS (SELECT sum(n) n,sum(y) y FROM ops_account_context),
      loc AS (SELECT location_id,period_month,sum(n) n,sum(y) y FROM ops_account_context GROUP BY 1,2),
      product AS (SELECT item_id,sum(n) n,sum(y) y FROM ops_product_context GROUP BY 1),
      product_loc AS (SELECT item_id,location_id,period_month,sum(n) n,sum(y) y FROM ops_product_context GROUP BY 1,2,3),
      refs AS (SELECT p.*,c.n-p.n peer_n,
       coalesce((t.y-g.y)/nullif(t.n-g.n,0),t.y/nullif(t.n,0)) global_rate,
       (l.y-pl.y+{strength}*coalesce((t.y-g.y)/nullif(t.n-g.n,0),t.y/nullif(t.n,0)))/
          (l.n-pl.n+{strength}) local_rate,
       c.y-p.y peer_y
       FROM ops_product_context p JOIN ops_account_context c USING(location_id,period_month,revenue_center_id,time_band,weekday)
       JOIN loc l USING(location_id,period_month) JOIN product g USING(item_id)
       JOIN product_loc pl USING(item_id,location_id,period_month) CROSS JOIN totals t)
      SELECT *, (peer_y+{strength}*local_rate)/(peer_n+{strength}) expected_rate,
       n*CASE WHEN peer_n>=30 THEN 1 ELSE 0 END supported_n FROM refs""")
    result=con.execute("""SELECT item_id,min(item_name) item_name,sum(n) accounts_with_product,sum(y) correction_accounts,
       sum(direct_y) directly_linked_accounts,sum(unlinked_n) unlinked_accounts,sum(unlinked_y) unlinked_corrections,
       sum(n*expected_rate) expected_corrections,sum(supported_n)/sum(n) peer_coverage,
       count(DISTINCT location_id) locations FROM ops_product_expected GROUP BY item_id""").fetchdf()
    if result.empty:
        return result,pd.DataFrame()
    result["corrections_per_1000"]=1000*result.correction_accounts/result.accounts_with_product
    result["expected_per_1000"]=1000*result.expected_corrections/result.accounts_with_product
    result["excess_per_1000"]=result.corrections_per_1000-result.expected_per_1000
    result["direct_per_1000"]=1000*result.directly_linked_accounts/result.accounts_with_product
    result["unlinked_rate_per_1000"]=1000*result.unlinked_corrections/result.unlinked_accounts.replace(0,np.nan)
    lo,hi=proportion_confint(result.correction_accounts,result.accounts_with_product,method="wilson")
    result["rate_low_per_1000"],result["rate_high_per_1000"]=1000*lo,1000*hi
    result["sufficient_support"]=(result.accounts_with_product>=cfg.minimum_product_checks)&(result.peer_coverage>=.8)
    result["interpretation"]="cooccurrence_not_cause; Wilson conditional independence is exploratory"
    temporal=con.execute("""SELECT item_id,period_month,sum(n) accounts,sum(y) corrections,sum(direct_y) direct_corrections,
       1000*sum(y-n*expected_rate)/sum(n) excess_per_1000 FROM ops_product_expected GROUP BY 1,2""").fetchdf()
    return result.sort_values(["sufficient_support","excess_per_1000"],ascending=False),temporal


def product_bootstrap(con,products,cfg):
    selected=products.loc[products.sufficient_support&(products.excess_per_1000>0)].head(cfg.drilldown_products)
    if selected.empty:
        return pd.DataFrame()
    con.register("selected_products",selected[["item_id"]])
    data=con.execute("""SELECT p.item_id,a.seller_id,count(*) n,sum(a.has_correction::INT) y,sum(e.expected_rate) expected
      FROM ops_products p JOIN selected_products s USING(item_id) JOIN ops_accounts a USING(uid)
      JOIN ops_product_expected e ON p.item_id=e.item_id AND a.location_id=e.location_id
        AND date_trunc('month',a.business_day)::DATE=e.period_month AND a.revenue_center_id=e.revenue_center_id
        AND a.time_band=e.time_band AND a.weekday=e.weekday
      WHERE a.eligible AND a.seller_id IS NOT NULL GROUP BY 1,2""").fetchdf()
    con.unregister("selected_products")
    rows=[];rng=np.random.default_rng(cfg.seed)
    for item,g in data.groupby("item_id"):
        n,y,expected=g.n.to_numpy(),g.y.to_numpy(),g.expected.to_numpy()
        if len(g)<5 or n.sum()==0:
            continue
        estimates=[]
        for _ in range(cfg.bootstrap_runs):
            idx=rng.integers(0,len(g),len(g))
            estimates.append(1000*(y[idx].sum()-expected[idx].sum())/n[idx].sum())
        estimate=1000*(y.sum()-expected.sum())/n.sum()
        se=float(np.std(estimates,ddof=1))
        rows.append({"family":"selected_product_excess","item_id":item,"accounts":int(n.sum()),"employees":len(g),
                     "excess_per_1000":estimate,"low_per_1000":np.quantile(estimates,.025),
                     "high_per_1000":np.quantile(estimates,.975),"replicates":cfg.bootstrap_runs,
                     "p_value":2*norm.sf(abs(estimate/se)) if se>0 else np.nan,
                     "limitation":"Selected after screening; fixed peer reference; clustered by employee; exploratory"})
    return by_correction(pd.DataFrame(rows))


def activity_models(con,cfg):
    """Two account-level binomial models, one sampled design, two-way covariance."""
    con.execute("""CREATE OR REPLACE TEMP TABLE ops_hours AS SELECT location_id,business_day,open_hour,
       count(*) hour_accounts FROM ops_accounts WHERE open_hour IS NOT NULL AND open_time_status='valid' GROUP BY ALL""")
    con.execute("""CREATE OR REPLACE TEMP TABLE ops_hour_reference AS SELECT location_id,
       quantile_cont(hour_accounts,.75) high_threshold,median(hour_accounts) median_hour_accounts FROM ops_hours GROUP BY 1""")
    con.execute("""CREATE OR REPLACE TEMP VIEW ops_activity AS SELECT a.*,h.hour_accounts,r.high_threshold,
       (h.hour_accounts>=r.high_threshold) busy_high FROM ops_accounts a
       LEFT JOIN ops_hours h USING(location_id,business_day,open_hour)
       LEFT JOIN ops_hour_reference r USING(location_id)""")
    population=con.execute("SELECT count(*) FROM ops_activity WHERE eligible AND seller_id IS NOT NULL AND hour_accounts IS NOT NULL").fetchone()[0]
    sample=con.execute("""SELECT seller_id,location_id,revenue_center_id,time_band,weekday,week_start,business_day,
       sales_category,item_count,traffic_count,gross,has_cash::INT cash,busy_high::INT busy,
       has_correction::INT correction,generic_discount::INT generic,
       hour_accounts FROM ops_activity WHERE eligible AND seller_id IS NOT NULL AND hour_accounts IS NOT NULL
       ORDER BY hash(uid,?) LIMIT ?""",[cfg.seed,cfg.sample_accounts]).fetchdf()
    status={"sample_n":len(sample),"population_n":population,"method":"uniform account hash sample, explanatory all-period analysis",
            "clusters":["seller_id","business_day"],"busy_definition":"at/above local 75th percentile of observed hour account counts",
            "limitations":"Opening volume is not staffing or simultaneous workload; thresholds contemporaneous, not prediction features"}
    if len(sample)<200 or sample.seller_id.nunique()<10 or sample.business_day.nunique()<10:
        return pd.DataFrame(),dict(status,status="skipped",reason="Soporte insuficiente para dos agrupamientos"),pd.DataFrame()
    for col in ("revenue_center_id","sales_category","time_band"):
        counts=sample[col].fillna("unknown").value_counts()
        keep=counts.loc[counts>=max(30,len(sample)*.005)].index
        sample[col]=sample[col].fillna("unknown").where(sample[col].fillna("unknown").isin(keep),"other")
    sample["log_gross"]=np.log1p(sample.gross.clip(lower=0))
    sample["log_items"]=np.log1p(sample.item_count.clip(lower=0))
    sample["log_traffic"]=np.log1p(sample.traffic_count.fillna(0).clip(lower=0))
    sample["missing_traffic"]=sample.traffic_count.isna().astype(int)
    x=patsy.dmatrix("busy + cash + C(location_id)+C(revenue_center_id)+C(time_band)+C(weekday)+C(week_start)+C(sales_category)+log_gross+log_items+log_traffic+missing_traffic",sample,return_type="dataframe")
    x=x.loc[:,(x.nunique()>1)|(x.columns=="Intercept")]
    rows=[];models=[];expected=[]
    for target in ("correction","generic"):
        if sample[target].nunique()<2:
            models.append({"target":target,"status":"skipped","reason":"Una sola clase"});continue
        try:
            with threadpool_limits(limits=1):
                model=sm.GLM(sample[target],x,family=sm.families.Binomial()).fit(maxiter=100)
                covariance,_,_=cov_cluster_2groups(model,pd.factorize(sample.seller_id)[0],pd.factorize(sample.business_day)[0])
            if not model.converged:
                raise ValueError("GLM no convergió")
            models.append({"target":target,"status":"fit","parameters":len(x.columns)})
            for term in ("busy","cash","log_gross","log_items"):
                if term not in x:
                    continue
                i=list(x.columns).index(term);v=covariance[i,i];se=np.sqrt(v) if v>0 else np.nan
                coef=float(model.params.iloc[i])
                rows.append({"family":target+"_associations","target":target,"contrast":term,"scale":"odds_ratio",
                             "estimate":np.exp(np.clip(coef,-30,30)),"low":np.exp(np.clip(coef-1.96*se,-30,30)),
                             "high":np.exp(np.clip(coef+1.96*se,-30,30)),"p_value":2*norm.sf(abs(coef/se)) if np.isfinite(se) else np.nan,
                             "n":len(sample),"covariance_valid":bool(v>0)})
            if "busy" in x:
                a,b=x.copy(),x.copy();a["busy"],b["busy"]=1.,0.
                pa,pb=expit(a.to_numpy()@model.params.to_numpy()),expit(b.to_numpy()@model.params.to_numpy())
                effect=float(np.mean(pa-pb))*100
                grad=((a.to_numpy()*(pa*(1-pa))[:,None]).mean(axis=0)-(b.to_numpy()*(pb*(1-pb))[:,None]).mean(axis=0))*100
                variance=float(grad@covariance@grad);se=np.sqrt(variance) if variance>0 else np.nan
                rows.append({"family":target+"_associations","target":target,"contrast":"busy","scale":"percentage_points",
                             "estimate":effect,"low":effect-1.96*se,"high":effect+1.96*se,
                             "p_value":2*norm.sf(abs(effect/se)) if np.isfinite(se) else np.nan,"n":len(sample),"covariance_valid":bool(variance>0)})
            pred=sample[["location_id","week_start"]].copy();pred["target"]=target
            pred["observed"]=sample[target];pred["expected"]=model.predict(x)
            expected.append(pred.groupby(["location_id","week_start","target"]).agg(n=("observed","size"),observed=("observed","sum"),expected=("expected","sum")).reset_index())
        except (ValueError,np.linalg.LinAlgError) as exc:
            models.append({"target":target,"status":"skipped","reason":str(exc)})
    return by_correction(pd.DataFrame(rows)),dict(status,status="complete",models=models),pd.concat(expected,ignore_index=True) if expected else pd.DataFrame()


def location_coding(con,cfg):
    locations=con.execute("""SELECT location_id,count(*) accounts,sum(has_correction::INT) correction_accounts,
       sum(generic_discount::INT) generic_accounts,sum((has_correction OR generic_discount)::INT) broad_accounts,
       sum(discount_events) discount_events,
       sum(generic_discount_amount) FILTER(WHERE eligible) generic_amount,
       sum(correction_discount_amount) FILTER(WHERE eligible) correction_amount,
       count(*) FILTER(WHERE eligible) eligible_accounts FROM ops_accounts GROUP BY 1""").fetchdf()
    for num,name in (("correction_accounts","correction_rate"),("generic_accounts","generic_rate"),("broad_accounts","broad_rate")):
        locations[name]=locations[num]/locations.accounts
    locations["specific_rank"]=locations.correction_rate.rank(ascending=False,method="min")
    locations["broad_rank"]=locations.broad_rate.rank(ascending=False,method="min")
    locations["absolute_rank_change"]=(locations.specific_rank-locations.broad_rank).abs()
    locations["sufficient_support"]=locations.accounts>=cfg.minimum_location_checks
    shares=con.execute("""SELECT location_id,operational_group,count(*) records,count(DISTINCT uid) accounts,
       sum(amount) exported_discount_amount FROM ops_events WHERE kind='discount' GROUP BY 1,2""").fetchdf()
    shares["record_share"]=shares.records/shares.groupby("location_id").records.transform("sum")
    return locations,shares


def sensitivity(con,products):
    rows=[]
    for minimum in (30,100,300):
        g=products.loc[(products.accounts_with_product>=minimum)&(products.peer_coverage>=.8)] if len(products) else products
        rows.append({"specification":"minimum_product_accounts","value":minimum,"eligible_products":len(g),
                     "positive_excess_products":int((g.excess_per_1000>0).sum()) if len(g) else 0})
    money=con.execute("""SELECT date_trunc('month',business_day)::DATE period_month,
       count(*) FILTER(WHERE eligible) primary_accounts,count(*) FILTER(WHERE alternative_eligible) alternative_accounts,
       sum(correction_discount_amount) FILTER(WHERE eligible) primary_correction_discount,
       sum(correction_discount_amount) FILTER(WHERE alternative_eligible) alternative_correction_discount,
       sum(discount) FILTER(WHERE eligible) primary_discount,
       sum(alternative_discount) FILTER(WHERE alternative_eligible) alternative_discount
       FROM ops_accounts GROUP BY 1 ORDER BY 1""").fetchdf()
    return pd.DataFrame(rows),money
