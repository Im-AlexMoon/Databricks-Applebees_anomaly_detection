"""Temporal teacher/student experiment. Labels describe anomalies, never fraud."""
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .iterative_data import KEY, PeerReference, VERSION

FEATURES = ["excess_rate", "excess_ratio", "excess_cash", "excess_void", "full_rate",
            "unexplained_rate", "motive_hhi", "manager_hhi", "discount_share"]
FORBIDDEN = {"seller_id", "location_id", "week_start", "cluster_id", "p_cluster", "gmm_anomaly",
             "if_anomaly", "pseudo_label", "pseudo_weight", "fraud_confirmed"}
assert not set(FEATURES) & FORBIDDEN


def transformed(frame):
    x = frame[FEATURES].astype(float).copy()
    # Fixed transforms, no data-dependent quantiles or clipping learned on targets.
    for col in ("excess_rate", "excess_ratio", "excess_cash", "excess_void"):
        x[col] = np.arcsinh(10*x[col])
    for col in ("full_rate", "unexplained_rate", "motive_hhi", "manager_hhi", "discount_share"):
        x[col] = logit(x[col].clip(.0001, .9999))
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Variables no finitas; revisar cobertura en vez de imputar silenciosamente")
    return x


class Teacher:
    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.seed = cfg.seed if seed is None else seed

    def fit(self, frame, architecture=None):
        train = frame.loc[frame.model_eligible].copy()
        if len(train)<self.cfg.min_training_rows or train.seller_id.nunique()<self.cfg.min_class_employees:
            raise ValueError("Soporte insuficiente para ajustar detectores")
        self.scaler = StandardScaler().fit(transformed(train))
        x = self.scaler.transform(transformed(train))
        self.selection = []
        candidates = [architecture] if architecture else [(k,cov) for k in range(1,7) for cov in ("diag","full")]
        with threadpool_limits(limits=1):
            for k,cov in candidates:
                runs = []
                for seed in ([self.seed] if architecture else [self.seed,self.seed+1,self.seed+2]):
                    if len(x)<=k:
                        continue
                    model = GaussianMixture(n_components=k,covariance_type=cov,random_state=seed,
                                            n_init=1,reg_covar=1e-4,max_iter=200).fit(x)
                    if model.converged_:
                        runs.append((model,model.bic(x)))
                if not runs:
                    continue
                labels = [m.predict(x) for m,_ in runs]
                stability = [adjusted_rand_score(labels[i],labels[j]) for i in range(len(labels)) for j in range(i)]
                self.selection.append({"components":k,"covariance":cov,"mean_bic":float(np.mean([b for _,b in runs])),
                                       "seed_ari":float(np.mean(stability)) if stability else 1.,"converged_seeds":len(runs)})
                if len(self.selection)==1 or self.selection[-1]["mean_bic"]<self.best_bic:
                    self.best_bic = self.selection[-1]["mean_bic"]
                    self.gmm = min(runs,key=lambda pair:pair[1])[0]
            if not self.selection:
                raise ValueError("Ningún GMM convergió")
            self.forest = IsolationForest(n_estimators=150,random_state=self.seed,n_jobs=1).fit(x)
            self.gmm_train = np.sort(-self.gmm.score_samples(x))
            self.if_train = np.sort(-self.forest.score_samples(x))
        self.thresholds = {"gmm_high":float(np.quantile(self.gmm_train,self.cfg.tail)),
                           "if_high":float(np.quantile(self.if_train,self.cfg.tail)),
                           "gmm_low":float(np.quantile(self.gmm_train,self.cfg.reference_quantile)),
                           "if_low":float(np.quantile(self.if_train,self.cfg.reference_quantile))}
        self.architecture = (self.gmm.n_components,self.gmm.covariance_type)
        return self

    def score(self, frame):
        result = frame.copy()
        for col in ("cluster_id","p_cluster","gmm_anomaly","if_anomaly","gmm_percentile","if_percentile"):
            result[col] = np.nan
        eligible = result.model_eligible
        if eligible.any():
            x = self.scaler.transform(transformed(result.loc[eligible]))
            with threadpool_limits(limits=1):
                p = self.gmm.predict_proba(x)
                result.loc[eligible,"cluster_id"] = p.argmax(axis=1)
                result.loc[eligible,"p_cluster"] = p.max(axis=1)
                result.loc[eligible,"gmm_anomaly"] = -self.gmm.score_samples(x)
                result.loc[eligible,"if_anomaly"] = -self.forest.score_samples(x)
            for name,ref in (("gmm",self.gmm_train),("if",self.if_train)):
                result.loc[eligible,name+"_percentile"] = np.searchsorted(ref,result.loc[eligible,name+"_anomaly"],side="right")/len(ref)
        result["both_high"] = eligible&(result.gmm_anomaly>=self.thresholds["gmm_high"])&(result.if_anomaly>=self.thresholds["if_high"])
        result["both_low"] = eligible&(result.gmm_anomaly<=self.thresholds["gmm_low"])&(result.if_anomaly<=self.thresholds["if_low"])
        result["detector_disagreement"] = eligible&((result.gmm_anomaly>=self.thresholds["gmm_high"]) != (result.if_anomaly>=self.thresholds["if_high"]))
        result["rank_score"] = result[["gmm_percentile","if_percentile"]].min(axis=1)
        return result


def label_scores(scored, previous=None):
    """Persistence needs an actual preceding calendar week, same seller/location.

    Earlier scores may come from a different expanding fold. Nothing is backfilled
    from future weeks. Unknown/unsupported weeks break persistence.
    """
    result = scored.copy()
    current = result[KEY+["excess_evidence","model_eligible"]]
    history = pd.concat([previous[KEY+["excess_evidence","model_eligible"]],current],ignore_index=True) if previous is not None and len(previous) else current.copy()
    if history.duplicated(KEY).any():
        raise AssertionError("La persistencia recibió semanas duplicadas")
    history = history.copy()
    history["week_start"] = pd.to_datetime(history.week_start)+pd.Timedelta(days=7)
    history["prior_excess"] = history.excess_evidence&history.model_eligible
    result = result.merge(history[KEY+["prior_excess"]],on=KEY,how="left",validate="one_to_one")
    result["persistent_excess"] = result.excess_evidence&result.prior_excess.eq(True)&result.model_eligible
    result["pseudo_label"] = -1
    result.loc[result.both_low&~result.persistent_excess,"pseudo_label"] = 0
    result.loc[result.both_high&result.persistent_excess,"pseudo_label"] = 1
    result["pseudo_reason"] = np.select([~result.model_eligible,result.pseudo_label==1,result.pseudo_label==0,result.detector_disagreement],
        ["insufficient_exposure_quality_or_peers","both_tails_and_persistent_peer_excess","both_reference_halves","detector_disagreement"],
        default="insufficient_concordant_evidence")
    result["label_version"] = VERSION
    result["fraud_confirmed"] = None
    return result.drop(columns="prior_excess")


def _slice(contexts, weekly, start, end):
    return (contexts.loc[contexts.week_start.between(start,end)],weekly.loc[weekly.week_start.between(start,end)])


def _stability(cfg, contexts_train, weekly_train, contexts_target, weekly_target, base, teacher, previous):
    """Cluster bootstrap: all weeks/locations of each sampled employee stay together.

    Refit references, scaler, selected GMM architecture and IF in each replicate.
    Weight = fraction retaining the assigned label (unknown base labels get zero).
    """
    rng = np.random.default_rng(cfg.seed)
    employees = np.sort(weekly_train.seller_id.unique())
    agreement = np.zeros(len(base))
    rows = []
    for b in range(cfg.bootstrap_runs):
        draws = rng.choice(employees,size=len(employees),replace=True)
        counts = pd.Series(draws).value_counts()
        ct = contexts_train.loc[contexts_train.index.repeat(contexts_train.seller_id.map(counts).fillna(0).astype(int))]
        reference = PeerReference(cfg).fit(ct)
        reference.cutoff += pd.Timedelta(days=6)  # actual last operating day, not Monday
        training_features = reference.transform(contexts_train,weekly_train)
        replicated = training_features.loc[training_features.index.repeat(training_features.seller_id.map(counts).fillna(0).astype(int))]
        try:
            model = Teacher(cfg,seed=cfg.seed+100+b).fit(replicated,architecture=teacher.architecture)
            target = reference.transform(contexts_target,weekly_target)
            # Re-score preceding weeks under this replicate; use only already observed history.
            prior = previous
            if previous is not None and len(previous):
                cutoff = pd.to_datetime(weekly_target.week_start.min())-pd.Timedelta(days=7)
                pw = weekly_train.loc[weekly_train.week_start==cutoff]
                pc = contexts_train.loc[contexts_train.week_start==cutoff]
                if len(pw):
                    prior = model.score(reference.transform(pc,pw))
            labeled = label_scores(model.score(target),prior)
            agreement += (labeled.pseudo_label.to_numpy()==base.pseudo_label.to_numpy())
            common = base.model_eligible.to_numpy() & labeled.model_eligible.to_numpy()
            ari = adjusted_rand_score(base.loc[common,"cluster_id"],labeled.loc[common,"cluster_id"]) if common.sum()>1 else np.nan
            for week, group in base.groupby("week_start"):
                other = labeled.loc[labeled.week_start==week]
                a = set(map(tuple,group.loc[group.model_eligible].nlargest(50,"rank_score")[KEY].to_numpy()))
                c = set(map(tuple,other.loc[other.model_eligible].nlargest(50,"rank_score")[KEY].to_numpy()))
                rows.append({"replicate":b,"week_start":str(week),"cluster_ari":ari,
                             "top50_overlap":len(a&c)/max(1,min(50,len(a),len(c)))})
        except ValueError as exc:
            rows.append({"replicate":b,"status":"failed","reason":str(exc)})
    base = base.copy()
    base["pseudo_weight"] = agreement/cfg.bootstrap_runs
    base.loc[base.pseudo_label==-1,"pseudo_weight"] = 0.
    return base,rows


def run_teachers(contexts, weekly, cfg, progress=print):
    contexts = contexts.copy()
    contexts["week_start"] = pd.to_datetime(contexts.week_start)
    weekly = weekly.copy()
    weekly["week_start"] = pd.to_datetime(weekly.week_start)
    training = weekly.loc[weekly.phase=="discovery"]
    discovery_weeks = sorted(training.week_start.unique())
    folds = []
    for pos in range(2,len(discovery_weeks),2):
        folds.append(("oof",pd.Timestamp(discovery_weeks[pos-1]),pd.Timestamp(discovery_weeks[pos]),
                      pd.Timestamp(discovery_weeks[min(pos+1,len(discovery_weeks)-1)])))
    for phase in ("validation","test"):
        target = weekly.loc[weekly.phase==phase]
        if len(target) and len(training):
            folds.append((phase,training.week_start.max(),target.week_start.min(),target.week_start.max()))
    all_scores, selections, models, stability = [],[],[],[]
    frozen = None
    for fold_index,(kind,cutoff,start,end) in enumerate(folds):
        progress(f"Detectores {kind}: referencia hasta {(cutoff+pd.Timedelta(days=6)).date()}, semanas {start.date()}–{end.date()}")
        ct,wt = _slice(contexts,training,training.week_start.min(),cutoff)
        # ct must also be restricted to discovery phase (boundary weeks are descriptive).
        ct = ct.loc[ct.week_start.isin(wt.week_start)]
        cx,wx = _slice(contexts,weekly,start,end)
        wx = wx.loc[wx.phase.isin(["discovery"] if kind=="oof" else [kind])]
        cx = cx.loc[cx.week_start.isin(wx.week_start)]
        if wx.empty:
            continue
        try:
            qa = wt.drop_duplicates("week_start")
            for total,matched in (("evaluable_n","reconciled_n"),("quantity_n","quantity_reconciled_n"),("discounted_n","discounted_reconciled_n")):
                if total=="evaluable_n" and total in qa and not qa[total].sum():
                    raise ValueError("Sin cuentas evaluables en entrenamiento")
                if total in qa and qa[total].sum() and qa[matched].sum()/qa[total].sum()<.95:
                    raise ValueError("Convención monetaria no concilia en entrenamiento: "+total)
            if kind in ("validation","test") and frozen is not None:
                reference,teacher,train_features = frozen
            else:
                reference = PeerReference(cfg).fit(ct)
                reference.cutoff += pd.Timedelta(days=6)
                train_features = reference.transform(ct,wt)
                teacher = Teacher(cfg).fit(train_features)
                if kind in ("validation","test"):
                    frozen = (reference,teacher,train_features)
            previous = pd.concat(all_scores,ignore_index=True) if all_scores else None
            # Keep only genuinely preceding rows; no duplicate validation/test histories.
            if previous is not None:
                previous = previous.loc[previous.week_start<start]
            scored = label_scores(teacher.score(reference.transform(cx,wx)),previous)
            scored,bootstrap = _stability(cfg,ct,wt,cx,wx,scored,teacher,previous)
            scored["fold"] = fold_index
            scored["score_period"] = kind
            all_scores.append(scored)
            selections.extend([dict(fold=fold_index,score_period=kind,**row) for row in teacher.selection])
            models.append({"fold":fold_index,"score_period":kind,"status":"fit","reference_end":str(cutoff+pd.Timedelta(days=6)),
                           "score_start":str(start),"score_end":str(end),"architecture":teacher.architecture,
                           "thresholds":teacher.thresholds,"training_rows":int(train_features.model_eligible.sum())})
            stability.extend([dict(fold=fold_index,score_period=kind,**row) for row in bootstrap])
        except ValueError as exc:
            models.append({"fold":fold_index,"score_period":kind,"status":"skipped","reason":str(exc)})
    scores = pd.concat(all_scores,ignore_index=True) if all_scores else pd.DataFrame()
    return {"scores":scores,"gmm_selection":pd.DataFrame(selections),"teacher_status":models,
            "bootstrap_stability":pd.DataFrame(stability)}


def run_students(scores, cfg):
    if scores.empty:
        return {"status":{"status":"skipped","reason":"Sin pseudoetiquetas temporales"},"metrics":pd.DataFrame(),"predictions":pd.DataFrame()}
    train = scores.loc[(scores.score_period=="oof")&(scores.pseudo_label>=0)&(scores.pseudo_weight>0)].copy()
    support = train.groupby("pseudo_label").agg(rows=("seller_id","size"),employees=("seller_id","nunique"),effective_weight=("pseudo_weight","sum"))
    enough = (set(support.index)=={0,1} and (support.rows>=cfg.min_class_rows).all()
              and (support.employees>=cfg.min_class_employees).all() and (support.effective_weight>=cfg.min_class_rows).all())
    if not enough:
        return {"status":{"status":"skipped","reason":"Dos clases con soporte insuficiente; no se fuerza el entrenamiento",
                          "support":support.reset_index().to_dict("records")},"metrics":pd.DataFrame(),"predictions":pd.DataFrame()}
    scaler = StandardScaler().fit(transformed(train))
    x = scaler.transform(transformed(train))
    metrics,predictions = [],[]
    for name,model in (("GaussianNB",GaussianNB(var_smoothing=1e-8)),("LogisticRegression",LogisticRegression(C=1.,max_iter=1000,random_state=cfg.seed))):
        with threadpool_limits(limits=1):
            model.fit(x,train.pseudo_label,sample_weight=train.pseudo_weight)
        for period in ("validation","test"):
            target = scores.loc[(scores.score_period==period)&scores.model_eligible].copy()
            if target.empty:
                continue
            p = model.predict_proba(scaler.transform(transformed(target)))[:,list(model.classes_).index(1)]
            target["student_score"] = p
            target["student_prediction"] = (p>=.5).astype(int)
            target["student"] = name
            labeled = target.loc[target.pseudo_label>=0]
            row = {"model":name,"period":period,"eligible_rows":len(target),"labeled_rows":len(labeled),
                   "weak_label_coverage":len(labeled)/len(target),"interpretation":"imitation_of_weak_labels_not_fraud_accuracy"}
            if len(labeled):
                row["agreement"] = float((labeled.pseudo_label==labeled.student_prediction).mean())
                row["confusion"] = json.dumps(confusion_matrix(labeled.pseudo_label,labeled.student_prediction,labels=[0,1]).tolist())
            if len(labeled) and labeled.pseudo_label.nunique()==2:
                row.update(balanced_agreement=balanced_accuracy_score(labeled.pseudo_label,labeled.student_prediction),
                           weak_f1=f1_score(labeled.pseudo_label,labeled.student_prediction,zero_division=0),
                           confusion=json.dumps(confusion_matrix(labeled.pseudo_label,labeled.student_prediction,labels=[0,1]).tolist()))
            overlaps=[]
            for _,group in target.groupby("week_start"):
                a = set(map(tuple,group.nlargest(50,"rank_score")[KEY].to_numpy()))
                b = set(map(tuple,group.nlargest(50,"student_score")[KEY].to_numpy()))
                overlaps.append(len(a&b)/max(1,min(50,len(a))))
            row["weekly_top50_overlap_teacher"] = float(np.mean(overlaps))
            metrics.append(row)
            predictions.append(target[KEY+["pseudo_label","pseudo_weight","student","student_score","student_prediction"]])
    return {"status":{"status":"fit","support":support.reset_index().to_dict("records"),"features":FEATURES,
                      "probability_warning":"Scores are not calibrated probabilities of fraud"},
            "metrics":pd.DataFrame(metrics),"predictions":pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame()}
