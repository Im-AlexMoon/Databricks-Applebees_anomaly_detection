"""Run the operational EDA, preserve human notes, and close exploratory hypotheses."""
from dataclasses import asdict
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd

from .operations_data import VERSION,OperationsConfig,build_operational_inputs,quote,write_json
from .operations_analysis import (account_reports,product_comparisons,product_bootstrap,activity_models,
                                   location_coding,sensitivity)
from .pipeline import records


def select_review(con,products,cfg):
    """Stratified review on six distributed source days; controls match context."""
    days=con.execute("SELECT DISTINCT business_day FROM ops_accounts ORDER BY 1").fetchdf().business_day.tolist()
    if not days:
        return pd.DataFrame()
    chosen=[days[i] for i in np.unique(np.linspace(0,len(days)-1,min(6,len(days)),dtype=int))]
    con.register("review_days",pd.DataFrame({"business_day":chosen}))
    top=products.loc[products.sufficient_support].head(cfg.drilldown_products)[["item_id"]] if len(products) else pd.DataFrame({"item_id":pd.Series(dtype=str)})
    con.register("review_products",top)
    tests={"discount_correction":"a.correction_discount", "void_correction":"a.correction_void",
           "generic_authorization":"a.generic_discount", "discount_void_overlap":"a.has_discount AND a.has_void",
           "product_cooccurrence":"EXISTS (SELECT 1 FROM ops_products p JOIN review_products r USING(item_id) WHERE p.uid=a.uid) AND a.has_correction"}
    selected=[];seen=set();per=max(1,cfg.review_accounts//6)
    for label,condition in tests.items():
        # Per-day ranking limits scanning raw JSON members during subsequent review.
        frame=con.execute(f"""SELECT a.*,row_number() OVER(PARTITION BY a.business_day ORDER BY hash(a.uid,?)) day_rank
          FROM ops_accounts a JOIN review_days d USING(business_day) WHERE ({condition}) AND a.eligible
          QUALIFY day_rank<=? ORDER BY day_rank,business_day""",[cfg.seed,per]).fetchdf()
        count=0
        for row in frame.to_dict("records"):
            if row["uid"] in seen:
                continue
            row.update(sample_group=label,control_of=None);selected.append(row);seen.add(row["uid"]);count+=1
            if count>=per:
                break
    # Random control candidates have no recorded correction/generic reason and
    # match a reviewed case on local, date, center and opening band.
    for case in selected.copy():
        if sum(r["sample_group"]=="matched_control" for r in selected)>=per:
            break
        frame=con.execute("""SELECT * FROM ops_accounts WHERE eligible AND NOT has_correction AND NOT generic_discount
          AND location_id=? AND business_day=? AND revenue_center_id IS NOT DISTINCT FROM ? AND time_band=?
          ORDER BY hash(uid,?) LIMIT 10""",[case["location_id"],case["business_day"],case["revenue_center_id"],case["time_band"],cfg.seed]).fetchdf()
        for row in frame.to_dict("records"):
            if row["uid"] not in seen:
                row.update(sample_group="matched_control",control_of=case["uid"]);selected.append(row);seen.add(row["uid"]);break
    con.unregister("review_days");con.unregister("review_products")
    return pd.DataFrame(selected)


def verify_review(sample,config,output,progress=print):
    rows=sample.to_dict("records")
    for source in sorted({r["source_file"] for r in rows}):
        targets={int(r["source_row"]):r for r in rows if r["source_file"]==source}
        progress("Revisión JSON: "+source.split("::")[-1])
        for number,raw in enumerate(records(source,config.json_prefixes.get("checks")),1):
            if number in targets:
                row=targets[number]
                discounts=list(raw.get("Discounts") or [])
                sold=raw.get("ItemsSold") or []
                for line in sold:
                    discounts.extend(line.get("Discounts") or [])
                voids=raw.get("Voids") or []
                discount=sum(float(e.get("Amount") or 0) for e in discounts)
                void=sum(float(e.get("Amount") or 0) for e in voids)
                identity=(str(raw.get("Id"))==str(row["check_id"]) and str(raw.get("LocationId"))==str(row["location_id"]))
                if not identity or not np.isclose(discount,row["discount"],atol=config.money_tolerance,rtol=0) or not np.isclose(void,row["exported_void_amount"],atol=config.money_tolerance,rtol=0):
                    raise AssertionError("Revisión JSON no coincide con la tabla operativa")
                row.update(identity_verified=True,discount_verified=True,void_verified=True,
                           discount_reasons=json.dumps(sorted({str(e.get("DiscountName")) for e in discounts if e.get("DiscountName")}),ensure_ascii=False),
                           void_reasons=json.dumps(sorted({str(e.get("VoidCategoryName") or e.get("CategoryName") or e.get("Reason")) for e in voids}),ensure_ascii=False),
                           item_lines=len(sold),observed="Identidad, importes y motivos contrastados con fuente",
                           alternative_explanation="Corrección legítima, cambio de cliente o convención de registro; verificar significado del código",
                           missing_information="Política, causa operativa, preparación real y costos no observados",
                           policy_verified=False,fraud_confirmed=None,reviewer_notes="")
            if number>=max(targets):
                break
    if not all(r.get("identity_verified") for r in rows):
        raise AssertionError("No se encontraron todas las filas seleccionadas")
    columns=["uid","sample_group","control_of","check_id","location_id","business_day","seller_id","source_file","source_row",
             "gross","discount","exported_void_amount","has_correction","generic_discount","identity_verified","discount_verified","void_verified",
             "discount_reasons","void_reasons","item_lines","observed","alternative_explanation","missing_information","policy_verified","fraud_confirmed","reviewer_notes"]
    pd.DataFrame(rows).reindex(columns=columns).to_csv(output/"reviewed_accounts.csv",index=False)
    return {"requested":len(sample),"reviewed":len(rows),"all_verified":True,
            "strata":sample.sample_group.value_counts().to_dict() if len(sample) else {},
            "interpretation":"Source consistency verified, not policy compliance, root cause or fraud"}


def close_hypotheses(products,bootstrap,models,model_status,locations,groups,quality,cfg):
    conclusions=[]
    total=int(quality.accounts.iloc[0])
    supported=products.loc[products.sufficient_support] if len(products) else products
    positive=bootstrap.loc[bootstrap.low_per_1000>0] if len(bootstrap) else bootstrap
    conclusions.append({"hypothesis_id":"H1","question":"¿Hay productos con más correcciones de lo esperado por contexto y exposición?",
       "population":"Cuentas conciliadas con cantidad positiva del producto; presencia única cuenta-producto",
       "denominator":int(supported.accounts_with_product.sum()) if len(supported) else 0,
       "denominator_note":"Exposiciones a productos, no cuentas únicas del negocio; no sumar importes sobre productos",
       "method":"Pares sin el producto, suavizado local/mes, bootstrap por empleado en selección exploratoria",
       "effect":f"{len(supported)} productos con soporte; {len(positive)} seleccionados con intervalo de exceso por encima de cero",
       "interval_artifact":"product_bootstrap.csv","alternative_explanations":"Modificadores, productos acompañantes, mezcla no observada y códigos provisionales",
       "decision":"continuar" if len(positive) else "reformular",
       "decision_reason":"Investigar asociación y vínculos directos; no atribuir causalidad al producto por coocurrencia",
       "next_action":"Contrastar productos y líneas directamente vinculadas, persistencia mensual y sensibilidad sin líneas con padre"})
    busy=models.loc[(models.target=="correction")&(models.contrast=="busy")&(models.scale=="percentage_points")] if len(models) else pd.DataFrame()
    if len(busy):
        r=busy.iloc[0];decision="continuar" if r.low>0 else "reformular" if r.high<0 else "descartar"
        effect=f"Diferencia ajustada en puntos porcentuales: {r.estimate:.3f}; IC95% [{r.low:.3f}, {r.high:.3f}]"
        reason="Asociación positiva para profundizar" if r.low>0 else "Los datos no respaldan un aumento general por mayor actividad en esta especificación"
    else:
        decision,effect,reason="reformular","Modelo no estimable",json.dumps(model_status,default=str)
    conclusions.append({"hypothesis_id":"H2","question":"¿Las horas de mayor actividad tienen mayor probabilidad de corrección?",
       "population":"Muestra uniforme de cuentas conciliadas, vendedor único y hora válida","denominator":model_status.get("sample_n",0),
       "denominator_note":"Probabilidad por cuenta, no número bruto de correcciones",
       "method":"GLM binomial con contexto/composición y covarianza agrupada por empleado y fecha; BY por familia",
       "effect":effect,"interval_artifact":"adjusted_models.csv","alternative_explanations":"Personal, capacidad y tiempos de preparación no observados; actividad medida por aperturas",
       "decision":decision,"decision_reason":reason,"next_action":"Comparar asociaciones por local/centro antes de recomendar dotación de personal"})
    largest=float(locations.absolute_rank_change.max()) if len(locations) else 0
    conclusions.append({"hypothesis_id":"H3","question":"¿La comparación entre locales depende de qué motivos contamos como correcciones?",
       "population":"Todas las cuentas incluidas por local","denominator":total,
       "denominator_note":"Superset corrección OR motivo genérico es sensibilidad, no reclasificación confirmada",
       "method":"Perfiles de motivos y cambio de ranking entre definición específica y amplia",
       "effect":f"Mayor desplazamiento del ranking: {largest:.0f} posiciones","interval_artifact":"location_coding.csv",
       "alternative_explanations":"Diferencias reales de operación, mezcla comercial o prácticas de registro",
       "decision":"reformular","decision_reason":"La sensibilidad no identifica por sí sola diferencias de captura; falta validación semántica",
       "next_action":"Revisar diccionario y ejemplos de locales con mayor cambio; no comparar desempeño sin homologar motivos"})
    generic=groups.loc[(groups.kind=="discount")&(groups.operational_group=="generic_authorization")]
    amount=float(generic.reconciled_discount_amount.sum())
    conclusions.append({"hypothesis_id":"H4","question":"¿Qué situaciones terminan registradas bajo autorizaciones genéricas?",
       "population":"Cuentas con motivos provisionales de autorización genérica","denominator":int(locations.generic_accounts.sum()) if len(locations) else 0,
       "denominator_note":"Cuentas distintas por local; importe solo descuentos conciliados",
       "method":"Perfiles de cuenta, motivos coexistentes, asociaciones ajustadas y revisión de fuentes",
       "effect":f"Importe bajo motivos genéricos: {amount:.2f}","interval_artifact":"generic_profiles.csv; adjusted_models.csv",
       "alternative_explanations":"Autorización legítima con motivo insuficiente; códigos no equivalen a incumplimiento",
       "decision":"continuar" if amount>0 else "descartar","decision_reason":"Mejorar explicación y trazabilidad antes de atribuir irregularidad",
       "next_action":"Definir submotivos verificables y reglas de autorización; revisar mezcla de productos/contexto"})
    return pd.DataFrame(conclusions)


def run_operations(con,config,cfg=None,output_root=None,review_json=True,progress=print):
    cfg=cfg or OperationsConfig()
    root=Path(output_root or config.output_dir.parent/"operacion_comercial");root.mkdir(parents=True,exist_ok=True)
    inputs,inventory,dictionary,info=build_operational_inputs(con,config,root,cfg,progress)
    modules=["operations_data.py","operations_analysis.py","operations.py"]
    code_hash=hashlib.sha256(b"".join((Path(__file__).parent/n).read_bytes() for n in modules)).hexdigest()
    fingerprint=hashlib.sha256((info["fingerprint"]+code_hash+json.dumps(asdict(cfg),sort_keys=True)+str(review_json)).encode()).hexdigest()
    output=root/(VERSION+"-"+fingerprint[:12]);output.mkdir(exist_ok=True)
    if (output/"complete.json").exists():
        progress("Reutilizando fase operativa completa: "+str(output));return output
    write_json(output/"manifest.json",{"version":VERSION,"config":asdict(cfg),"input_directory":str(inputs.resolve()),
        "input_fingerprint":info["fingerprint"],"code_sha256":code_hash,"started_utc":datetime.now(timezone.utc).isoformat(),
        "python":platform.python_version(),"packages":{p:version(p) for p in ("duckdb","pandas","numpy","statsmodels","scipy")},
        "scope":"Explanatory EDA over all observed dates; no prospective validation or fraud labels",
        "dictionary_semantics":"Provisional unless individually verified; names are evidence, not policy definitions"})
    dictionary.to_csv(output/"reason_dictionary.csv",index=False)
    inventory.to_csv(output/"reason_inventory.csv",index=False)
    progress("Descripciones, cobertura y comparación de motivos")
    reports=account_reports(con)
    for name,frame in reports.items():
        frame.to_csv(output/(name+".csv"),index=False)
    progress("Comparación de productos con pares sin el producto y denominadores explícitos")
    products,temporal=product_comparisons(con,cfg)
    products.to_parquet(output/"product_metrics.parquet",index=False)
    temporal.to_parquet(output/"product_months.parquet",index=False)
    bootstrap=product_bootstrap(con,products,cfg)
    bootstrap.to_csv(output/"product_bootstrap.csv",index=False)
    progress("Modelos explicativos de correcciones y autorizaciones genéricas")
    models,status,expected=activity_models(con,cfg)
    models.to_csv(output/"adjusted_models.csv",index=False);write_json(output/"model_status.json",status)
    expected.to_csv(output/"observed_expected_sample.csv",index=False)
    activity=con.execute("""SELECT location_id,busy_high,count(*) accounts,sum(has_correction::INT) corrections,
       1000*avg(has_correction::DOUBLE) correction_per_1000 FROM ops_activity
       WHERE eligible AND busy_high IS NOT NULL GROUP BY ALL""").fetchdf()
    activity.to_csv(output/"activity_sensitivity.csv",index=False)
    locations,shares=location_coding(con,cfg)
    locations.to_csv(output/"location_coding.csv",index=False);shares.to_csv(output/"local_reason_shares.csv",index=False)
    con.execute("""SELECT generic_discount,revenue_center_id,time_band,sales_category,count(*) accounts,
      median(gross) FILTER(WHERE eligible) median_gross,median(item_count) median_item_lines,
      avg(has_correction::DOUBLE) correction_rate,avg(full_discount::DOUBLE) FILTER(WHERE eligible) fully_discounted_rate,
      sum(discount) FILTER(WHERE eligible) discount_amount FROM ops_accounts GROUP BY ALL""").fetchdf().to_csv(output/"generic_profiles.csv",index=False)
    support,money=sensitivity(con,products)
    support.to_csv(output/"support_sensitivity.csv",index=False);money.to_csv(output/"money_sensitivity.csv",index=False)
    hypotheses=close_hypotheses(products,bootstrap,models,status,locations,reports["amount_by_reason_group"],reports["quality"],cfg)
    hypotheses.to_csv(output/"hypotheses.csv",index=False)
    # Keep analyst notes separate so changing versions never erases interpretation.
    notes=root/"analyst_notes.csv"
    if not notes.exists():
        pd.DataFrame({"hypothesis_id":["H1","H2","H3","H4"],"reviewer_notes":"","business_validation":"pending"}).to_csv(notes,index=False)
    sample=select_review(con,products,cfg)
    sample.to_parquet(output/"review_selection.parquet",index=False)
    review=verify_review(sample,config,output,progress) if review_json else {"status":"skipped_development_only"}
    write_json(output/"review_status.json",review)
    summary={"output":str(output.resolve()),"inputs":info,"products":len(products),
        "supported_products":int(products.sufficient_support.sum()) if len(products) else 0,
        "hypotheses":hypotheses[["hypothesis_id","decision","effect"]].to_dict("records"),"review":review,
        "completed_utc":datetime.now(timezone.utc).isoformat()}
    write_json(output/"complete.json",summary)
    (root/"latest.txt").write_text(str(output.resolve()),encoding="utf-8")
    return output
