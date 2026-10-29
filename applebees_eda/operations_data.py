"""Operational exploration: distinct accounts, reason provenance and product exposure."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .iterative_data import build_inputs, attach_inputs

VERSION = "operations-v1"
GROUPS = {"correction", "generic_authorization", "promotion_candidate", "employee_benefit_candidate",
          "collection_candidate", "customer_change", "availability_candidate", "pending"}
# These are hypotheses about labels, not verified policies or causes.
LABELS = {
    "SERVERERR": ("correction", "service"), "KITCHNERR": ("correction", "kitchen"),
    "GST CORECT": ("correction", "guest_correction"), "SERV ERR": ("correction", "service"),
    "SERVER ERR": ("correction", "service"), "WRONG ITEM": ("correction", "wrong_item"),
    "MANAGER": ("generic_authorization", "unspecified_manager"),
    "EMP 50": ("employee_benefit_candidate", "employee_50_label"),
    "KIDS FREE": ("promotion_candidate", "kids_free_label"), "FREEAPP": ("promotion_candidate", "free_app_label"),
    "MILTARY 10": ("promotion_candidate", "military_label"), "MARKETING": ("promotion_candidate", "marketing_label"),
    "LRM10%": ("promotion_candidate", "lrm_label"),
    "WALKOUT": ("collection_candidate", "walkout_label"), "NOSHOW": ("collection_candidate", "noshow_label"),
    "CUST CHANG": ("customer_change", "customer_change_label"),
    "86'D ITEM": ("availability_candidate", "availability_label"),
}


@dataclass(frozen=True)
class OperationsConfig:
    seed: int = 42
    sample_accounts: int = 40000
    bootstrap_runs: int = 500
    minimum_product_checks: int = 100
    minimum_location_checks: int = 100
    peer_strength: float = 30.0
    drilldown_products: int = 12
    review_accounts: int = 36

    def __post_init__(self):
        if min(self.sample_accounts,self.minimum_product_checks,self.minimum_location_checks,self.peer_strength)<1:
            raise ValueError("Soportes y muestra deben ser positivos")
        if self.bootstrap_runs<2 or self.review_accounts<6 or self.drilldown_products<1:
            raise ValueError("Remuestreos, revisión o selección inválidos")


def quote(path):
    return str(Path(path).resolve()).replace("'", "''")


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding="utf-8")


def attach_base(con, config, root):
    """Locate a compatible iterative cache; never pick an arbitrary latest run."""
    from . import iterative_data
    fingerprint=hashlib.sha256((config.output_dir/"parquet"/"manifest.json").read_bytes()+
        Path(iterative_data.__file__).read_bytes()+str(config.money_tolerance).encode()).hexdigest()
    for marker in sorted(config.output_dir.parent.glob("eda_iterativo/inputs-*/inputs.json")):
        info=json.loads(marker.read_text(encoding="utf-8"))
        if info.get("fingerprint")==fingerprint and all((marker.parent/(n+".parquet")).exists() for n in ("accounts","contexts","weeks","pairs","motives")):
            attach_inputs(con,marker.parent)
            return info
    return build_inputs(con,config,Path(root)/("base-"+fingerprint[:12]))


def reason_inventory(con):
    con.execute("""CREATE OR REPLACE TEMP VIEW ops_raw_events AS
      SELECT 'discount' kind,d.uid,a.location_id,d.discount_id reason_id,d.discount_name reason_name,
        d.amount,d.manager_id,d.employee_id,d.level,d.line_index,d.event_index,NULL::VARCHAR item_id
      FROM discounts d JOIN iterative_accounts a USING(uid)
      UNION ALL
      SELECT 'void',v.uid,a.location_id,v.category_id,v.category_name,v.amount,v.manager_id,v.employee_id,
        'void',NULL::BIGINT,v.event_index,v.item_id
      FROM voids v JOIN iterative_accounts a USING(uid)""")
    con.execute("""CREATE OR REPLACE TEMP VIEW ops_keyed_events AS SELECT *,
      sha256(to_json([kind,location_id,reason_id,reason_name])) reason_key FROM ops_raw_events""")
    return con.execute("""SELECT reason_key,kind,location_id,reason_id,reason_name,
       count(*) records,count(DISTINCT uid) checks,sum(amount) exported_amount,
       sum(greatest(amount,0)) positive_amount,count(*) FILTER(WHERE amount IS NULL) missing_amounts
       FROM ops_keyed_events GROUP BY ALL ORDER BY kind,location_id,reason_name,reason_id""").fetchdf()


def prepare_dictionary(inventory, root):
    """An editable dictionary survives reruns; each run freezes its own copy."""
    path=Path(root)/"reason_dictionary.csv"
    columns=["reason_key","kind","location_id","reason_id","reason_name","operational_group","subgroup","review_status","interpretation","notes"]
    result=inventory[["reason_key","kind","location_id","reason_id","reason_name"]].copy()
    labels=result.reason_name.fillna("").str.strip().str.upper()
    result["operational_group"]=[LABELS.get(n,("pending","unknown"))[0] for n in labels]
    result["subgroup"]=[LABELS.get(n,("pending","unknown"))[1] for n in labels]
    result["review_status"]=np.where(result.operational_group=="pending","pending","provisional")
    result["interpretation"]="Inferido del nombre; requiere definición operativa"
    result["notes"]=""
    if path.exists():
        prior=pd.read_csv(path,dtype=str,keep_default_na=False)
        if not set(columns).issubset(prior.columns) or prior.reason_key.duplicated().any():
            raise ValueError("Diccionario inválido: faltan columnas o sobran claves duplicadas")
        if not set(prior.operational_group).issubset(GROUPS) or not set(prior.review_status).issubset({"pending","provisional","confirmed"}):
            raise ValueError("Grupo o estado del diccionario desconocido")
        edits=["operational_group","subgroup","review_status","interpretation","notes"]
        result=result.merge(prior[["reason_key"]+edits],on="reason_key",how="left",suffixes=("","_prior"),validate="one_to_one")
        for col in edits:
            result[col]=result[col+"_prior"].where(result[col+"_prior"].notna(),result[col])
        result=result[columns]
    result.to_csv(path,index=False)
    return result,path


def build_operational_inputs(con, config, root, cfg, progress=print):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    base=attach_base(con,config,root)
    progress("Inventario de motivos y diccionario editable")
    inventory=reason_inventory(con)
    dictionary,path=prepare_dictionary(inventory,root)
    fingerprint=hashlib.sha256(base["fingerprint"].encode()+Path(__file__).read_bytes()+path.read_bytes()).hexdigest()
    output=root/("inputs-"+fingerprint[:12]);output.mkdir(exist_ok=True)
    tables=("events","accounts","products")
    marker=output/"complete.json"
    if marker.exists() and all((output/(n+".parquet")).exists() for n in tables):
        for name in tables:
            con.execute(f"CREATE OR REPLACE TEMP VIEW ops_{name} AS SELECT * FROM read_parquet('{quote(output/(name+'.parquet'))}')")
        return output,inventory,dictionary,json.loads(marker.read_text(encoding="utf-8"))
    con.register("ops_dictionary",dictionary)
    con.execute("""CREATE OR REPLACE TEMP TABLE _ops_events AS SELECT e.*,d.operational_group,d.subgroup,d.review_status,
       (m.id IS NOT NULL) manager_matched
       FROM ops_keyed_events e JOIN ops_dictionary d USING(reason_key)
       LEFT JOIN unique_employees m ON e.manager_id=m.id""")
    before=con.execute("SELECT count(*),coalesce(sum(amount),0) FROM ops_raw_events").fetchone()
    after=con.execute("SELECT count(*),coalesce(sum(amount),0) FROM _ops_events").fetchone()
    if before[0]!=after[0] or not np.isclose(before[1],after[1],atol=config.money_tolerance,rtol=0):
        raise AssertionError("El diccionario alteró eventos o importes")
    progress("Construyendo cuentas con indicadores operativos, sin sumar descuentos y anulaciones")
    con.execute("""CREATE OR REPLACE TEMP TABLE _ops_accounts AS WITH e AS (
      SELECT uid,count(*) FILTER(WHERE kind='discount') discount_events,count(*) FILTER(WHERE kind='void') void_events,
        bool_or(operational_group='correction' AND (kind='void' OR amount>0)) has_correction,
        bool_or(kind='discount' AND operational_group='correction' AND amount>0) correction_discount,
        bool_or(kind='void' AND operational_group='correction') correction_void,
        bool_or(kind='discount' AND operational_group='generic_authorization' AND amount>0) generic_discount,
        sum(amount) FILTER(WHERE kind='void') exported_void_amount,
        sum(amount) FILTER(WHERE kind='discount' AND operational_group='correction' AND amount>0) correction_discount_amount,
        sum(amount) FILTER(WHERE kind='discount' AND operational_group='generic_authorization' AND amount>0) generic_discount_amount,
        count(*) FILTER(WHERE kind='discount' AND amount>0 AND NOT manager_matched) unmatched_manager_events
      FROM _ops_events GROUP BY uid), hours AS (
       SELECT uid,extract(hour FROM try_cast(open_time AS TIMESTAMP))::INT open_hour,
        date_diff('second',try_cast(open_utc AS TIMESTAMPTZ),try_cast(close_utc AS TIMESTAMPTZ))/60.0 duration_minutes
       FROM checks)
      SELECT a.* REPLACE(coalesce(a.revenue_center_id,'[missing]') AS revenue_center_id),
       a.revenue_center_id source_revenue_center_id,h.open_hour,h.duration_minutes,
       coalesce(e.discount_events,0) discount_events,coalesce(e.void_events,0) void_events,
       coalesce(e.has_correction,false) has_correction,coalesce(e.correction_discount,false) correction_discount,
       coalesce(e.correction_void,false) correction_void,coalesce(e.generic_discount,false) generic_discount,
       coalesce(e.exported_void_amount,0) exported_void_amount,
       coalesce(e.correction_discount_amount,0) correction_discount_amount,
       coalesce(e.generic_discount_amount,0) generic_discount_amount,
       coalesce(e.unmatched_manager_events,0) unmatched_manager_events
      FROM iterative_accounts a LEFT JOIN e USING(uid) LEFT JOIN hours h USING(uid)""")
    # Reject fan-out even if source checks contain duplicate ids outside the included population.
    n=con.execute("SELECT count(*),count(DISTINCT uid),sum(discount) FROM _ops_accounts").fetchone()
    baseline=con.execute("SELECT count(*),sum(discount) FROM iterative_accounts").fetchone()
    if n[0]!=baseline[0] or n[1]!=n[0] or not np.isclose(n[2],baseline[1],atol=config.money_tolerance,rtol=0):
        raise AssertionError("La unión operativa multiplicó cuentas/importes")
    progress("Reduciendo líneas vendidas a presencia única de producto por cuenta")
    con.execute("""CREATE OR REPLACE TEMP TABLE _ops_products AS WITH present AS (
      SELECT s.uid,s.item_id,count(*) sold_lines,sum(s.quantity) positive_quantity,
        bool_or(s.parent_sequence IS NULL OR s.parent_sequence IN ('0','')) has_unlinked_line,
        bool_or(s.parent_sequence IS NOT NULL AND s.parent_sequence NOT IN ('0','')) has_linked_line,
        bool_or(s.parent_sequence IS NULL) missing_parent,
        sum(s.gross_price) gross_line_amount
      FROM sold_items s JOIN _ops_accounts a USING(uid)
      WHERE s.item_id IS NOT NULL AND s.quantity>0 GROUP BY 1,2), direct AS (
       SELECT DISTINCT e.uid,s.item_id FROM _ops_events e JOIN sold_items s
        ON e.uid=s.uid AND e.line_index=s.line_index
       WHERE e.operational_group='correction' AND e.kind='discount' AND e.level='item' AND e.amount>0
       UNION
       SELECT DISTINCT uid,item_id FROM _ops_events WHERE operational_group='correction' AND kind='void')
      SELECT p.*,d.uid IS NOT NULL directly_linked_correction,
        i.name item_name,i.sales_category_id catalog_category
      FROM present p LEFT JOIN direct d USING(uid,item_id) LEFT JOIN unique_items i ON p.item_id=i.id""")
    for name in tables:
        con.execute(f"COPY _ops_{name} TO '{quote(output/(name+'.parquet'))}' (FORMAT PARQUET,COMPRESSION ZSTD)")
        con.execute(f"CREATE OR REPLACE TEMP VIEW ops_{name} AS SELECT * FROM read_parquet('{quote(output/(name+'.parquet'))}')")
    inventory.to_csv(output/"reason_inventory.csv",index=False)
    dictionary.to_csv(output/"reason_dictionary.csv",index=False)
    info={"version":VERSION,"fingerprint":fingerprint,"base_fingerprint":base["fingerprint"],"accounts":n[0],
          "recorded_discount":n[2],"events":after[0],"dictionary_rows":len(dictionary),
          "mapping_verified":bool((dictionary.review_status=="confirmed").all()),
          "money_convention":"line / records_sum / subtract; alternative retained",
          "void_valuation":"exported face amount, not physical waste or loss"}
    write_json(marker,info)
    con.unregister("ops_dictionary")
    return output,inventory,dictionary,info
