"""Stream JSON into typed Parquet and aggregate each child table before joining.

No names or dates of birth are read into the employee analytical table. Raw
files are never changed. Monetary values are exploratory floats, reconciled
with a configurable tolerance; they are not an accounting ledger.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import glob
import hashlib
import json
import shutil
import uuid

import duckdb
import ijson
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class Config:
    checks: tuple[str, ...] = ("data/raw/checks*.json",)
    catalogs: dict[str, str] = field(default_factory=lambda: {
        "revenue_centers": "data/raw/revenueCenters.json",
        "locations": "data/raw/locations.json",
        "items": "data/raw/items.json",
        "employees": "data/raw/employees.json",
    })
    output_dir: Path = Path("outputs/eda")
    # Arrays use 'item'; a wrapper may use 'Checks.item', for example.
    json_prefixes: dict[str, str] = field(default_factory=dict)
    batch_checks: int = 500
    memory_limit: str = "2GB"
    threads: int = 4
    start_date: str = "2026-04-01"
    end_date: str = "2026-08-31"
    price_mode: str = "unresolved"  # line | unit | unresolved
    discount_mode: str = "unresolved"  # records_sum | price_difference | price_difference_plus_check
    gratuity_mode: str = "unresolved"  # add | subtract | ignore
    money_tolerance: float = 0.02
    minimum_reconciliation: float = 0.95
    minimum_peer_checks: int = 30
    minimum_peer_employees: int = 3
    minimum_training_rows: int = 50
    minimum_training_sellers: int = 5
    random_state: int = 42

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.batch_checks < 1:
            raise ValueError("batch_checks debe ser positivo")
        if self.threads < 1:
            raise ValueError("threads debe ser positivo")
        if self.start_date > self.end_date:
            raise ValueError("Rango de fechas invertido")
        datetime.fromisoformat(self.start_date)
        datetime.fromisoformat(self.end_date)
        choices = {
            "price_mode": {"unresolved", "line", "unit"},
            "discount_mode": {"unresolved", "records_sum", "price_difference", "price_difference_plus_check"},
            "gratuity_mode": {"unresolved", "add", "subtract", "ignore"},
        }
        for key, options in choices.items():
            if getattr(self, key) not in options:
                raise ValueError(f"{key}: opciones válidas {sorted(options)}")
        if self.money_tolerance <= 0 or not 0 <= self.minimum_reconciliation <= 1:
            raise ValueError("Tolerancia o fracción de conciliación inválida")


S, F, I, B = pa.string(), pa.float64(), pa.int64(), pa.bool_()


def schema(*fields):
    return pa.schema(fields)


SCHEMAS = {
    "checks": schema(("uid", S), ("source_file", S), ("source_row", I),
        ("check_id", S), ("location_id", S), ("business_date", S),
        ("revenue_center_id", S), ("open_time", S), ("close_time", S),
        ("open_utc", S), ("close_utc", S), ("open_time_status", S),
        ("close_time_status", S), ("traffic_count", F), ("service_charge", F), ("arrays_complete", B)),
    "sold_items": schema(("uid", S), ("line_index", I), ("sequence", S),
        ("parent_sequence", S), ("item_id", S), ("employee_id", S),
        ("revenue_center_id", S), ("quantity", F), ("gross_price", F),
        ("sold_price", F), ("order_time", S), ("plu", S),
        ("major_category_id", S), ("major_category_name", S),
        ("sub_category_id", S), ("sub_category_name", S), ("pos_source_id", S)),
    "discounts": schema(("uid", S), ("level", S), ("line_index", I),
        ("event_index", I), ("discount_id", S), ("discount_name", S),
        ("amount", F), ("employee_id", S), ("manager_id", S)),
    "payments": schema(("uid", S), ("event_index", I), ("category_id", S),
        ("category_name", S), ("amount", F), ("gratuity", F),
        ("employee_id", S), ("pos_source_id", S), ("is_credit_card", B)),
    "voids": schema(("uid", S), ("event_index", I), ("item_id", S),
        ("employee_id", S), ("manager_id", S), ("quantity", F), ("amount", F),
        ("category_id", S), ("category_name", S)),
    "taxes": schema(("uid", S), ("event_index", I), ("category_id", S), ("amount", F)),
    "revenue_centers": schema(("id", S), ("name", S)),
    "locations": schema(("id", S), ("name", S), ("open_date", S),
        ("closed_date", S), ("time_zone", S)),
    "items": schema(("id", S), ("name", S), ("plu", S), ("sales_category_id", S), ("external_id", S)),
    "employees": schema(("id", S), ("location_id", S), ("location_employee_id", S),
        ("payroll_employee_id", S), ("deleted", B), ("borrowed", B),
        ("home_location_id", S), ("alternate_employee_id", S),
        ("primary_job_id", S), ("hire_date", S)),
    "employee_jobs": schema(("employee_id", S), ("job_id", S)),
}


def _id(value):
    return None if value is None or (isinstance(value,str) and not value.strip()) else str(value)


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Booleano en campo numérico")
    number = float(value)
    import math
    if not math.isfinite(number):
        raise ValueError("Valor numérico no finito")
    return number


def _bool(value):
    if value is not None and not isinstance(value, bool):
        raise ValueError("Se esperaba true/false/null, no texto ni número")
    return value


def _array(obj, key, issues):
    value = obj.get(key)
    if value is None:
        issues[f"missing_or_null_array:{key}"] += 1
        return []
    if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
        raise ValueError(f"{key} debe ser una lista de objetos")
    return value


def records(path: Path, prefix: str | None = None) -> Iterator[dict]:
    """Stream a top-level array, explicit wrapper, or NDJSON (.jsonl/.ndjson)."""
    with path.open("rb") as handle:
        # BOM is valid in some exports, but not accepted by ijson.
        if handle.read(3) != b"\xef\xbb\xbf":
            handle.seek(0)
        if prefix is None:
            if path.suffix.lower() in {".jsonl", ".ndjson"}:
                prefix = ""
            else:
                position = handle.tell()
                first = handle.read(4096).lstrip()[:1]
                handle.seek(position)
                if first != b"[":
                    raise ValueError(f"{path.name}: se esperaba [...]; configura json_prefixes para un wrapper")
                prefix = "item"
        count = 0
        for obj in ijson.items(handle, prefix, multiple_values=(prefix == "")):
            if not isinstance(obj, dict):
                raise ValueError(f"{path.name}: registro {count + 1} no es un objeto")
            count += 1
            yield obj
        if count == 0 and prefix != "item":
            raise ValueError(f"{path.name}: ningún registro en el prefijo {prefix!r}")


def local_time_info(raw, zone):
    """Keep local civil time; refuse to invent a UTC instant at DST transitions."""
    if not raw:
        return None, None, "missing_time"
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return str(raw), None, "invalid_time"
    try:
        tz = ZoneInfo(zone) if zone else None
    except (ZoneInfoNotFoundError, ValueError):
        tz = None
    if dt.tzinfo:
        local = dt.astimezone(tz) if tz else dt
        return local.replace(tzinfo=None).isoformat(), dt.astimezone(timezone.utc).isoformat(), (
            "valid" if tz else "aware_without_zone")
    if tz is None:
        return dt.isoformat(), None, "missing_or_invalid_zone"
    candidates = [dt.replace(tzinfo=tz, fold=fold) for fold in (0, 1)]
    valid = [v for v in candidates if v.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None) == dt]
    if not valid:
        return dt.isoformat(), None, "nonexistent_local_time"
    if len({v.utcoffset() for v in valid}) > 1:
        return dt.isoformat(), None, "ambiguous_local_time"
    return dt.isoformat(), valid[0].astimezone(timezone.utc).isoformat(), "valid"


class _Writer:
    def __init__(self, root):
        self.root = root
        self.buffers = {key: [] for key in SCHEMAS}
        self.parts = Counter()
        self.counts = Counter()

    def add(self, table, row):
        self.buffers[table].append(row)

    def flush(self):
        for table, rows in self.buffers.items():
            if not rows:
                continue
            directory = self.root / table
            directory.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[table]),
                directory / f"part-{self.parts[table]:06d}.parquet", compression="zstd")
            self.parts[table] += 1
            self.counts[table] += len(rows)
            rows.clear()

    def finish(self):
        self.flush()
        for table in SCHEMAS:
            if not self.parts[table]:
                directory = self.root / table
                directory.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist([], schema=SCHEMAS[table]), directory / "part-000000.parquet")


def _catalog_row(name, obj):
    # Deliberately select only needed fields; no employee Name or DateOfBirth.
    maps = {
        "revenue_centers": {"id": "Id", "name": "Name"},
        "locations": {"id": "Id", "name": "Name", "open_date": "OpenDate", "closed_date": "ClosedDate", "time_zone": "TimeZoneName"},
        "items": {"id": "Id", "name": "Name", "plu": "Plu", "sales_category_id": "SalesCategoryId", "external_id": "ExternalId"},
        "employees": {"id": "Id", "location_id": "LocationId", "location_employee_id": "LocationEmployeeId", "payroll_employee_id": "PayrollEmployeeId", "deleted": "Deleted", "borrowed": "Borrowed", "home_location_id": "HomeLocationId", "alternate_employee_id": "AlternateEmployeeId", "primary_job_id": "PrimaryJobID", "hire_date": "HireDate"},
    }
    return {dst: (_bool(obj.get(src)) if dst in {"deleted", "borrowed"} else _id(obj.get(src)))
        for dst, src in maps[name].items()}


def ingest(config: Config) -> dict:
    """Write a fresh Parquet snapshot. Existing snapshots require a new directory.

    Memory is bounded by batch_checks plus the size of a single check and the
    small location lookup. Missing catalogs are supported and reported.
    """
    files = sorted({str(Path(p).resolve()) for pattern in config.checks
        for p in glob.glob(str(Path(pattern).expanduser()), recursive=True) if Path(p).is_file()})
    if not files:
        raise FileNotFoundError("No hay checks para las rutas configuradas")
    target = config.output_dir.resolve() / "parquet"
    if target.exists():
        raise FileExistsError("Ya existe el snapshot. Reutilízalo o configura otro output_dir; no se sobrescribe.")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / (".ingest-" + uuid.uuid4().hex)
    staging.mkdir()
    writer = _Writer(staging)
    issues = Counter()
    location_rows = {}
    catalog_status = {}
    catalog_files = {}
    counts = Counter()
    try:
        for name in ("revenue_centers", "locations", "items", "employees"):
            configured = config.catalogs.get(name)
            path = Path(configured).expanduser() if configured else None
            if path is None or not path.is_file():
                catalog_status[name] = "missing"
                catalog_files[name] = None
                continue
            catalog_status[name] = "loaded"
            catalog_files[name] = {"path": str(path.resolve()), "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns}
            for obj in records(path, config.json_prefixes.get(name)):
                writer.add(name, _catalog_row(name, obj))
                if name == "locations":
                    key = _id(obj.get("Id"))
                    if key in location_rows:
                        location_rows[key] = None  # Ambiguous keys cannot supply a timezone.
                    else:
                        location_rows[key] = obj.get("TimeZoneName")
                if name == "employees":
                    jobs = obj.get("JobID", []) or []
                    if not isinstance(jobs, list):
                        raise ValueError("employees.JobID debe ser una lista")
                    for job in set(_id(j) for j in jobs if j is not None):
                        writer.add("employee_jobs", {"employee_id": _id(obj.get("Id")), "job_id": job})
                counts[name] += 1
                if counts[name] % config.batch_checks == 0:
                    writer.flush()
        writer.flush()
        for filename in files:
            path = Path(filename)
            token = hashlib.sha256(filename.encode()).hexdigest()[:20]
            for row_number, c in enumerate(records(path, config.json_prefixes.get("checks")), 1):
                uid = f"{token}:{row_number}"
                zone = location_rows.get(_id(c.get("LocationId")))
                op, ou, os = local_time_info(c.get("OpenTime"), zone)
                cl, cu, cs = local_time_info(c.get("CloseTime"), zone)
                writer.add("checks", dict(uid=uid, source_file=filename, source_row=row_number,
                    check_id=_id(c.get("Id")), location_id=_id(c.get("LocationId")),
                    business_date=_id(c.get("BusinessDate")), revenue_center_id=_id(c.get("RevenueCenterId")),
                    open_time=op, close_time=cl, open_utc=ou, close_utc=cu,
                    open_time_status=os, close_time_status=cs, traffic_count=_number(c.get("TrafficCount")),
                    service_charge=_number(c.get("ServiceCharge")),
                    arrays_complete=all(isinstance(c.get(k),list) for k in ("ItemsSold","Discounts","Payments","Voids","Tax"))
                        and all(isinstance(i,dict) and isinstance(i.get("Discounts"),list) for i in c["ItemsSold"])))
                def discounts(obj, level, line_index=None):
                    for event_index, d in enumerate(_array(obj, "Discounts", issues)):
                        writer.add("discounts", dict(uid=uid, level=level, line_index=line_index,
                            event_index=event_index, discount_id=_id(d.get("DiscountId")),
                            discount_name=_id(d.get("DiscountName")), amount=_number(d.get("Amount")),
                            employee_id=_id(d.get("EmployeeId")), manager_id=_id(d.get("ManagerId"))))
                discounts(c, "check")
                for index, item in enumerate(_array(c, "ItemsSold", issues)):
                    writer.add("sold_items", dict(uid=uid, line_index=index, sequence=_id(item.get("Sequence")),
                        parent_sequence=_id(item.get("ParentSequence")), item_id=_id(item.get("ItemId")),
                        employee_id=_id(item.get("EmployeeId")), revenue_center_id=_id(item.get("RevenueCenterId")),
                        quantity=_number(item.get("Quantity")), gross_price=_number(item.get("GrossPrice")),
                        sold_price=_number(item.get("SoldPrice")), order_time=_id(item.get("OrderTime")),
                        plu=_id(item.get("ItemPLU")), major_category_id=_id(item.get("ItemMajorCatId")),
                        major_category_name=_id(item.get("ItemMajorCatName")), sub_category_id=_id(item.get("ItemSubCatId")),
                        sub_category_name=_id(item.get("ItemSubCatName")), pos_source_id=_id(item.get("PosSourceId"))))
                    discounts(item, "item", index)
                for index, p in enumerate(_array(c, "Payments", issues)):
                    writer.add("payments", dict(uid=uid, event_index=index, category_id=_id(p.get("PaymentCategoryId")),
                        category_name=_id(p.get("PayCatName")), amount=_number(p.get("Amount")),
                        gratuity=_number(p.get("Gratuity")), employee_id=_id(p.get("EmployeeId")),
                        pos_source_id=_id(p.get("PosSourceId")), is_credit_card=_bool(p.get("IsCreditCard"))))
                for index, v in enumerate(_array(c, "Voids", issues)):
                    writer.add("voids", dict(uid=uid, event_index=index, item_id=_id(v.get("ItemId")),
                        employee_id=_id(v.get("EmployeeId")), manager_id=_id(v.get("ManagerId")),
                        quantity=_number(v.get("Quantity")), amount=_number(v.get("Amount")),
                        category_id=_id(v.get("VoidCategoryId")), category_name=_id(v.get("VoidCategoryName"))))
                for index, t in enumerate(_array(c, "Tax", issues)):
                    writer.add("taxes", dict(uid=uid, event_index=index,
                        category_id=_id(t.get("TaxCategoryId")), amount=_number(t.get("Amount"))))
                counts["checks"] += 1
                if counts["checks"] % config.batch_checks == 0:
                    writer.flush()
        writer.finish()
        manifest = {"version": 2, "files": [{"path": f, "size": Path(f).stat().st_size,
            "mtime_ns": Path(f).stat().st_mtime_ns} for f in files],
            "catalogs": catalog_status, "catalog_files": catalog_files,
            "counts": dict(writer.counts), "issues": dict(issues),
            "json_prefixes": config.json_prefixes}
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.rename(target)
        return manifest
    except Exception:
        # This directory was created by this call and is confined to output_dir.
        shutil.rmtree(staging)
        raise


def open_database(config: Config):
    root = config.output_dir.resolve() / "parquet"
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError("Primero ejecuta ingest(config)")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 2:
        raise ValueError("Snapshot de una versión anterior; usa otro output_dir.")
    current = sorted({str(Path(p).resolve()) for pattern in config.checks
        for p in glob.glob(str(Path(pattern).expanduser()), recursive=True) if Path(p).is_file()})
    signatures = [{"path": f, "size": Path(f).stat().st_size, "mtime_ns": Path(f).stat().st_mtime_ns} for f in current]
    catalog_files = {}
    for name in ("revenue_centers", "locations", "items", "employees"):
        path = Path(config.catalogs[name]).expanduser() if config.catalogs.get(name) else None
        catalog_files[name] = ({"path": str(path.resolve()), "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns} if path and path.is_file() else None)
    if (signatures != manifest["files"] or catalog_files != manifest.get("catalog_files")
            or config.json_prefixes != manifest["json_prefixes"]):
        raise ValueError("Los archivos o prefijos cambiaron desde el snapshot. Usa otro output_dir y vuelve a ingerir.")
    con = duckdb.connect()
    con.execute("SET memory_limit = ?", [config.memory_limit])
    con.execute("SET threads = ?", [config.threads])
    spill = config.output_dir.resolve() / "duckdb_tmp"
    spill.mkdir(exist_ok=True)
    con.execute("SET temp_directory = ?", [str(spill)])
    con.execute("SET TimeZone = 'UTC'")
    for name in SCHEMAS:
        pattern = str(root / name / "*.parquet").replace("'", "''")
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{pattern}')")
    for name in ("locations", "employees", "items", "revenue_centers"):
        con.execute(f"CREATE VIEW unique_{name} AS SELECT * EXCLUDE (key_count) FROM "
            f"(SELECT *, count(*) OVER (PARTITION BY id) key_count FROM {name}) WHERE key_count=1 AND id IS NOT NULL")
    return con


def prepare_facts(con, config: Config):
    """Only unique, valid, in-range checks enter EDA; all raw rows remain for QA."""
    con.execute("""CREATE OR REPLACE TEMP TABLE check_base AS
    WITH source AS (
      SELECT *, try_cast(business_date AS DATE) AS business_day,
             count(*) OVER (PARTITION BY location_id,business_date,check_id) AS key_count
      FROM checks
    ), it AS (
      SELECT uid, count(*) item_count, count(DISTINCT employee_id) seller_count,
        min(employee_id) only_seller, count(*) FILTER (WHERE employee_id IS NULL) missing_seller_items,
        sum(gross_price) gross_line, sum(sold_price) sold_line,
        sum(gross_price*quantity) gross_unit, sum(sold_price*quantity) sold_unit,
        count(*) FILTER (WHERE quantity IS NULL OR gross_price IS NULL OR sold_price IS NULL) missing_item_money,
        count(*) FILTER (WHERE quantity<0 OR gross_price<0 OR sold_price<0) negative_items,
        count(*) FILTER (WHERE quantity<>1) quantities_not_one
      FROM sold_items GROUP BY uid
    ), d AS (
      SELECT uid, count(*) discount_records,
        count(*) FILTER (WHERE amount>0) positive_discounts,
        count(*) FILTER (WHERE amount IS NULL) missing_discount_amounts,
        count(*) FILTER (WHERE amount<0) negative_discounts,
        count(*) FILTER (WHERE level='check') check_discount_records,
        count(*) FILTER (WHERE level='item') item_discount_records,
        sum(amount) recorded_discount,
        coalesce(sum(amount) FILTER (WHERE level='check'),0) check_discount,
        coalesce(sum(amount) FILTER (WHERE level='item'),0) item_discount
      FROM discounts GROUP BY uid
    ), p AS (
      SELECT uid, count(*) payment_records, sum(amount) payment_amount, sum(gratuity) gratuity_amount,
        count(*) FILTER (WHERE amount IS NULL OR gratuity IS NULL) missing_payment_money,
        count(*) FILTER (WHERE amount<0) negative_payments,
        bool_or(upper(trim(category_name))='CASH' AND amount>0) has_cash,
        sum(CASE WHEN upper(trim(category_name))='CASH' AND amount>0 THEN amount ELSE 0 END) cash_amount,
        sum(CASE WHEN amount>0 THEN amount ELSE 0 END) positive_payment_amount
      FROM payments GROUP BY uid
    ), v AS (
      SELECT uid, count(*) void_records, sum(amount) void_amount FROM voids GROUP BY uid
    ), t AS (
      SELECT uid, sum(amount) tax_amount, count(*) FILTER (WHERE amount IS NULL) missing_tax_amounts
      FROM taxes GROUP BY uid
    )
    SELECT s.*, coalesce(it.item_count,0) item_count, coalesce(it.seller_count,0) seller_count,
      CASE WHEN it.seller_count=1 AND it.missing_seller_items=0 THEN it.only_seller END seller_id,
      coalesce(it.missing_seller_items,0) missing_seller_items,
      it.gross_line,it.sold_line,it.gross_unit,it.sold_unit,
      coalesce(it.missing_item_money,0) missing_item_money,
      coalesce(it.negative_items,0) negative_items, coalesce(it.quantities_not_one,0) quantities_not_one,
      coalesce(d.recorded_discount,0) recorded_discount, coalesce(d.check_discount,0) check_discount,
      coalesce(d.item_discount,0) item_discount, coalesce(d.discount_records,0) discount_records,
      coalesce(d.positive_discounts,0)>0 has_discount,
      coalesce(d.missing_discount_amounts,0) missing_discount_amounts,
      coalesce(d.negative_discounts,0) negative_discounts,
      coalesce(d.check_discount_records,0) check_discount_records,
      coalesce(d.item_discount_records,0) item_discount_records,
      coalesce(p.payment_records,0) payment_records, p.payment_amount,
      coalesce(p.gratuity_amount,0) gratuity_amount, coalesce(p.missing_payment_money,0) missing_payment_money,
      coalesce(p.negative_payments,0) negative_payments,
      coalesce(p.has_cash,false) has_cash, coalesce(p.cash_amount,0) cash_amount,
      p.positive_payment_amount,
      coalesce(v.void_records,0) void_records, coalesce(v.void_amount,0) void_amount,
      coalesce(t.tax_amount,0) tax_amount, coalesce(t.missing_tax_amounts,0) missing_tax_amounts,
      CASE WHEN s.check_id IS NULL OR s.location_id IS NULL OR s.business_day IS NULL THEN 'invalid_key'
        WHEN s.key_count>1 THEN 'duplicate_key'
        WHEN NOT s.arrays_complete THEN 'incomplete_arrays'
        WHEN s.business_day NOT BETWEEN ?::DATE AND ?::DATE THEN 'outside_period'
        ELSE 'included' END selection_status
    FROM source s LEFT JOIN it USING(uid) LEFT JOIN d USING(uid) LEFT JOIN p USING(uid)
      LEFT JOIN v USING(uid) LEFT JOIN t USING(uid)
    """, [config.start_date, config.end_date])
    con.execute("""CREATE OR REPLACE TEMP TABLE facts AS
    SELECT b.*, loc.name location_name, loc.time_zone,
      try_cast(loc.open_date AS DATE) location_open_date,
      try_cast(loc.closed_date AS DATE) location_closed_date,
      rc.name revenue_center_name, e.deleted seller_deleted, e.borrowed seller_borrowed,
      e.location_id employee_location_id, e.home_location_id, e.primary_job_id,
      loc.id IS NOT NULL location_matched, rc.id IS NOT NULL revenue_center_matched,
      e.id IS NOT NULL seller_matched,
      date_trunc('week',business_day)::DATE week_start,
      extract(isodow FROM business_day)::INTEGER weekday,
      extract(hour FROM try_cast(open_time AS TIMESTAMP))::INTEGER open_hour,
      CASE WHEN try_cast(open_time AS TIMESTAMP) IS NULL THEN 'unknown'
           WHEN extract(hour FROM try_cast(open_time AS TIMESTAMP))<11 THEN '00-10'
           WHEN extract(hour FROM try_cast(open_time AS TIMESTAMP))<16 THEN '11-15'
           WHEN extract(hour FROM try_cast(open_time AS TIMESTAMP))<21 THEN '16-20' ELSE '21-23' END time_band,
      CASE WHEN try_cast(close_utc AS TIMESTAMPTZ) IS NOT NULL AND try_cast(open_utc AS TIMESTAMPTZ) IS NOT NULL
        THEN date_diff('second',try_cast(open_utc AS TIMESTAMPTZ),try_cast(close_utc AS TIMESTAMPTZ))/60.0 END duration_minutes,
      NULL::DOUBLE gross_sales, NULL::DOUBLE discount_amount, NULL::DOUBLE net_sales,
      NULL::DOUBLE discount_ratio, NULL::DOUBLE reconciliation_residual,
      false money_eligible, false full_discount
    FROM check_base b LEFT JOIN unique_locations loc ON b.location_id=loc.id
      LEFT JOIN unique_revenue_centers rc ON b.revenue_center_id=rc.id
      LEFT JOIN unique_employees e ON b.seller_id=e.id
    WHERE selection_status='included'
    """)
    if "unresolved" not in (config.price_mode, config.discount_mode, config.gratuity_mode):
        gross, sold = f"gross_{config.price_mode}", f"sold_{config.price_mode}"
        discount = {"records_sum": "recorded_discount", "price_difference": f"({gross}-{sold})",
            "price_difference_plus_check": f"({gross}-{sold}+check_discount)"}[config.discount_mode]
        tip = {"add": "+gratuity_amount", "subtract": "-gratuity_amount", "ignore": ""}[config.gratuity_mode]
        con.execute(f"UPDATE facts SET gross_sales={gross}, discount_amount={discount}, net_sales={gross}-({discount}), "
            f"reconciliation_residual=payment_amount-({gross}-({discount})+tax_amount+coalesce(service_charge,0){tip})")
        con.execute("""UPDATE facts SET money_eligible=(
            item_count>0 AND gross_sales>0 AND payment_records>0 AND payment_amount>=0
            AND close_time IS NOT NULL AND missing_item_money=0 AND missing_discount_amounts=0
            AND missing_payment_money=0 AND missing_tax_amounts=0 AND service_charge IS NOT NULL
            AND negative_items=0 AND negative_discounts=0 AND negative_payments=0
            AND discount_amount>=0 AND net_sales>=-? AND abs(reconciliation_residual)<=?),
            discount_ratio=CASE WHEN gross_sales>0 THEN discount_amount/gross_sales END,
            full_discount=(gross_sales>0 AND abs(gross_sales-discount_amount)<=?)
        """, [config.money_tolerance]*3)
        if not monetary_gate(con,config)[0]:
            con.execute("UPDATE facts SET money_eligible=false")
    # Explicit invariants: fan-out is a correctness error, never a warning.
    count = con.execute("SELECT count(*) FROM check_base WHERE selection_status='included'").fetchone()[0]
    actual = con.execute("SELECT count(*),count(DISTINCT uid) FROM facts").fetchone()
    if actual != (count, count):
        raise AssertionError("Un cruce multiplicó cuentas")
    before = con.execute("SELECT coalesce(sum(recorded_discount),0) FROM check_base WHERE selection_status='included'").fetchone()[0]
    after = con.execute("SELECT coalesce(sum(recorded_discount),0) FROM facts").fetchone()[0]
    if abs(before-after) > config.money_tolerance:
        raise AssertionError("El cruce alteró importes registrados")


def _reconciliation_summary(con, config, price, mode, gratuity):
    gross, sold = f"gross_{price}", f"sold_{price}"
    discount = {"records_sum": "recorded_discount", "price_difference": f"({gross}-{sold})",
        "price_difference_plus_check": f"({gross}-{sold}+check_discount)"}[mode]
    tip = {"add": "+gratuity_amount", "subtract": "-gratuity_amount", "ignore": ""}[gratuity]
    q = f"""WITH r AS (SELECT payment_amount-({gross}-({discount})+tax_amount+service_charge{tip}) residual,
       quantities_not_one>0 nonunit, discount_records>0 discounted
       FROM check_base WHERE selection_status='included' AND item_count>0 AND payment_records>0
       AND close_time IS NOT NULL AND missing_item_money=0 AND missing_discount_amounts=0
       AND missing_payment_money=0 AND missing_tax_amounts=0 AND service_charge IS NOT NULL
       AND negative_items=0 AND negative_discounts=0 AND negative_payments=0)
       SELECT count(*) n, avg((abs(residual)<=?)::DOUBLE) match_fraction,
       median(abs(residual)) median_abs_residual, quantile_cont(abs(residual),0.95) p95_abs_residual,
       count(*) FILTER (WHERE nonunit) quantity_subset_n,
       avg((abs(residual)<=?)::DOUBLE) FILTER (WHERE nonunit) quantity_subset_match,
       count(*) FILTER (WHERE discounted) discounted_subset_n,
       avg((abs(residual)<=?)::DOUBLE) FILTER (WHERE discounted) discounted_subset_match FROM r"""
    return con.execute(q,[config.money_tolerance]*3).fetchdf().iloc[0].to_dict()


def reconciliation_candidates(con, config):
    rows = []
    for price in ("line", "unit"):
        for mode in ("records_sum","price_difference","price_difference_plus_check"):
            for gratuity in ("add","subtract","ignore"):
                values = _reconciliation_summary(con,config,price,mode,gratuity)
                rows.append(dict(price_mode=price,discount_mode=mode,gratuity_mode=gratuity,**values))
    import pandas as pd
    return pd.DataFrame(rows).sort_values(["match_fraction", "median_abs_residual"], ascending=[False, True])


def quality_report(con, config):
    """Small aggregated reports; never collect all check rows into pandas."""
    import pandas as pd
    report = {}
    report["selection"] = con.execute("SELECT selection_status,count(*) checks FROM check_base GROUP BY 1").fetchdf()
    report["coverage"] = con.execute("""SELECT business_day,count(*) checks,count(DISTINCT location_id) locations,
        avg(has_discount::DOUBLE) discount_incidence FROM facts GROUP BY 1 ORDER BY 1""").fetchdf()
    catalog_rows = []
    for name in ("locations", "employees", "items", "revenue_centers"):
        row = con.execute(f"SELECT count(*) AS row_count,count(*) FILTER (WHERE id IS NULL) missing_id,"
            f"count(DISTINCT id) distinct_ids FROM {name}").fetchone()
        dup = con.execute(f"SELECT coalesce(sum(n),0) FROM (SELECT count(*) n FROM {name} WHERE id IS NOT NULL GROUP BY id HAVING count(*)>1)").fetchone()[0]
        catalog_rows.append(dict(catalog=name, rows=row[0], missing_id=row[1], distinct_ids=row[2], ambiguous_rows=dup))
    report["catalogs"] = pd.DataFrame(catalog_rows)
    reference_rows = []
    refs = [("checks", "location_id", "locations"), ("checks", "revenue_center_id", "revenue_centers"),
        ("sold_items", "item_id", "items"), ("sold_items", "employee_id", "employees"),
        ("sold_items", "revenue_center_id", "revenue_centers"),
        ("discounts", "employee_id", "employees"), ("discounts", "manager_id", "employees"),
        ("payments", "employee_id", "employees"), ("voids", "employee_id", "employees"),
        ("voids", "manager_id", "employees"), ("voids", "item_id", "items"),
        ("employees", "location_id", "locations"), ("employees", "home_location_id", "locations")]
    for source, column, target in refs:
        row = con.execute(f"SELECT count(*),count(*) FILTER (WHERE a.{column} IS NULL),"
            f"count(*) FILTER (WHERE a.{column} IS NOT NULL AND b.id IS NULL) "
            f"FROM {source} a LEFT JOIN unique_{target} b ON a.{column}=b.id").fetchone()
        reference_rows.append(dict(source=source, field=column, catalog=target,
            records=row[0], missing_id=row[1], unmatched_or_ambiguous=row[2]))
    report["references"] = pd.DataFrame(reference_rows)
    report["check_issues"] = con.execute("""SELECT count(*) checks,
        count(*) FILTER (WHERE seller_count>1) multi_seller,
        count(*) FILTER (WHERE seller_id IS NULL) no_unambiguous_seller,
        count(*) FILTER (WHERE check_discount_records>0 AND item_discount_records>0) both_discount_levels,
        count(*) FILTER (WHERE missing_item_money+missing_discount_amounts+missing_payment_money+missing_tax_amounts>0 OR service_charge IS NULL) missing_money,
        count(*) FILTER (WHERE negative_items+negative_discounts+negative_payments>0) negative_values,
        count(*) FILTER (WHERE gross_line=0) zero_gross,
        count(*) FILTER (WHERE duration_minutes<0) reversed_times,
        count(*) FILTER (WHERE seller_deleted) deleted_seller,
        count(*) FILTER (WHERE seller_borrowed) borrowed_seller,
        count(*) FILTER (WHERE employee_location_id<>location_id) employee_location_differs,
        count(*) FILTER (WHERE home_location_id='0') home_location_zero,
        count(*) FILTER (WHERE business_day<location_open_date OR business_day>location_closed_date) outside_location_dates
        FROM facts""").fetchdf()
    report["time_status"] = con.execute("""SELECT 'open' field,open_time_status status,count(*) checks FROM facts GROUP BY 2
        UNION ALL SELECT 'close',close_time_status,count(*) FROM facts GROUP BY 2""").fetchdf()
    report["category_mapping"] = con.execute("""SELECT count(*) matched_lines,
        count(*) FILTER (WHERE a.sub_category_id=b.sales_category_id) matches_subcategory,
        count(*) FILTER (WHERE a.major_category_id=b.sales_category_id) matches_major_category,
        count(*) FILTER (WHERE a.plu IS DISTINCT FROM b.plu) plu_mismatches
        FROM sold_items a JOIN unique_items b ON a.item_id=b.id""").fetchdf()
    report["reconciliation"] = reconciliation_candidates(con, config)
    return report


def monetary_gate(con, config):
    if "unresolved" in (config.price_mode, config.discount_mode, config.gratuity_mode):
        return False, "Importes derivados pendientes: selecciona las convenciones después de revisar conciliación."
    chosen = _reconciliation_summary(con,config,config.price_mode,config.discount_mode,config.gratuity_mode)
    if chosen["n"] == 0 or chosen["match_fraction"] < config.minimum_reconciliation:
        return False, "La convención seleccionada no alcanza la conciliación mínima; se mantienen análisis de frecuencia."
    # A good overall score can hide a wrong quantity/discount convention.
    for label in ("quantity", "discounted"):
        if chosen[f"{label}_subset_n"] and chosen[f"{label}_subset_match"] < config.minimum_reconciliation:
            return False, f"Conciliación insuficiente en el subconjunto {label}; revisa la semántica."
    return True, "Convenciones seleccionadas y conciliadas; los importes se limitan a cuentas money_eligible."
