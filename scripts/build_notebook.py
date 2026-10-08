"""Rebuild the clean, unexecuted notebook from its Spanish narrative/cells."""
import json
from pathlib import Path
import textwrap


cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip()+"\n"})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": textwrap.dedent(source).strip()+"\n",
        "execution_count": None, "outputs": []})


md('''
# EDA de descuentos en Applebee’s · abril–agosto de 2026

**Objetivo:** comprender los descuentos habituales y generar señales para revisión.
Una anomalía no demuestra fraude; el importe descontado no es pérdida demostrada.
No hay etiquetas de fraude ni políticas de descuento disponibles.

El notebook se ejecuta en tu entorno con tus archivos. Procesa JSON por lotes,
consulta Parquet con DuckDB y lleva únicamente agregados/muestras a pandas.
Los reportes quedan en `outputs/`, excluido de Git. No se incorporan nombres ni
fechas de nacimiento de empleados a las tablas analíticas.

Ejecuta las celdas en orden. Instala primero `requirements.txt` y selecciona ese
entorno como kernel. En Databricks, configura rutas del sistema de archivos
accesibles desde Python (por ejemplo, `/Volumes/...`) y añade el repositorio a `sys.path`.
Esta versión usa DuckDB en el proceso del notebook, sin requerir Spark.
''')
code('''
from pathlib import Path
import sys, os, json

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "applebees_eda").is_dir()), None)
if ROOT is None:
    raise RuntimeError("Abre el notebook desde el repositorio o añade su ruta a sys.path.")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from applebees_eda import Config, ingest, open_database, prepare_facts, quality_report
from applebees_eda.pipeline import monetary_gate
from applebees_eda.analysis import (
    baseline, discount_types, check_sample, seller_contexts, actor_roles,
    employee_manager_pairs, weekly_trends, associations, explore_anomalies,
    review_checks, export_reports,
)

sns.set_theme(style="whitegrid", palette="deep")
pd.set_option("display.max_columns", 25)
''')
md('''
## Configuración

Edita las rutas. Los cuatro catálogos se esperan como arrays `[{...}]`; un catálogo
ausente se reporta y permite continuar el análisis de cuentas. `JobID` es una
lista; `PrimaryJobID` conserva exactamente esa capitalización.

**Convenciones monetarias:** empieza con `unresolved`. Después de revisar la
pregunta 1, selecciona `price_mode` (`line` o `unit`), `discount_mode`
(`records_sum`, `price_difference` o `price_difference_plus_check`) y
`gratuity_mode` (`add`, `subtract` o `ignore`) y vuelve a ejecutar desde la
preparación de hechos. La mejor conciliación es evidencia, no prueba de semántica:
varias hipótesis pueden coincidir cuando Quantity=1 o no hay descuentos.

- `records_sum`: suma de Amount en registros de descuento de cuenta y artículo.
- `price_difference`: diferencia entre GrossPrice y SoldPrice, ajustada según precio.
- `price_difference_plus_check`: esa diferencia más descuentos registrados de cuenta.

Cada opción debe contrastarse con casos con cantidad distinta de 1, descuentos
en ambos niveles y ajustes/propinas. No se calcula un total económico validado
automáticamente al escoger la hipótesis con más coincidencias.
''')
code('''
RAW = Path(os.environ.get("APPLEBEES_RAW", str(ROOT / "data" / "raw")))
config = Config(
    checks=(os.environ.get("APPLEBEES_CHECKS", str(RAW / "checks*.json")),),
    catalogs={
        "revenue_centers": str(RAW / "revenueCenters.json"),
        "locations": str(RAW / "locations.json"),
        "items": str(RAW / "items.json"),
        "employees": str(RAW / "employees.json"),
    },
    output_dir=Path(os.environ.get("APPLEBEES_OUTPUT", str(ROOT / "outputs" / "eda"))),
    start_date="2026-04-01", end_date="2026-08-31",
    batch_checks=500, memory_limit="2GB",
    price_mode=os.environ.get("APPLEBEES_PRICE_MODE", "unresolved"),
    discount_mode=os.environ.get("APPLEBEES_DISCOUNT_MODE", "unresolved"),
    gratuity_mode=os.environ.get("APPLEBEES_GRATUITY_MODE", "unresolved"),
    money_tolerance=0.02, minimum_reconciliation=0.95,
    minimum_peer_checks=30, minimum_peer_employees=3,
    minimum_training_rows=50, minimum_training_sellers=5,
    random_state=42,
)
# Para archivos dentro de subcarpetas:
# config.checks = (str(RAW / "checks" / "**" / "*.json"),)
# Para JSON {"Checks": [...]}: config.json_prefixes = {"checks": "Checks.item"}
# .jsonl/.ndjson se reconocen por extensión; no se convierten a arrays en memoria.
# Si el nombre del archivo de revenueCenters es diferente, cambia su ruta arriba.
print("Período:", config.start_date, "a", config.end_date)
print("Destino:", config.output_dir)
''')
md('''
## Carga reproducible

La memoria de ingestión depende del lote y del tamaño del mayor check, no de los
10 GB completos. Parquet se comprime con Zstandard. DuckDB dispone de una carpeta
temporal para operaciones que excedan su memoria de trabajo; reserva espacio en disco.
El límite de DuckDB no limita pandas ni todo el proceso Python.

Un snapshot existente se reutiliza; no se sobrescribe. Cuando cambien archivos,
catálogos o prefijos JSON, usa otro `output_dir`. Los identificadores de cuenta
duplicados o inválidos se conservan para diagnóstico y se excluyen del EDA.
También se excluyen cuentas con arrays faltantes/nulos: ausencia de datos no
equivale automáticamente a ausencia de descuentos o pagos.
''')
code('''
snapshot = config.output_dir / "parquet"
if not snapshot.exists():
    manifest = ingest(config)
else:
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
print("Registros cargados:", manifest["counts"])
print("Catálogos:", manifest["catalogs"])
print("Arrays faltantes/nulos:", manifest["issues"])
if "con" in globals():
    con.close()
con = open_database(config)
prepare_facts(con, config)

def finish_plot(ax, title, x, y, note=""):
    ax.set(title=title, xlabel=x, ylabel=y)
    ax.figure.text(.01, .01, f"Fuente: JSON configurados · {config.start_date} a {config.end_date}. {note}",
                   fontsize=8, wrap=True)
    ax.figure.tight_layout(rect=(0, .09, 1, 1))
    display(ax.figure)
    plt.close(ax.figure)
''')
md('''
## 1. ¿Los datos permiten medir correctamente los descuentos?

**Métodos:** conteos, cobertura, referencias y conciliación con tolerancia de 0.02
unidades monetarias. La moneda no está identificada en el esquema: no sumar entre
monedas si posteriormente se descubre que las localizaciones utilizan monedas distintas.

Los catálogos con Id duplicados se marcan ambiguos y no aportan atributos al cruce.
`Deleted=true` se conserva; `Borrowed` y diferencias de localización son contexto,
no reglas de fraude. `HomeLocationId=0` necesita definición del proveedor.
La relación de SalesCategoryId con categorías del check se evalúa, no se asume.

Los horarios civiles originales se conservan. Se convierten a UTC únicamente con
zona válida o un offset explícito. Horas ambiguas/inexistentes por horario de verano
se reportan y no se inventa su instante UTC. BusinessDate sigue siendo fecha operativa.
''')
code('''
quality = quality_report(con, config)
for name in ["selection", "catalogs", "references", "check_issues", "time_status", "category_mapping"]:
    print("\\n", name)
    display(quality[name])
display(quality["coverage"].head(10))
coverage = quality["coverage"]
expected_days = pd.date_range(config.start_date, config.end_date)
observed_days = pd.DatetimeIndex(coverage.business_day)
print("Días sin cuentas en los archivos:", len(expected_days.difference(observed_days)))
print("La ausencia de cuentas no demuestra cierre ni inactividad: puede faltar un archivo.")
''')
code('''
display(quality["reconciliation"])
amounts_ready, money_message = monetary_gate(con, config)
print(money_message)
print("Controles de conservación: una fila por cuenta incluida e importes registrados sin cambios después de cruces.")
display(con.execute("""SELECT count(*) included_checks,
    count(*) FILTER (WHERE money_eligible) money_eligible_checks,
    count(*) FILTER (WHERE seller_id IS NULL) checks_without_unique_seller FROM facts""").fetchdf())
''')
md('''
## 2. ¿Cuál es el comportamiento habitual?

Una cuenta tiene descuento observado si existe un registro de descuento con
Amount positivo; registros de importe cero se cuentan por separado. Las métricas
económicas solo se interpretan después de conciliar, sobre cuentas `money_eligible`.
Los importes de registros sin resolver se etiquetan como provisionales.

Revisa medianas, percentiles y distribución; un ranking por importe total también
refleja cuánto vende cada localización. El ejemplo no tiene una columna de total
de cuenta: el total debe reconstruirse y validarse.
''')
code('''
by_location = baseline(con, "location_id")
by_center = baseline(con, "revenue_center_id")
types = discount_types(con)
display(by_location.head(20))
display(by_center)
display(types.head(20))
if not by_location.empty:
    plot = by_location.nlargest(15, "checks").copy()
    plot["rate_pct"] = 100*plot.discount_rate
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=plot, x="location_id", y="rate_pct", color="steelblue", errorbar=None, ax=ax)
    finish_plot(ax, "Cuentas con descuento · 15 localizaciones con mayor volumen", "LocationId", "Cuentas con descuento (%)")
sample = check_sample(con, seed=config.random_state, monetary=amounts_ready)
if amounts_ready and not sample.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(sample.discount_ratio*100, bins=30, ax=ax)
    finish_plot(ax, "Distribución del porcentaje descontado por cuenta", "Descuento / venta bruta (%)", "Cuentas (conteo)",
                f"Muestra determinista de hasta 10,000 cuentas conciliadas; n={len(sample)}.")
    fig, ax = plt.subplots(figsize=(9, 4))
    top_locations = by_location.nlargest(8, "reconciled_checks").location_id
    subset = sample[sample.location_id.isin(top_locations)].copy()
    if not subset.empty:
        subset["ratio_pct"] = subset.discount_ratio*100
        sns.boxplot(data=subset,x="location_id",y="ratio_pct",ax=ax)
        finish_plot(ax,"Porcentaje descontado por cuenta y localización","LocationId","Descuento / venta bruta (%)","Muestra de cuentas conciliadas.")
    else:
        plt.close(fig)
else:
    print("Distribución monetaria pendiente de conciliación. Las tasas de frecuencia sí están disponibles.")
''')
md('''
## 3. ¿Qué empleados se apartan de compañeros comparables?

La unidad es vendedor × localización × centro de ingresos × franja de apertura.
El vendedor se obtiene de ItemsSold.EmployeeId únicamente si todos los artículos
tienen un mismo empleado identificado. Las cuentas de varios vendedores se
reportan aparte; el EmployeeId del pago no se usa como propietario de la venta.

**Métodos:** tasa de cuentas con descuento, intervalos de Wilson al 95%, diferencia
frente a la mediana de los demás empleados y percentil entre pares. Para comparar
se exigen 30 cuentas por empleado/contexto y al menos tres compañeros elegibles.
Son umbrales exploratorios configurables, no reglas de fraude. Wilson supone
ensayos independientes; las cuentas pueden estar correlacionadas por día/turno,
por lo que el intervalo se usa como referencia descriptiva, no como prueba formal.
''')
code('''
sellers = seller_contexts(con, config)
roles = actor_roles(con)
display(sellers.head(25))
display(roles.head(20))
if not sellers.empty:
    print("Contextos con muestra limitada:", int(sellers.limited_sample.sum()))
    print("Señales de tasa elevada respecto de pares:", int(sellers.review_signal.sum()))
    plot = sellers[sellers.peer_median_rate.notna()].head(15).copy()
    if not plot.empty:
        plot["label"] = plot.seller_id+" / local "+plot.location_id+" / "+plot.time_band
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(plot.discount_rate*100, np.arange(len(plot)),
                    xerr=np.maximum(0,np.vstack([(plot.discount_rate-plot.wilson_low)*100,
                                    (plot.wilson_high-plot.discount_rate)*100])), fmt="o", label="Empleado, Wilson 95%")
        ax.scatter(plot.peer_median_rate*100,np.arange(len(plot)),marker="x",label="Mediana de otros empleados")
        ax.set_yticks(np.arange(len(plot)),plot.label)
        ax.legend()
        finish_plot(ax,"Frecuencia de descuentos frente a pares","Cuentas con descuento (%)","Empleado / localización / franja")
print("Aplicadores y gerentes tienen conteos separados. Sin actividad total de autorizaciones no se inventa una tasa de aprobación.")
''')
md('''
## 4. ¿Qué empleados y gerentes concentran descuentos?

Se analizan registros positivos, pares empleado–gerente y cuentas distintas por
localización. Las participaciones se calculan dentro de los descuentos observados;
no representan la probabilidad de que un gerente autorice una petición.
La razón observado/esperado usa los márgenes de registros de la misma localización.
Puede reflejar asignaciones habituales de trabajo; no es evidencia causal de colusión.
''')
code('''
pairs = employee_manager_pairs(con)
display(pairs.head(25))
if not pairs.empty:
    selected_location = pairs.location_id.iloc[0]
    local = pairs[pairs.location_id==selected_location]
    top_employees = local.groupby("employee_id").pair_checks.sum().nlargest(15).index
    top_managers = local.groupby("manager_id").pair_checks.sum().nlargest(10).index
    matrix = local[local.employee_id.isin(top_employees) & local.manager_id.isin(top_managers)].pivot_table(
        index="employee_id", columns="manager_id", values="pair_checks", aggfunc="sum", fill_value=0)
    if not matrix.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(matrix,cmap="Blues",ax=ax,cbar_kws={"label": "Cuentas distintas por pareja (conteo)"})
        finish_plot(ax,f"Descuentos por pareja · localización {selected_location}","ManagerId","EmployeeId",
                    "Selección de empleados/gerentes con mayor volumen; una cuenta puede tener varias parejas.")
''')
md('''
## 5. ¿Hay cambios persistentes entre abril y agosto?

**Métodos:** tasas semanales y mediana móvil de cuatro semanas consecutivas
observadas. Los huecos interrumpen la ventana; no se rellenan como cero ventas.
Las semanas inicial/final pueden ser parciales: revisa observed_business_days y n.
OpenTime y CloseTime dan contexto, no la hora en que se aplicó el descuento.
Cinco meses permiten explorar patrones semanales, no estacionalidad anual.
''')
code('''
trends = weekly_trends(con)
display(trends.head(20))
weekly_locations = baseline(con,"week_start").sort_values("week_start")
if not weekly_locations.empty:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(weekly_locations.week_start,100*weekly_locations.discount_rate,marker="o")
    finish_plot(ax,"Frecuencia semanal de descuentos · todas las cuentas incluidas","Inicio de semana (fecha)","Cuentas con descuento (%)")
hourly = con.execute("""SELECT open_hour,count(*) checks,avg(has_discount::DOUBLE) discount_rate
    FROM facts WHERE open_hour IS NOT NULL GROUP BY 1 ORDER BY 1""").fetchdf()
if not hourly.empty:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hourly.open_hour,100*hourly.discount_rate,marker="o")
    finish_plot(ax,"Descuentos según hora local de apertura de la cuenta","Hora local de apertura (0–23)","Cuentas con descuento (%)",
                "Contexto horario; no hora de aplicación del descuento.")
''')
md('''
## 6. ¿Qué otras características acompañan los descuentos?

**Métodos:** tasas y diferencias de tasas dentro de localización, centro, franja y
día de semana; conteos de cuentas que contienen categorías de productos.
Una categoría se cuenta una vez por cuenta, aunque haya muchos artículos o modificadores.

`has_cash` identifica pagos de importe positivo cuyo PayCatName, al normalizar
espacios/mayúsculas, es CASH. Una cuenta mixta puede tener efectivo. Revisa los
nombres del proveedor antes de interpretar cuentas sin CASH como otro medio.
Los artículos gratuitos y modificadores no se consideran fraude por sí solos.
''')
code('''
related = associations(con)
display(related["cash_void"].head(20))
display(related["payment_rate_differences"].head(20))
display(related["categories"].sort_values("checks",ascending=False).head(20))
display(con.execute("""SELECT category_id,category_name,count(*) payment_records,
    count(*) FILTER (WHERE amount<0) negative_records FROM payments
    GROUP BY 1,2 ORDER BY payment_records DESC LIMIT 30""").fetchdf())
if amounts_ready:
    display(con.execute("""SELECT location_id,count(*) full_discount_checks FROM facts
        WHERE money_eligible AND full_discount GROUP BY 1 ORDER BY 2 DESC""").fetchdf())
''')
md('''
## 7. ¿Existen combinaciones inusuales de señales?

**ML exploratorio:** Isolation Forest, 200 árboles, semilla 42 y sin fijar una
proporción esperada de fraude. Variables por vendedor–localización–semana:
exceso de frecuencia de descuentos, descuento/venta bruta, anulaciones y efectivo.
Las expectativas proceden de medianas de compañeros por contexto, calculadas
solo con abril–julio y excluyendo al empleado evaluado.

Se requieren al menos 50 observaciones de referencia y cinco vendedores; cada
semana debe tener 30 cuentas, 30 conciliadas y al menos 80% de cobertura de
contextos y de importes. Los contextos de agosto sin pares históricos suficientes
se excluyen, en lugar de asignarles una expectativa cero.

El entrenamiento termina el 31 de julio; agosto solo se puntúa. Las semanas que
cruzan julio/agosto se separan por fecha de cada cuenta. La puntuación y su percentil
frente a la referencia no son probabilidades de fraude. Las señales adjuntas
describen variables observadas; no son una explicación causal del modelo.
Sin suficiente histórico o conciliación, esta sección se omite con un motivo.
''')
code('''
model_result = explore_anomalies(con, config)
print(model_result["status"], "·", model_result["message"])
print(model_result["diagnostics"])
if model_result["status"]=="completed":
    display(model_result["candidates"])
    scored = model_result["scored_august"]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(scored.anomaly_score,bins=20,ax=ax)
    finish_plot(ax,"Puntuaciones de anomalía en agosto","Puntuación Isolation Forest (mayor = más inusual)","Vendedor-semanas (conteo)",
                "Referencia abril–julio; sin etiquetas de fraude.")
''')
md('''
## Casos para revisión y exportación

Si el modelo pudo puntuar agosto, se seleccionan cuentas de sus 25 vendedor-semanas
más inusuales con descuentos. Si no, se muestran ejemplos priorizados por señales
observables (descuento y anulación, efectivo y múltiples registros), sin fingir una
clasificación de fraude. La lista se limita a 100 cuentas y conserva la procedencia.

Los CSV incluyen calidad, contextos de vendedores, parejas y casos. Los importes
registrados llevan la etiqueta provisional; los derivados solo se usan con
conciliación. Revisa las exclusiones y cobertura antes de generalizar resultados.
''')
code('''
candidate_weeks = model_result["candidates"] if model_result["status"]=="completed" else None
cases = review_checks(con,config,sellers=candidate_weeks,limit=100)
display(cases.head(25))
report_dir = export_reports(config,quality,sellers,pairs,cases,model_result)
print("Reportes locales:",report_dir)
print("Conclusiones a completar después de ejecutar con todos los datos:")
print("1. Cobertura y principales límites de calidad.")
print("2. Descuentos habituales y diferencias entre contextos comparables.")
print("3. Casos y preguntas concretas para revisión operativa.")
print("4. Definiciones o políticas que faltan para distinguir descuentos legítimos de abuso.")
''')
md('''
## Referencias y límites

- [DuckDB y Parquet](https://duckdb.org/docs/current/data/parquet/overview).
- [Wilson en statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html).
- [Desviación absoluta mediana en SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html).
- [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

Los catálogos son snapshots: Deleted, Borrowed, PrimaryJobID y localización del
empleado pueden no describir su situación en la fecha de una venta. Sin calendario
de turnos, asignaciones, permisos o promociones, quedan factores explicativos sin
observar. No se calculan precisión/recall de fraude ni pérdidas demostradas.
''')


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    target = root/"notebooks"/"eda_descuentos.ipynb"
    target.parent.mkdir(exist_ok=True)
    for n,cell in enumerate(cells):
        cell["id"] = f"eda-{n:03d}"
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print(target)
