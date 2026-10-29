# EDA y pipeline Databricks de Applebee’s

La entrega de la **fase 2** está documentada en
[docs/phase2/README.md](docs/phase2/README.md). Incluye el bundle Databricks,
el job con cuatro tareas, transformaciones Bronze/Silver/Gold, pruebas de
calidad y las instrucciones de carga incremental. El EDA local continúa abajo.

Notebook de Python para explorar abril–agosto de 2026 y priorizar casos para
revisión. Las señales no son etiquetas de fraude y los descuentos registrados
no equivalen a pérdidas demostradas.

## Empezar

Desde la raíz del repositorio, crea un entorno con Python 3.11 o 3.12:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name applebees-eda --display-name "Applebees EDA"
.venv\Scripts\python.exe -m jupyterlab
```

Abre `notebooks/eda_descuentos.ipynb`, selecciona **Applebees EDA** y modifica la
celda **Configuración**. Ejecuta las celdas en orden. La versión del repositorio
no contiene datos ni resultados ejecutados.

Las rutas predeterminadas son:

```text
data/raw/checks*.json
data/raw/revenueCenters.json
data/raw/locations.json
data/raw/items.json
data/raw/employees.json
```

También puedes indicar rutas fuera del repositorio sin copiar los archivos.
Se admiten arrays JSON, NDJSON con extensión `.jsonl`/`.ndjson` y colecciones
dentro de un objeto si configuras `json_prefixes`, por ejemplo
`{"checks": "Checks.item"}`. Para múltiples carpetas, configura un glob recursivo
o varias rutas en `config.checks`. Los catálogos pueden faltar; su ausencia se
reporta y no impide los análisis basados en checks.

Los esquemas respetan los ejemplos: `TimeZoneName`, `ClosedDate`, `Plu`,
`PrimaryJobID` y la lista `JobID`. Se conservan empleados con `Deleted=true`.
`JobID` queda en una tabla auxiliar y no multiplica las cuentas. No se exportan
Name ni DateOfBirth del catálogo de empleados.

## Importes y validación

La configuración inicial mantiene las convenciones monetarias en `unresolved`.
La pregunta 1 compara 18 hipótesis de precio por línea/unidad, representación de
descuentos y signo de propina. Revisa cuentas con Quantity distinto de 1 y
descuentos en ambos niveles antes de escoger:

- `price_mode`: `line` o `unit`.
- `discount_mode`: `records_sum`, `price_difference` o `price_difference_plus_check`.
- `gratuity_mode`: `add`, `subtract` o `ignore`.

Vuelve a ejecutar desde `prepare_facts` después de elegir las convenciones.
Los importes derivados se limitan a cuentas positivas, completas y conciliadas.
El modelo requiere que concilie al menos el 95% de cuentas evaluables, también
en los subconjuntos de cantidades distintas de 1 y cuentas con descuentos.
Las hipótesis pueden ser indistinguibles en ciertos datos: la conciliación no
sustituye las definiciones del proveedor.

Los identificadores de cuenta duplicados se excluyen por completo del EDA y
permanecen en el reporte de calidad. Las cuentas con arrays faltantes/nulos también
se excluyen; no se interpreta la ausencia de datos como ausencia de descuentos o
pagos. Los Id duplicados en catálogos no se eligen arbitrariamente; sus atributos
se consideran ambiguos. El vendedor se identifica
a partir de los artículos vendidos, separado del aplicador, gerente y cobrador.
Las cuentas con varios vendedores se analizan a nivel cuenta y se excluyen de
comparaciones individuales.

Las comparaciones usan localización, centro de ingresos y franja de apertura,
30 cuentas por empleado/contexto y tres compañeros como mínimo. Wilson es una
referencia exploratoria: las cuentas de un mismo turno pueden no ser independientes.
Sin historia de puestos/turnos, los atributos actuales del empleado no permiten
reconstruir su función en cada transacción. Los códigos y nombres de descuento
se conservan por localización; no se asume que un Id tenga el mismo significado
en todos los restaurantes.

## Escala, snapshots y resultados

La ingestión usa `ijson` y lotes de 500 cuentas; su memoria depende del lote más
el mayor check individual. Escribe Parquet comprimido y consulta con DuckDB
(2 GB de memoria de trabajo por defecto, con spill local). pandas recibe
muestras de hasta 10,000 cuentas o agregados de hasta 200,000 filas. Si un
agregado excede ese límite, reduce el período; no se trunca silenciosamente.
Reserva disco para Parquet y archivos temporales. Se verificó una ejecución
completa del histórico en ZIP; el rendimiento depende del tamaño de los checks
y del equipo. La primera conversión es más lenta que reutilizar el snapshot.

`outputs/eda/parquet/` contiene el snapshot y su manifiesto. Reejecutar reutiliza
ese snapshot; cambiar archivos, catálogos o prefijos exige otro `output_dir`.
Se verifican ruta, tamaño y fecha de modificación; no son hashes criptográficos
de contenido. Las fuentes se deben mantener disponibles e inmutables al reutilizar.

`outputs/eda/reports/` contiene CSV de calidad, vendedores, parejas y cuentas
para revisión, además del estado y candidatos del modelo. El notebook incluye
gráficos y notas de interpretación. `data/`, `outputs/`, entornos y cachés quedan
fuera de Git. Los reportes sí contienen identificadores y procedencia para la
revisión local.

Isolation Forest usa abril–julio como referencia y solo puntúa agosto. Ajusta las
variables con pares del histórico, excluyendo al vendedor evaluado. Requiere
50 vendedor-semanas y cinco vendedores de referencia; las semanas necesitan 30
cuentas, 30 conciliadas y al menos 80% de cobertura monetaria y contextual.
Los resultados no incluyen precisión de detección de fraude sin casos confirmados.
Con un solo día o sin convenciones monetarias, el notebook omite el modelo y
continúa con el EDA descriptivo. No utiliza deep learning.

En Databricks, instala `requirements.txt` en el entorno del notebook, trabaja
con rutas accesibles desde Python (por ejemplo `/Volumes/...`) y conserva
`applebees_eda/` disponible desde el repositorio. Esta implementación usa DuckDB
en el proceso del notebook; no distribuye el cálculo con Spark.

## Verificación de desarrollo

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/build_notebook.py
.venv\Scripts\python.exe scripts/validate_notebook.py --checks "C:\ruta\checks.json"
```

Las pruebas usan datos ficticios para verificar conservación de importes,
duplicados, referencias, horarios de verano, valores negativos y separación
temporal. La validación ejecuta el notebook en un kernel real y guarda una copia
local en `outputs/validation/`; no inserta resultados en el notebook versionado.

Para la configuración de Downloads con los JSON diarios dentro de `facts.zip`:

```powershell
.venv\Scripts\python.exe scripts/validate_notebook.py --checks "C:\Users\Alexander_Moon\Downloads\facts.zip::facts/checks/*/checks.json" --raw "C:\Users\Alexander_Moon\Downloads" --output outputs/validation/downloads --eda-output outputs/eda --timeout 3600
```

`--eda-output` permite reutilizar después el mismo snapshot desde el notebook;
`--timeout` especifica segundos por celda. Los archivos ZIP se leen directamente.
Si una celda falla, se guarda una copia con el error en la carpeta de validación.

## EDA iterativo y etiquetas débiles

El segundo notebook es [`notebooks/eda_iterativo.ipynb`](notebooks/eda_iterativo.ipynb).
Reutiliza `outputs/eda/parquet/`; ejecuta sus celdas en orden con el mismo entorno.
El EDA inicial sigue disponible en `eda_descuentos.ipynb`.

1. Verifica rutas a Downloads y al snapshot en la configuración del nuevo notebook.
2. Ejecuta todo: genera tablas semanales, estadística y el experimento temporal.
3. Revisa cobertura, efectos e intervalos antes de las prioridades de los modelos.
4. Anota la decisión en `outputs/eda_iterativo/hypothesis_log.csv`; cambia una
   especificación por iteración y vuelve a ejecutar. Las notas no se sobrescriben.

La unidad es empleado–localización–semana. Se conservan semanas parciales para
descripción. Exploración: 4 mayo–28 junio; validación: 6–26 julio; evaluación
retrospectiva: 3–16 agosto. Las pseudoetiquetas de entrenamiento se generan con
referencias expansivas y bloques posteriores de dos semanas. Julio y agosto usan
el mismo ajuste final de mayo–junio, sin aprendizaje posterior.

Incluye regresión binomial con controles de contexto/composición y errores
agrupados por empleado y fecha; bootstrap por empleado; todas las parejas y
motivos con marginales semanales; corrección BY por familias; suavizado
beta-binomial y comparación monetaria alternativa. La regresión usa una muestra
uniforme reproducible de hasta 40,000 cuentas de exploración, declarada en el
reporte. La composición se resume con categoría dominante, venta, artículos y
comensales; no reconstruye reglas promocionales ni turnos.

GMM compara 1–6 componentes, covarianzas diagonal/completa y tres semillas.
Se contrasta baja densidad con Isolation Forest. Una pseudoetiqueta elevada
requiere exposición/cobertura, ambas colas del 5% y exceso persistente sobre
pares. Los perfiles ambiguos quedan en −1. El peso mide estabilidad al remuestrear
empleados completos, reajustando referencias, escalado y detectores; la
arquitectura GMM elegida queda fija en cada remuestreo. Diez réplicas son una
primera exploración, no una estimación precisa de confianza.

GaussianNB y logística regularizada aprenden solo de etiquetas temporales con
peso positivo. Se exige soporte por clase, empleados y peso efectivo; si no
existe, se registra el motivo y se omite el entrenamiento. Identificadores,
salidas del primer modelo y pseudoetiquetas no entran como predictores.
Concordancia y solapamiento de las 50 prioridades semanales miden imitación y
estabilidad; ninguna puntuación es probabilidad de fraude. Agosto ya fue
explorado: validar prospectivamente exige datos posteriores y evidencia operativa.

Cada iteración conserva manifiesto con versiones/semillas/hashes y artefactos
locales bajo `outputs/eda_iterativo/weak-eda-v1-*`: tabla y puntuaciones semanales
Parquet, bitácora, calidad, estadística, selección GMM, estabilidad, comparación de
modelos y trazabilidad por cuenta. `json_review.csv` contrasta identidad e importe
con filas de las fuentes originales para una muestra de alertas disponibles,
desacuerdos y controles de referencia aleatorios. `fraud_confirmed` queda vacío.
La selección de controles es aleatoria por semana-empleado; la cuenta elegida
dentro de esa semana prioriza una cuenta con descuento para revisar sus motivos.

Para ejecutar desde terminal o verificar el notebook completo:

```powershell
uv run --python .venv\Scripts\python.exe python scripts/run_iterative.py
.venv\Scripts\python.exe scripts/build_iterative_notebook.py
.venv\Scripts\python.exe scripts/validate_notebook.py --notebook notebooks/eda_iterativo.ipynb --checks "C:\Users\Alexander_Moon\Downloads\facts.zip::facts/checks/*/checks.json" --raw "C:\Users\Alexander_Moon\Downloads" --eda-output outputs/eda --output outputs/validation/iterativo --timeout 3600
```

También puedes ejecutar `scripts/run_iterative.py` directamente con Python del
entorno. El generador solo reconstruye el segundo notebook sin resultados; si
editas sus celdas manualmente, incorpora los cambios al generador antes de usarlo.
Para repetir modelos y métricas bajo la convención alternativa, cambia
`MONETARY_MODE` en el notebook o usa
`scripts/run_iterative.py --monetary-mode price_difference_plus_check`. Reutiliza
las cuentas Parquet y crea otra iteración; conserva ambos resultados para comparar.
Las reducciones sin explicación se buscan también entre cuentas evaluables que
no concilian, en `unexplained_reductions.csv`.

Las pruebas sintéticas adicionales cubren separación temporal (cambiar agosto no
modifica etiquetas/pesos anteriores), baja exposición, persistencia sin saltos de
semanas/local, roles, conservación de importes, promociones compartidas y descuentos
individuales. Ejecuta `python -m unittest discover -s tests -v`.

Referencias: [Gaussian Mixture y BIC](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html),
[GaussianNB supervisado](https://scikit-learn.org/stable/modules/naive_bayes.html),
[covarianza con dos agrupamientos](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_cluster_2groups.html)
y [comparaciones múltiples](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).

## Operación comercial: tercera fase

[`notebooks/eda_operacion_comercial.ipynb`](notebooks/eda_operacion_comercial.ipynb)
investiga correcciones operativas y uso de motivos genéricos. Reutiliza el Parquet
existente y conserva los dos notebooks anteriores. Ejecuta las celdas en orden.
Es un EDA explicativo sobre el período disponible; no genera etiquetas de fraude
ni ofrece validación predictiva prospectiva.

Las cuatro hipótesis son: productos con exceso de correcciones respecto a pares,
asociación con horas de mayor actividad, sensibilidad de comparaciones entre locales
a la definición del motivo y perfiles de autorizaciones genéricas.

```powershell
.venv\Scripts\python.exe scripts/run_operations.py
.venv\Scripts\python.exe scripts/build_operations_notebook.py
.venv\Scripts\python.exe scripts/validate_notebook.py --notebook notebooks/eda_operacion_comercial.ipynb --checks "C:\Users\Alexander_Moon\Downloads\facts.zip::facts/checks/*/checks.json" --raw "C:\Users\Alexander_Moon\Downloads" --eda-output outputs/eda --output outputs/validation/operacion-comercial --timeout 3600
```

Con el entorno ya preparado también puedes usar `uv run python scripts/run_operations.py`.
La memoria de trabajo DuckDB es 4 GB por defecto en esta fase; usa
`--memory-limit 2GB` o `APPLEBEES_MEMORY_LIMIT=2GB` en el notebook para reducirla.
La primera reducción de las líneas vendidas puede tardar varios minutos y usar
disco temporal; se conserva para no repetirla en cada iteración.

El diccionario editable está en `outputs/operacion_comercial/reason_dictionary.csv`.
Cada fila conserva tipo de evento, local, código y nombre. Cambia
`operational_group`, `subgroup`, `review_status`, `interpretation` o `notes` según
la evidencia. Los grupos admitidos son `correction`, `generic_authorization`,
`promotion_candidate`, `employee_benefit_candidate`, `collection_candidate`,
`customer_change`, `availability_candidate` y `pending`. Estados: `pending`,
`provisional` y `confirmed`; este último requiere validación operativa externa.
Cambiar el diccionario crea una versión nueva. Las notas de interpretación se
conservan aparte en `analyst_notes.csv` y no se sobrescriben.

Los resultados quedan en `outputs/operacion_comercial/operations-v1-*`:

- `findings.md`, `hypotheses.csv` y `next_priorities.csv`: efectos, límites,
  decisiones y acciones propuestas por pregunta.
- Inventario/diccionario congelado, calidad, importes por grupo y tasas semanales.
- Métricas por producto, mes y local; fichas con vínculos directos, sensibilidad
  de líneas vinculadas y bootstrap por empleado.
- Modelos binomiales ajustados, muestra/población declaradas y covarianza agrupada
  por empleado y fecha. Corrección BY por familia de comparaciones.
- Perfiles de motivos por local, cambios de ranking y sensibilidad monetaria.
- Muestra estratificada de hasta 36 cuentas, controles comparables y contraste
  de identidad/importes con las filas JSON/ZIP originales.

La tasa por producto cuenta una sola presencia del producto en cada cuenta,
aunque se repitan sus líneas. Un descuento a toda la cuenta no se atribuye a todos
sus productos: se distingue coocurrencia de vínculo directo por línea o ItemId.
Los pares de cada producto excluyen ese producto, ajustan por local, mes, centro,
franja y día, y suavizan los contextos pequeños. Wilson es descriptivo; el bootstrap
por empleado de los productos seleccionados usa una referencia fija y sigue
siendo exploratorio tras selección.

Las anulaciones son valores exportados, no costo de desperdicio. No se suman a
descuentos como pérdida. El volumen de aperturas por hora no mide plantilla,
ocupación simultánea ni tiempo de preparación; su asociación no establece causa.
La clasificación por nombres es provisional y el contraste con JSON verifica
coherencia, no política, causa, legitimidad o fraude.
