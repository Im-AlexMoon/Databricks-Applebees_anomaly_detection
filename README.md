# EDA de descuentos en Applebee’s

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
Reserva disco para Parquet y archivos temporales. No se ha probado el histórico
completo de 10 GB: el rendimiento depende del tamaño de los checks y del equipo.

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
