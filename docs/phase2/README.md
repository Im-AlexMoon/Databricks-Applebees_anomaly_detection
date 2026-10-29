# Fase 2: control de descuentos y correcciones

Esta entrega convierte la exploración local en un flujo reproducible de Spark,
SQL y Delta en Databricks. Su propósito es explicar descuentos y correcciones,
comparar locales con denominadores explícitos y conservar el vínculo con las
cuentas originales. Los importes descontados y los valores exportados de
anulaciones representan conceptos distintos. Ninguno prueba fraude o pérdida.

## Recorrido del repositorio

- `databricks.yml`: variables y destino del bundle.
- `resources/jobs/daily_orchestration_job.yml`: dependencias, parámetros,
  horario pausado, tiempos máximos y reintento limitado.
- `resources/pipelines/lakeflow_pipeline.yml`: pipeline Lakeflow serverless.
- `src/transformations/bronze/`: ingestión incremental con Auto Loader.
- `src/transformations/silver/`: normalización, roles y conciliación.
- `src/transformations/gold/`: salidas para decisiones y presentación.
- `src/validation/`: reglas compartidas y pruebas sintéticas independientes.
- `src/tasks/`: notebooks de preflight, aceptación y exportación.
- `phase2/` y `scripts/prepare_phase2.py`: transporte local de JSON a NDJSON.gz
  y acceso a las API oficiales. Las transformaciones analíticas ocurren en Spark.
- `notebooks/`: EDA original, EDA iterativo y operación comercial, conservados
  como exploración y referencia de contraste.

Los datos originales, Parquet locales, diccionario provisional con motivos,
credenciales, estado del despliegue y evidencia detallada permanecen fuera de Git.

## Arquitectura y ejecución

El archivo [architecture.mmd](architecture.mmd) documenta el flujo completo.
El job ejecuta cuatro tareas consecutivas:

1. **preflight** verifica tamaño y SHA256 de cada archivo del manifiesto y
   ejecuta pruebas Spark con datos sintéticos temporales.
2. **update_pipeline** actualiza las tablas Bronze, Silver y Gold. Auto Loader
   mantiene el progreso de ingestión bajo el control de Lakeflow.
3. **validate_results** compara los resultados observados con el manifiesto,
   comprueba conservación de filas e importes y contrasta la misma cohorte local.
4. **presentation_outputs** exporta agregados acotados solo después de la
   aceptación. Un fallo anterior impide ejecutar esta tarea.

Una expectativa que falla puede detener un flujo sin revertir todos los demás.
Por eso la aceptación del job constituye la condición de publicación. Consultar
tablas durante una actualización no garantiza una versión aceptada y consistente.

## Instalación y autenticación

Se requiere Python 3.11/3.12, `uv`, Databricks CLI oficial v1 y un workspace
con permisos sobre el catálogo y esquema de destino.

```powershell
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-phase2.txt
databricks auth login --host https://TU-WORKSPACE --profile APPLEBEES_PHASE2
```

El helper busca la CLI en `.cache/tools/databricks/1.22.1/databricks.exe`.
Para una instalación propia, definir `DATABRICKS_CLI_PATH`. El perfil debe estar
en `.cache/databricks/config`, o puede copiarse allí la configuración pública
de host/perfil. Los secretos OAuth los gestiona la CLI en el almacén seguro del
sistema operativo. Nunca subir tokens a Git.

Configurar `DATABRICKS_HOST` si se usa un workspace distinto. El helper acepta
el catálogo `workspace` y el esquema `applebees_phase2` por defecto. Estos nombres
son variables del bundle, no identificadores incrustados en las transformaciones.

## Preparación y despliegue

El diccionario de motivos es un insumo privado versionado por hash. La fase
operativa del EDA lo genera en `outputs/operacion_comercial/reason_dictionary.csv`.
Antes de usar sus grupos como política oficial, un responsable debe revisar su
significado. `review_status` acompaña cada clasificación hasta Gold.

```powershell
.venv/Scripts/python.exe scripts/prepare_phase2.py --workers 4
.venv/Scripts/python.exe scripts/phase2_cloud.py upload RUTA/AL/manifest.json
.venv/Scripts/python.exe scripts/phase2_cloud.py validate
.venv/Scripts/python.exe scripts/phase2_cloud.py deploy
.venv/Scripts/python.exe scripts/phase2_cloud.py run
.venv/Scripts/python.exe scripts/phase2_cloud.py status ID_DE_EJECUCION
.venv/Scripts/python.exe scripts/phase2_cloud.py evidence
```

`prepare_phase2.py --max-sources 3` produce un manifiesto de prueba con los
primeros tres archivos diarios. Sin esa opción prepara todo el histórico.
Acepta `--checks`, `--raw`, `--rules` y `--output` para cambiar las rutas.
Los valores predeterminados leen `facts.zip` y los cuatro catálogos de Downloads.

Los paquetes comprimidos contienen una fila envolvente por cuenta y conservan el
JSON original de la cuenta como texto, junto con origen, posición y huella.
Los números decimales conservan su representación exacta. El catálogo de empleados
incluye únicamente identificadores, puestos y contexto laboral necesario.

La carpeta de un snapshot es inmutable. Si una fuente cambia, preparar un nuevo
snapshot. Un archivo remoto existente con tamaño o hash diferente produce error.
No reemplazar objetos ya ingeridos: perjudicaría la trazabilidad y los checkpoints.

## Incremento, repetición y recuperación

Para la demostración de incorporación del último día:

```powershell
.venv/Scripts/python.exe -m phase2.batches RUTA/AL/manifest.json --through 2026-08-19
```

Primero cargar y ejecutar ese manifiesto acumulado. Después cargar el manifiesto
completo, actualizar el job con `deploy` y ejecutar otra vez. Las carpetas de
entrada son acumulativas: no se puede volver a un manifiesto pequeño después de
cargar uno mayor en el mismo snapshot. Para ensayos aislados usar otro esquema
y snapshot. Al repetir el manifiesto completo sin nuevas fuentes, los conteos y
los importes deben permanecer iguales.

Ante un fallo: consultar `status ID`, revisar eventos con `events`, corregir la
causa, desplegar el código y ejecutar de nuevo. Los archivos ya subidos se
verifican y reutilizan. La relectura exacta de una cuenta con el mismo UID y payload
se reduce a una fila; payloads incompatibles para un UID pasan a cuarentena.
No usar `full_refresh` como reparación automática: reconstruye estado y puede
consumir cuota. Documentar cualquier reconstrucción explícita.

## Ingeniería y límites

- **Streaming:** ingestión por archivos nuevos con Auto Loader en tablas Bronze.
  El job es disparado por lote, con horario automático pausado.
- **Vistas materializadas:** Silver y Gold conservan resultados preparados para
  consulta. Lakeflow decide si puede actualizarlas incrementalmente. Un plan de
  recomputación completa es válido y debe quedar visible en los eventos.
- **Liquid clustering:** `silver_checks` declara localización y fecha operativa
  como claves. No se combina con particiones ni Z-ORDER. Su beneficio requiere
  medir tamaño, consultas y mantenimiento; esta entrega no inventa una mejora.
- **Expectativas y excepciones:** claves, conciliación, conservación, rescate de
  esquema y referencias al catálogo producen aceptación, advertencia o rechazo
  explícitos. La cuarentena conserva fuente y causa.
- **Cuota:** Free Edition usa serverless y puede suspender cómputo por cuota.
  El job admite una sola ejecución simultánea y reintentos acotados. No crea
  recursos de pago ni activa un horario desatendido.

Las fechas son operativas (`BusinessDate`). Los horarios civiles se interpretan
con la zona del local y las horas ambiguas/inexistentes por DST quedan marcadas.
Abrir o cerrar una cuenta no indica cuándo se autorizó un descuento.

## Pruebas y evidencia

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/validate_notebook.py --checks RUTA/AL/EJEMPLO.json --raw RUTA/CATALOGOS
```

Las pruebas locales cubren transporte, privacidad, tolerancia monetaria, roles,
fuga temporal y explicaciones alternativas. El preflight adicional ejecuta las
transformaciones Spark compartidas en el runtime real de Databricks.

La evidencia privada por lote se descarga a `outputs/phase2_cloud/<batch_version>`:
estado del job, eventos, `validation.json` y `presentation_data.json`. Este último
incluye SQL y la procedencia Databricks Gold. La presentación debe usar únicamente
un lote aceptado y citar su ejecución, snapshot y versión de motivos.

## Fuentes técnicas

Documentación oficial de [Auto Loader](https://docs.databricks.com/aws/en/ingestion/cloud-object-storage/auto-loader/),
[expectativas](https://docs.databricks.com/aws/en/ldp/expectation-patterns),
[vistas materializadas](https://docs.databricks.com/aws/en/ldp/materialized-views),
[liquid clustering](https://docs.databricks.com/aws/en/delta/clustering) y
[límites de Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations).
