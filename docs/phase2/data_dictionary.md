# Contratos y unidades de análisis

## Bronze

`bronze_checks` tiene una fila por registro transportado. `record_uid` combina la
huella del miembro fuente y su posición. `source_file`, `source_row`, `source_id`,
`snapshot_id`, `rules_version`, `landing_file` e `ingested_at` preservan el linaje.
`payload` conserva el JSON de la cuenta. `_rescued_data` y `_corrupt_record`
registran datos fuera del contrato y errores de lectura.

`bronze_catalogs` usa el mismo envoltorio e identifica el catálogo en `entity`.
Incluye localizaciones, centros, productos, empleados y reglas provisionales.

## Silver

- `silver_check_candidates`: un registro por UID dentro del snapshot. Distingue
  repetición exacta, UID contradictorio, clave inválida, clave lógica duplicada,
  payload inválido y arrays incompletos. No elige arbitrariamente un duplicado.
- `silver_quarantine`: registros excluidos con payload, fuente y causa.
- `silver_checks`: una cuenta por `(location_id, business_day, check_id)` válida.
  Los importes usan `DECIMAL(20,6)`. `gross` suma precios brutos por línea;
  `discount` suma descuentos registrados de cuenta y artículo.
  `net_sales = gross - discount`.
- `reconciliation_residual = payment - (net_sales + tax + service_charge - gratuity)`.
  `money_evaluable` requiere datos monetarios completos y positivos de acuerdo
  con el contrato. `money_eligible` exige además conciliación a 0.02 y neto válido.
  La alternativa con diferencia de precios queda disponible para sensibilidad.
- `seller_id`: empleado único de los artículos vendidos, con cobertura completa.
  Cuentas con atribución incompleta o múltiple conservan seller nulo. El empleado
  de pago permanece en pagos y no sustituye al vendedor.
- `silver_sold_items`: cuenta y `line_index`, con ItemId, cantidad, precios,
  secuencia y padre. PLU es identificador. No multiplica precio por cantidad.
- `silver_discounts`: cuenta, nivel, línea y evento. Conserva vendedor/aplicador
  y gerente como roles independientes. Descuentos cero también se conservan.
- `silver_payments`, `silver_voids`, `silver_taxes`: cuenta e índice del evento.
  El importe exportado de una anulación no se suma a descuentos como pérdida.
- `silver_locations`, `silver_items`, `silver_revenue_centers`,
  `silver_employees`: claves de catálogo únicas. Id ambiguos no se cruzan.
  Empleados eliminados siguen disponibles para hechos históricos.
- `silver_employee_jobs`: empleado y puesto, separada de transacciones.
- `silver_reason_rules`: clave del motivo, grupo y estado de revisión. La
  semántica es provisional y su versión se identifica por SHA256.
- `silver_operational_events`: evento de descuento/anulación y contexto de cuenta.
  La correspondencia exige tipo, localización, identificador y nombre del motivo.
  Gerentes sin correspondencia se reportan como cobertura incompleta.
- `silver_account_operations`: una cuenta con banderas de corrección y
  autorización genérica, calculadas después de agrupar eventos por cuenta.

## Gold

- `gold_quality`: una fila con conteos, importes, cuarentena, cobertura y
  conciliación de la versión materializada. La tarea de aceptación la compara
  con el manifiesto y la misma cohorte del EDA local.
- `gold_discount_reasons`: local, semana, tipo, id, nombre, grupo y revisión.
  Las cuentas pueden aparecer en varios motivos, por lo que su conteo no es
  aditivo. El importe conciliado de descuentos es distinto del valor de voids.
- `gold_operations_weekly`: local y semana. Tasas por mil cuentas, cuentas
  observadas y elegibles, días cubiertos e importes de la población conciliada.
  No compara totales entre semanas sin considerar días y volumen.
- `gold_product_metrics`: local, mes y producto presente por cuenta elegible.
  Una cuenta cuenta una vez por producto aunque tenga varias líneas.
  Distingue coocurrencia y enlace directo de corrección por línea/ItemId.
  Los marcadores de flujo y modificadores tienen un rol candidato explícito.
  Una asociación elevada no prueba que el producto causó un error.

Los nombres y fechas de nacimiento de empleados no entran al análisis. Las
identidades sirven para trazabilidad y contexto, nunca como números de riesgo.
