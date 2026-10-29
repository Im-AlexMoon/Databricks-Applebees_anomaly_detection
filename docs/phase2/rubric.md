# Cobertura de la rúbrica

1. **GitHub y documentación:** README de la fase, instrucciones reproducibles,
   estructura exigida, diccionario, pruebas y evidencia de ejecución.
2. **Job y diseño:** YAML del job con cuatro tareas dependientes. El manifiesto
   define la población. La aceptación determina cuándo exportar.
3. **Transformaciones y valor de tablas:** Bronze conserva fuentes, Silver
   resuelve grano y conciliación, Gold permite comparar motivos, locales,
   semanas y presencia de productos.
4. **Streaming, vistas, clustering y expectativas:** Auto Loader, vistas
   materializadas, claves de liquid clustering, cuarentena y reglas con
   advertencias/rechazos. Los eventos muestran el plan realmente elegido.
5. **Calidad:** claves, duplicados, arrays nulos, rescate de esquema, importes,
   conteos de hijos, roles, DST y cobertura de catálogos. Los casos sintéticos
   están separados de las fuentes históricas.
6. **Análisis y negocio:** métricas de descuentos conciliados, clasificación
   provisional de motivos, sensibilidad de rankings y distinción entre productos
   y marcadores operativos. Proponer revisión de procesos con soporte observable.
7. **Comunicación:** ocho diapositivas, fuentes y denominadores en notas, demo
   breve de DAG, aceptación y consulta de Gold, con respaldo de la evidencia.

## Guion de demo, tres minutos

- 0:00–0:40: abrir el job y explicar dependencias, horario pausado y parámetros.
- 0:40–1:20: mostrar la ejecución aceptada y el DAG Bronze/Silver/Gold.
- 1:20–2:00: abrir `gold_quality`, comprobar población y conciliación.
- 2:00–2:40: consultar motivos/semanas y explicar el denominador y el importe.
- 2:40–3:00: mostrar incremento y repetición sin duplicados. Cerrar con una
  decisión operativa y con la información que falta para validarla.

Si el cómputo no está disponible, usar el respaldo de la ejecución verificada.
Identificarlo expresamente como evidencia grabada de esa ejecución.
