"""Build the clean operational notebook; leave the two earlier notebooks untouched."""
import json
from pathlib import Path
import textwrap

cells=[]


def md(s):
    cells.append({"cell_type":"markdown","metadata":{},"source":textwrap.dedent(s).strip()+"\n"})


def code(s):
    cells.append({"cell_type":"code","metadata":{},"source":textwrap.dedent(s).strip()+"\n","execution_count":None,"outputs":[]})


md('''
# Operación comercial: preguntas, evidencia y decisiones

**Pregunta central:** ¿qué situaciones generan correcciones recurrentes, dónde
se concentran y qué proceso merece investigarse para mejorar la operación?

Este notebook reutiliza el Parquet del proyecto. No requiere extraer de nuevo el
ZIP ni reemplaza los dos EDA anteriores. La fase es explicativa sobre todo el
período disponible; no produce un modelo predictivo ni etiquetas de fraude.

La taxonomía guía el orden: **calidad y mecánica → descripción → relaciones →
hipótesis y variables útiles para decisiones**. Cada pregunta termina en continuar,
reformular o descartar, con efecto, población, incertidumbre y explicaciones
alternativas. Las decisiones son propuestas analíticas, no decisiones disciplinarias.
''')
code('''
from pathlib import Path
import os,sys,json
ROOT=next((p for p in [Path.cwd(),*Path.cwd().parents] if (p/'applebees_eda').is_dir()),None)
if ROOT is None:
    raise RuntimeError('Abre el notebook desde el repositorio')
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from IPython.display import display
from applebees_eda import Config,open_database
from applebees_eda.operations import run_operations
from applebees_eda.operations_data import OperationsConfig
from applebees_eda.operations_reporting import write_findings
from applebees_eda.iterative import read_csv_optional
sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns',22)
''')
md('''
## Configuración y ejecución reproducible

Las rutas por defecto usan Downloads y `outputs/eda/parquet`. Requieren el snapshot
existente y las fuentes sin cambios. El diccionario editable y las notas se guardan
en `outputs/operacion_comercial/`, fuera de Git. Los hashes de entradas, código,
diccionario y parámetros identifican cada iteración; repetirla reutiliza resultados.

Se parte de 100 cuentas con el producto y cobertura de pares del 80% para la
selección exploratoria. Se contrasta soporte de 30/100/300 cuentas. Las categorías
con pocos casos permanecen visibles sin encabezar la selección por tasas extremas.
''')
code('''
DOWNLOADS=Path.home()/'Downloads'
RAW=Path(os.environ.get('APPLEBEES_RAW',str(DOWNLOADS)))
config=Config(checks=(os.environ.get('APPLEBEES_CHECKS',str(DOWNLOADS/'facts.zip')+'::facts/checks/*/checks.json'),),
    catalogs={key:str(RAW/name) for key,name in {'revenue_centers':'revenueCenters.json','locations':'locations.json',
              'items':'items.json','employees':'employees.json'}.items()},
    output_dir=Path(os.environ.get('APPLEBEES_OUTPUT',str(ROOT/'outputs'/'eda'))),
    memory_limit=os.environ.get('APPLEBEES_MEMORY_LIMIT','4GB'))
settings=OperationsConfig()  # mismo protocolo que scripts/run_operations.py
OUTPUT_ROOT=Path(os.environ.get('APPLEBEES_OPERATIONS_OUTPUT',str(ROOT/'outputs'/'operacion_comercial')))
con=open_database(config)
RUN=run_operations(con,config,settings,OUTPUT_ROOT,review_json=True)
summary=json.loads((RUN/'complete.json').read_text(encoding='utf-8'))
briefs,next_priorities=write_findings(RUN,con)
print('Iteración:',RUN)
print('Cuentas:',summary['inputs']['accounts'],'; productos distintos:',summary['products'])
''')
md('''
## 1. Calidad y mecánica: ¿qué representa cada registro?

¿Los códigos significan lo mismo por local? ¿Una anulación y un descuento
describen eventos distintos dentro de una cuenta? ¿Hay líneas vinculadas a otras
líneas que podrían ser componentes o modificadores? ¿Qué representan TrafficCount
y el tiempo de apertura a cierre?

El diccionario conserva **tipo de evento + local + código + nombre**, con una clave
estable; identifica cambios de nombre y no homologa silenciosamente códigos.
La clasificación inicial se infiere de etiquetas y queda provisional o pendiente.
El estado `confirmed` solo debe usarse tras verificar una definición operativa.

Las cuentas no se multiplican por eventos, productos, gerentes o puestos. Los
productos se reducen a una presencia por cuenta, incluso si tienen varias líneas.
La duración es tiempo de cuenta abierta; TrafficCount no se interpreta como
comensales confirmados. No tenemos tiempo de preparación, horas trabajadas ni costos.
''')
code('''
quality=read_csv_optional(RUN/'quality.csv')
dictionary=read_csv_optional(RUN/'reason_dictionary.csv')
display(quality.T)
display(dictionary.groupby(['kind','operational_group','review_status']).size().rename('filas_diccionario').reset_index())
display(read_csv_optional(RUN/'reason_id_name_conflicts.csv').head(15))
display(read_csv_optional(RUN/'product_structure.csv').T)
print('Editar motivos:',OUTPUT_ROOT/'reason_dictionary.csv')
''')
md('''
## 2. Descripción: magnitud, denominadores y motivo registrado

¿Qué importe y cuántas cuentas aparecen bajo cada grupo? ¿Cuántas cuentas
combinan descuentos y anulaciones? ¿Cómo cambian las tasas entre semanas/locales?

**Importes separados:** descuento conciliado es importe descontado; anulación
es valor exportado del evento. No se suman como pérdida y no miden desperdicio.
Una cuenta puede tener varios motivos: los conteos por grupo no son aditivos.
Venta neta significa venta bruta menos descuento, antes de impuestos y propinas.

El gráfico temporal incluye semanas parciales para descripción. Días observados
no equivalen a cobertura garantizada de todos los turnos del restaurante.
''')
code('''
groups=read_csv_optional(RUN/'amount_by_reason_group.csv')
display(groups)
fig,axes=plt.subplots(1,2,figsize=(13,4))
for kind,col,title,ax in [('discount','reconciled_discount_amount','Descuentos conciliados',axes[0]),
                           ('void','exported_amount','Anulaciones: valor exportado',axes[1])]:
    plot=groups.loc[groups.kind==kind].groupby('operational_group')[col].sum().sort_values()
    if len(plot):
        plot.plot.barh(ax=ax,color='#4477AA')
    ax.set(title=title,xlabel='Importe en unidades monetarias del sistema',ylabel='Grupo provisional de motivo')
plt.tight_layout();fig.savefig(RUN/'amounts_by_group.png',dpi=140);plt.show()
weekly=read_csv_optional(RUN/'weekly_locations.csv')
weekly['week_start']=pd.to_datetime(weekly.week_start)
network=weekly.groupby('week_start')[['accounts','corrections','generic_accounts']].sum().sort_index()
fig,ax=plt.subplots(figsize=(11,3))
(1000*network[['corrections','generic_accounts']].div(network.accounts,axis=0)).rename(
    columns={'corrections':'Corrección registrada','generic_accounts':'Autorización genérica'}).plot(ax=ax)
ax.set(title='Incidencia semanal sobre cuentas incluidas',xlabel='Semana operativa',ylabel='Cuentas por cada 1,000 cuentas')
plt.tight_layout();fig.savefig(RUN/'weekly_rates.png',dpi=140);plt.show()
''')
md('''
## 3. H1: productos, exposición y atribución

¿Hay productos con más correcciones que las esperadas por contexto y volumen?
Los pares son cuentas **sin el producto**, dentro del mismo local, mes, centro,
franja y día de semana; los contextos pequeños se suavizan hacia local/mes y red,
también excluyendo el producto. Si un producto aparece en todas las cuentas de
un contexto, no hay una comparación suficiente y se reduce su cobertura.

La tasa principal mide cuentas que contienen producto y corrección: **coocurrencia,
no causa**. Un descuento de cuenta no identifica un producto responsable. Se
informa aparte el vínculo directo por línea de descuento o ItemId de anulación.
Una anulación de un artículo ya retirado de la venta puede no tener producto
vendido correspondiente; no se inventa esa exposición.

El catálogo también puede contener marcadores de proceso como “DON T MAKE FOOD”.
Las fichas los separan como candidatos por nombre/estructura/precio. Su presencia
puede ser consecuencia de una corrección: no usarla para predecir una corrección
futura sin establecer la secuencia. La lista comercial es una propuesta que
también requiere validación del catálogo, no una clasificación definitiva.

Los intervalos Wilson son referencias exploratorias bajo independencia. La
selección se profundiza con bootstrap por empleado, referencia de pares fija y
corrección BY en esa familia seleccionada. No corrige el sesgo de haber seleccionado
los productos destacados: los intervalos no constituyen confirmación independiente.
''')
code('''
products=pd.read_parquet(RUN/'product_metrics.parquet')
months=pd.read_parquet(RUN/'product_months.parquet')
bootstrap=read_csv_optional(RUN/'product_bootstrap.csv')
supported=products.loc[products.sufficient_support].copy() if not products.empty else products
columns=['item_id','item_name','accounts_with_product','corrections_per_1000','expected_per_1000',
         'excess_per_1000','direct_per_1000','unlinked_rate_per_1000','peer_coverage']
display(supported[columns].head(15) if not supported.empty else supported)
display(bootstrap)
display(briefs.head(12))
display(read_csv_optional(RUN/'commercial_candidate_shortlist.csv')[
    ['item_id','item_name','accounts_with_product','excess_per_1000','direct_per_1000','item_role_candidate']])
if len(supported):
    top=supported.head(12).copy().sort_values('excess_per_1000')
    top['label']=top.item_id.astype(str)+' · '+top.item_name.fillna('[sin nombre]').str.slice(0,30)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.barh(top.label,top.excess_per_1000,color='#4477AA')
    ax.set(title='Correcciones asociadas con artículos del catálogo',xlabel='Exceso frente a pares por 1,000 cuentas con el artículo',ylabel='Artículo registrado (incluye marcadores)')
    plt.tight_layout();fig.savefig(RUN/'product_excess.png',dpi=140);plt.show()
    display(months.loc[months.item_id.isin(top.item_id)].sort_values(['item_id','period_month']))
''')
md('''
## 4. H2: ¿aumentan las correcciones con mayor actividad?

Actividad es cantidad de cuentas abiertas en el mismo local, fecha operativa y
hora. Se contrasta el cuartil local más alto con el resto. **No mide personal
disponible, ocupación simultánea, tiempo de espera o capacidad de cocina.**

La regresión binomial usa hasta 40,000 cuentas conciliadas con vendedor único
y hora disponible, mediante muestra uniforme reproducible. Controla local,
centro, calendario, franja, categoría dominante, venta, líneas, TrafficCount y
efectivo. Los intervalos agrupan simultáneamente por empleado y fecha. La salida
es asociación ajustada, no causalidad. Una varianza inválida no se reemplaza por
un intervalo artificial. Los umbrales son contemporáneos: no sirven directamente
para anticipar una hora que todavía no termina.
''')
code('''
models=read_csv_optional(RUN/'adjusted_models.csv')
display(json.loads((RUN/'model_status.json').read_text(encoding='utf-8')))
display(models)
activity=read_csv_optional(RUN/'activity_sensitivity.csv')
display(activity.head(20))
if not models.empty:
    effects=models.loc[(models.contrast=='busy')&(models.scale=='percentage_points')].dropna(subset=['low','high'])
    if len(effects):
        fig,ax=plt.subplots(figsize=(8,3))
        ax.errorbar(effects.estimate,effects.target,xerr=np.vstack([effects.estimate-effects.low,effects.high-effects.estimate]),fmt='o',capsize=4)
        ax.axvline(0,color='gray',linestyle='--')
        ax.set(title='Asociación ajustada con horas de actividad alta',xlabel='Diferencia en puntos porcentuales · IC 95%',ylabel='Resultado registrado')
        plt.tight_layout();fig.savefig(RUN/'activity_effects.png',dpi=140);plt.show()
''')
md('''
## 5. H3 y H4: diferencias entre locales y autorizaciones genéricas

¿La comparación cambia si incluimos motivos genéricos junto a correcciones
específicas? La definición amplia es una sensibilidad, no una afirmación de que
MANAGER sea un error o incumplimiento. Un cambio de ranking puede reflejar
captura, mezcla comercial o diferencias operativas; no distingue sus causas.

¿Qué situaciones se registran como autorizaciones genéricas? Comparamos
perfiles de cuenta y asociaciones ajustadas. La identidad del gerente ayuda a
rastrear, pero no verifica la política ni sustituye un motivo específico.
''')
code('''
locations=read_csv_optional(RUN/'location_coding.csv')
display(locations.sort_values('absolute_rank_change',ascending=False).head(15))
shares=read_csv_optional(RUN/'local_reason_shares.csv')
display(shares.sort_values('exported_discount_amount',ascending=False).head(20))
display(read_csv_optional(RUN/'generic_profiles.csv').sort_values('accounts',ascending=False).head(15))
fig,ax=plt.subplots(figsize=(6,4))
ax.scatter(1000*locations.correction_rate,1000*locations.broad_rate,s=18,alpha=.6)
limit=max(1,float((1000*locations.broad_rate).max())*1.05)
ax.plot([0,limit],[0,limit],color='gray',linestyle='--')
ax.set(title='Sensibilidad de la definición por local',xlabel='Corrección específica por 1,000 cuentas',ylabel='Corrección o genérico por 1,000 cuentas')
plt.tight_layout();fig.savefig(RUN/'coding_sensitivity.png',dpi=140);plt.show()
''')
md('''
## 6. Robustez y revisión de fuentes

La sensibilidad conserva umbrales de soporte, perfiles temporales, estructura de
líneas y convención monetaria alternativa. La muestra de revisión incluye
correcciones por descuento, anulaciones, genéricos, solapamientos, productos y
controles elegidos aleatoriamente dentro del mismo local/fecha/centro/franja.
Se distribuye sobre seis días del histórico para que el contraste con JSON sea
repetible y acotado; no es una muestra representativa de pérdidas.

Se leen los miembros del ZIP y filas originales. Se verifica identidad e importes
y se conservan motivos y procedencia. La revisión confirma coherencia de fuentes;
no confirma causa, cumplimiento de política, costo ni fraude.
''')
code('''
display(read_csv_optional(RUN/'support_sensitivity.csv'))
display(read_csv_optional(RUN/'money_sensitivity.csv'))
display(json.loads((RUN/'review_status.json').read_text(encoding='utf-8')))
review=read_csv_optional(RUN/'reviewed_accounts.csv')
display(review[['sample_group','check_id','location_id','business_day','discount','exported_void_amount',
                'discount_reasons','void_reasons','identity_verified','policy_verified']])
''')
md('''
## 7. Decisiones y siguiente iteración

Cada hipótesis tiene población, denominador, método, efecto, intervalo o evidencia,
explicaciones alternativas, decisión y acción propuesta. Se puede descartar una
especificación sin afirmar que el problema nunca existe. Las decisiones explicativas
no usan significancia como único criterio ni convierten el importe en ahorro esperado.

Edita `analyst_notes.csv` para añadir interpretación sin perderla al repetir.
Si se confirma el significado de un motivo, actualiza el diccionario y vuelve a
ejecutar: se generará una versión nueva. Prioriza contraste de productos/contextos
y mejora del registro según la evidencia; después decide entre operación,
promociones o demanda. Promociones rentables requieren costos y causalidad;
pronóstico requiere un corte temporal nuevo y comparación contra una referencia.
''')
code('''
hypotheses=read_csv_optional(RUN/'hypotheses.csv')
for row in hypotheses.to_dict('records'):
    print(row['hypothesis_id'],row['question'])
    print('Decisión:',row['decision'],'|',row['effect'])
    print('Motivo:',row['decision_reason'])
    print('Siguiente acción:',row['next_action'])
    print()
display(hypotheses)
display(next_priorities)
display(read_csv_optional(OUTPUT_ROOT/'analyst_notes.csv'))
print('Resultados:',RUN)
con.close()
''')


def main():
    for i,c in enumerate(cells):
        c['id']=f'operations-{i:02d}'
    nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
    path=Path(__file__).resolve().parents[1]/'notebooks'/'eda_operacion_comercial.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':
    main()
