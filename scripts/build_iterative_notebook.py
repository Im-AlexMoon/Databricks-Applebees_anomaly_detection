"""Build the second notebook without touching the original EDA or its edits."""
import json
from pathlib import Path
import textwrap

cells=[]


def md(source):
    cells.append({"cell_type":"markdown","metadata":{},"source":textwrap.dedent(source).strip()+"\n"})


def code(source):
    cells.append({"cell_type":"code","metadata":{},"source":textwrap.dedent(source).strip()+"\n",
                  "execution_count":None,"outputs":[]})


md('''
# EDA iterativo · hipótesis, incertidumbre y aprendizaje débil

Unidad: **empleado–localización–semana**. Buscamos descuentos inusuales,
persistentes y económicamente relevantes, con trazabilidad hasta el JSON.
Este notebook reutiliza el Parquet del EDA inicial; no vuelve a convertir los 10 GB.

Ruta: **estadística → GMM + Isolation Forest → pseudoetiquetas → GaussianNB / logística**.
GaussianNB es supervisado. Las etiquetas representan hipótesis de anomalía;
su imitación no mide detección de fraude. `fraud_confirmed` permanece sin asignar.
Los identificadores, nombres, edad y estado actual del empleado no son predictores.

Las convenciones provisionales son precio por línea, descuentos registrados y
resta de propina exportada. La elegibilidad monetaria se calcula por cuenta y
se valida con el entrenamiento de cada partición: agosto no modifica mayo.
''')
code('''
from pathlib import Path
import os, sys, json
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'applebees_eda').is_dir()), None)
if ROOT is None:
    raise RuntimeError('Abre este notebook desde el repositorio')
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from IPython.display import display
from applebees_eda import Config, open_database
from applebees_eda.iterative_data import ExperimentConfig
from applebees_eda.iterative import run_experiment, read_csv_optional
from applebees_eda.weak_supervision import FEATURES
from applebees_eda.iterative_reporting import compare_models
sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns',20)
''')
md('''
## 1. Configuración y períodos congelados

Exploración: 4 mayo–28 junio; validación: 6–26 julio; evaluación retrospectiva:
3–16 agosto. Las otras semanas se conservan para descripción. Una semana es
completa si hay siete fechas de fuente; esto no supone siete días trabajados por
empleado ni prueba que la extracción de cada local esté completa.

Entrenamiento de pseudoetiquetas: dos semanas iniciales de referencia y bloques
posteriores de dos semanas con referencia expansiva. Julio y agosto usan la misma
referencia final de mayo–junio, sin reajuste con julio. Los umbrales 95%/50%, las
variables y los mínimos se definen aquí antes de evaluar agosto. Para cambiarlos,
crea otra iteración y documenta la decisión; agosto sigue siendo retrospectivo.
''')
code('''
DOWNLOADS = Path.home()/'Downloads'
RAW = Path(os.environ.get('APPLEBEES_RAW',str(DOWNLOADS)))
config = Config(
    checks=(os.environ.get('APPLEBEES_CHECKS',str(DOWNLOADS/'facts.zip')+'::facts/checks/*/checks.json'),),
    catalogs={key:str(RAW/name) for key,name in {
        'revenue_centers':'revenueCenters.json','locations':'locations.json',
        'items':'items.json','employees':'employees.json'}.items()},
    output_dir=Path(os.environ.get('APPLEBEES_OUTPUT',str(ROOT/'outputs'/'eda'))),
)
experiment = ExperimentConfig(seed=42, min_checks=30, prior_checks=30, min_peers=3,
    bootstrap_runs=int(os.environ.get('APPLEBEES_BOOTSTRAP_RUNS','10')),
    bootstrap_ci_runs=300, statistical_sample=40000)
OUTPUT_ROOT=Path(os.environ.get('APPLEBEES_ITERATIVE_OUTPUT',str(ROOT/'outputs'/'eda_iterativo')))
MONETARY_MODE='records_sum'  # alternativa: 'price_difference_plus_check'
con=open_database(config)  # exige el snapshot existente y fuentes sin cambios
display(pd.DataFrame([vars(experiment)]).T.rename(columns={0:'valor'}))
''')
md('''
## 2. Ejecutar o reutilizar una iteración

Cada ejecución se identifica con hashes del snapshot, código y parámetros.
Conserva tabla semanal, puntuaciones, selección GMM, estadística, métricas y
procedencia bajo `outputs/eda_iterativo/`; los datos quedan fuera de Git.
El primer cálculo puede tardar varios minutos. Repetir una iteración completa
reutiliza sus resultados. Cambiar parámetros crea otra carpeta.

La regresión usa una muestra uniforme reproducible de hasta 40,000 cuentas
conciliadas de mayo–junio. Controla local, centro, franja de apertura, calendario,
venta, número de artículos, comensales y categoría dominante. Usa covarianza
agrupada simultáneamente por empleado y fecha. Un intervalo no válido se deja
vacío. No interpreta esas asociaciones como efectos causales.
''')
code('''
RUN = run_experiment(con, config, experiment, OUTPUT_ROOT, review_json=True, monetary_mode=MONETARY_MODE)
summary=json.loads((RUN/'complete.json').read_text(encoding='utf-8'))
print('Resultados:',RUN)
print('Filas semanales:',summary['weekly_rows'],'; puntuadas:',summary['scored_rows'])
print('Regresión:',summary['glm']['status'],'; segunda pasada:',summary['student']['status'])
weekly=pd.read_parquet(RUN/'weekly_dataset.parquet')
scores=pd.read_parquet(RUN/'weekly_scores.parquet')
display(weekly.groupby('phase').agg(filas=('seller_id','size'),empleados=('seller_id','nunique'),
    cuentas=('n','sum'),con_exposicion=('enough_exposure','sum'),calidad=('quality_ok','sum')))
''')
md('''
## 3. Calidad, composición y convención alternativa

Las cuentas sin vendedor único se conservan en el resumen de calidad y no se
atribuyen al empleado del pago. Empleados eliminados y HomeLocationId=0 siguen
siendo contexto histórico, sin entrar en las variables del modelo. Los gerentes
sin correspondencia reducen cobertura; no forman una identidad sospechosa común.

Compara registros de descuento frente a diferencias de precios más descuento
de cuenta. Los importes excluyen cuentas no conciliadas; su suma es descuento,
no pérdida probada. Examina semanas/locales discordantes antes de concluir.
''')
code('''
quality=read_csv_optional(RUN/'quality_weekly.csv')
quality['week_start']=pd.to_datetime(quality.week_start)
quality=quality.sort_values('week_start')
sensitivity=read_csv_optional(RUN/'weekly_location_distribution.csv')
display(quality)
display(sensitivity.assign(diferencia=lambda d:d.alternative_discount-d.discount)
    .sort_values('diferencia',key=abs,ascending=False).head(15))
display(read_csv_optional(RUN/'unexplained_reductions.csv').head(20))
fig,ax=plt.subplots(figsize=(11,3))
quality.plot(x='week_start',y='reconciliation',ax=ax,legend=False)
ax.set(title='Conciliación provisional por semana',ylabel='Proporción conciliada',xlabel='Semana')
ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=3))
fig.autofmt_xdate()
plt.tight_layout(); fig.savefig(RUN/'quality.png',dpi=140); plt.show()
''')
md('''
## 4. Frecuencia ajustada, efectivo y anulaciones

`context_glm.csv` contiene razones de probabilidades y diferencias ajustadas en
puntos porcentuales, con intervalos del 95%. La interacción separa la asociación
conjunta de efectivo y anulaciones. Se corrigen comparaciones con Benjamini–Yekutieli
por familia, permitiendo dependencia. No seleccionar resultados solo por p/q.

Las tasas esperadas del experimento usan pares por local, centro, franja y día
de semana, excluyendo al empleado. Los contextos pequeños se suavizan hacia la
referencia local y global. Los intervalos beta describen incertidumbre condicional
a esa referencia; no corrigen por sí solos la dependencia entre turnos.
''')
code('''
glm=read_csv_optional(RUN/'context_glm.csv')
display(json.loads((RUN/'context_glm_status.json').read_text(encoding='utf-8')))
display(glm)
if not scores.empty:
    fig,ax=plt.subplots(figsize=(7,4))
    plot=scores.loc[scores.score_period=='oof'].sample(min(3000,(scores.score_period=='oof').sum()),random_state=42)
    ax.scatter(plot.peer_rate,plot.smoothed_rate,s=7,alpha=.25)
    ax.plot([0,1],[0,1],color='gray',linestyle='--')
    upper=min(1,max(.35,float(plot[['peer_rate','smoothed_rate']].max().max())+.05))
    ax.set_xlim(0,upper); ax.set_ylim(0,upper)
    ax.set(xlabel='Frecuencia esperada entre pares',ylabel='Frecuencia observada suavizada',title='Mayo–junio: resultados fuera de partición')
    plt.tight_layout(); fig.savefig(RUN/'observed_expected.png',dpi=140); plt.show()
''')
md('''
## 5. Intensidad, reducción sin explicación y cambios compartidos

El bootstrap remuestrea empleados conservando juntas sus observaciones. Compara
intensidad en efectivo frente a otros pagos, con intervalos y denominadores.
La descripción incluye medianas, percentiles y MAD. Revisa cambios compartidos
por varios empleados: una promoción puede mover a todo el local.

`location_sensitivity.csv` es una sensibilidad descriptiva sin ajuste; sus pruebas
suponen independencia y no sustituyen la regresión agrupada. Las diferencias entre
descuentos de cuenta/artículo se muestran sin duplicar importes de transacciones.
''')
code('''
display(read_csv_optional(RUN/'intensity_bootstrap.csv'))
shifts=read_csv_optional(RUN/'shared_shifts.csv')
display(shifts.loc[shifts.shared_shift_candidate].sort_values('rate_change_pp',key=abs,ascending=False).head(20))
fig,ax=plt.subplots(figsize=(11,3))
sensitivity.groupby('week_start')[['discount','alternative_discount']].sum().plot(ax=ax)
ax.set(title='Sensibilidad de importes conciliados',xlabel='Semana',ylabel='Importe descontado')
plt.tight_layout(); fig.savefig(RUN/'monetary_sensitivity.png',dpi=140); plt.show()
''')
md('''
## 6. Motivos y parejas empleado–gerente

Se recalculan todas las parejas, incluidas las poco frecuentes. La comparación
semanal usa actividad marginal de empleados y gerentes dentro del local.
La exposición real depende de turnos no disponibles: concentración no demuestra
colaboración indebida. Los gerentes desconocidos quedan fuera de estas pruebas
y se reflejan en cobertura. HHI de motivos/autorizaciones usa registros positivos,
no un denominador inventado de todas las cuentas atendidas por un gerente.
''')
code('''
pairs=pd.read_parquet(RUN/'pair_tests.parquet')
print('Parejas-semanas con gerente conocido:',len(pairs))
display(pairs.sort_values(['persistent','discount'],ascending=False).head(20) if not pairs.empty else pairs)
motives=pd.read_parquet(RUN/'motive_tests.parquet')
display(motives.sort_values(['persistent','discount'],ascending=False).head(15) if not motives.empty else motives)
display(weekly[['motive_hhi','manager_hhi','manager_coverage']].describe())
''')
md('''
## 7. GMM, baja densidad y pseudoetiquetas

Se comparan 1–6 componentes y covarianza diagonal/completa mediante BIC,
convergencia y ARI entre tres semillas. Un componente es válido. Una pertenencia
alta o un grupo pequeño no significa fraude. El escalado se ajusta en entrenamiento.
Las variables constantes aparecen sin correlaciones definidas en el mapa.

**1:** ≥30 cuentas, cobertura ≥80%, ambos detectores sobre el percentil 95 de
entrenamiento y exceso sobre pares respaldado por intervalo durante dos semanas
consecutivas. **0:** ambos en la mitad menos inusual, sin exceso persistente.
**−1:** resto. El 5% es selección experimental, no prevalencia.

El peso es la fracción de remuestreos por empleado que conserva la pseudoetiqueta;
se reajustan pares, transformaciones y detectores. La arquitectura GMM elegida
queda fija dentro de esos remuestreos. Diez réplicas dan una exploración inicial;
para una siguiente iteración aumenta `bootstrap_runs` y compara estabilidad.
''')
code('''
display(read_csv_optional(RUN/'gmm_selection.csv'))
display(read_csv_optional(RUN/'label_coverage.csv'))
teacher_comparison,convention_comparison=compare_models(RUN)
display(teacher_comparison)
display(convention_comparison)
stability=read_csv_optional(RUN/'bootstrap_stability.csv')
display(stability.groupby('score_period').top50_overlap.describe() if 'top50_overlap' in stability else stability)
if not scores.empty:
    fig,ax=plt.subplots(figsize=(8,6))
    sns.heatmap(scores.loc[scores.score_period=='oof',FEATURES].corr(method='spearman'),
                cmap='coolwarm',vmin=-1,vmax=1,ax=ax)
    ax.set_title('Correlaciones de señales: entrenamiento temporal')
    plt.tight_layout(); fig.savefig(RUN/'feature_correlations.png',dpi=140); plt.show()
    display(scores.loc[scores.pseudo_label==1,
        ['seller_id','location_id','week_start','n','discount','peer_rate','smoothed_rate','pseudo_weight','pseudo_reason']]
        .sort_values('discount',ascending=False).head(20))
''')
md('''
## 8. Segunda pasada: qué aporta y qué no aporta

GaussianNB y regresión logística regularizada usan solo pseudoetiquetas temporales
de mayo–junio y `sample_weight`. Se excluyen identificadores, pertenencias,
puntuaciones y etiquetas de las variables predictoras. Reutilizar las señales
originales todavía induce una tarea de imitación: no descubre verdad independiente.

Se exige soporte por clase en filas, empleados y peso efectivo. Si falta, se
omite el ajuste. Las métricas son concordancia con etiquetas débiles y solapamiento
con las 50 prioridades semanales del primer modelo. `predict_proba` no es una
probabilidad de fraude. Julio sirve para observar estabilidad; agosto es evaluación
retrospectiva, no una validación prospectiva independiente.
''')
code('''
display(json.loads((RUN/'student_status.json').read_text(encoding='utf-8')))
display(read_csv_optional(RUN/'student_metrics.csv'))
''')
md('''
## 9. Revisar cuentas y decidir la próxima iteración

La muestra incluye alertas elevadas si existen, desacuerdos y controles elegidos
aleatoriamente entre perfiles de referencia. Verifica identidad e importe leyendo
las filas fuente del JSON/ZIP, conserva motivos registrados y anota información
faltante. Esa revisión solo observa datos del sistema; no confirma legitimidad ni fraude.

Edita la bitácora persistente `outputs/eda_iterativo/hypothesis_log.csv` con hipótesis,
población, denominador, método, efecto, intervalo, explicaciones y decisión
(`continuar`, `reformular`, `descartar`). El notebook no sobrescribe tus notas.
Cada carpeta también conserva una plantilla con referencias a la evidencia.

Para profundizar: cambia una especificación por iteración, aumenta remuestreos,
contrasta convención monetaria y sensibilidad por período/localización; compara
prioridades y documenta qué cambia. Buscar fraude real requerirá datos posteriores
y evidencia operativa independiente.

Para repetir el experimento con la convención alternativa, cambia `MONETARY_MODE`
a `price_difference_plus_check`. Se reutilizan las mismas cuentas Parquet, se
recalculan importes/cobertura y se genera otra carpeta con modelos y métricas.
Al regresar a esta iteración, la sección 7 compara automáticamente ambas
convenciones si encuentra una ejecución completa con los mismos parámetros.
Las reducciones sin explicación se buscan entre cuentas evaluables incluso si
no concilian: descartarlas al principio ocultaría precisamente esa señal.
''')
code('''
display(read_csv_optional(RUN/'json_review.csv'))
display(read_csv_optional(OUTPUT_ROOT/'hypothesis_log.csv'))
print('Trazabilidad: ',RUN/'account_trace.parquet')
# Ejemplo, sin multiplicar cuentas: filtrar por la clave semanal de una señal.
if not scores.empty:
    target=scores.sort_values(['rank_score','discount'],ascending=False).iloc[0]
    trace=str(RUN/'account_trace.parquet').replace("'","''")
    display(con.execute(f"SELECT uid,check_id,business_day,source_file,source_row,gross,discount FROM read_parquet('{trace}') WHERE seller_id=? AND location_id=? AND week_start=? ORDER BY discount DESC LIMIT 10",
        [target.seller_id,target.location_id,target.week_start.to_pydatetime()]).fetchdf())
con.close()
''')


def main():
    for i,cell in enumerate(cells):
        cell['id']=f'iterative-{i:02d}'
    document={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
              "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":5}
    target=Path(__file__).resolve().parents[1]/'notebooks'/'eda_iterativo.ipynb'
    target.write_text(json.dumps(document,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(target)


if __name__=='__main__':
    main()
