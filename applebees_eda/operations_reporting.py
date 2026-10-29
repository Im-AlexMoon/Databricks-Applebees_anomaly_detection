"""Post-analysis case briefs and next decisions; never fed back into the models."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .iterative import read_csv_optional


def _read_ids(path):
    try:
        return pd.read_csv(path,dtype={'item_id':'string'})
    except (FileNotFoundError,pd.errors.EmptyDataError):
        return pd.DataFrame()


def write_findings(run,con=None):
    run=Path(run)
    report_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    marker=run/'findings_manifest.json'
    if marker.exists() and json.loads(marker.read_text()).get('report_source_sha256')==report_hash:
        return _read_ids(run/'product_case_briefs.csv'),read_csv_optional(run/'next_priorities.csv')
    quality=read_csv_optional(run/'quality.csv').iloc[0]
    hypotheses=read_csv_optional(run/'hypotheses.csv')
    products=pd.read_parquet(run/'product_metrics.parquet')
    months=pd.read_parquet(run/'product_months.parquet')
    boot=_read_ids(run/'product_bootstrap.csv')
    groups=read_csv_optional(run/'amount_by_reason_group.csv')
    locations=read_csv_optional(run/'location_coding.csv')
    dictionary=read_csv_optional(run/'reason_dictionary.csv')
    review=json.loads((run/'review_status.json').read_text(encoding='utf-8'))
    products['workflow_marker_candidate']=products.item_name.fillna('').str.upper().str.contains(
        r"DON.?T.*(?:MAKE|POUR)|PRIORITY\s*TCKT",regex=True)
    products['item_role_candidate']=np.where(products.workflow_marker_candidate,'workflow_marker_candidate',
        np.where(products.unlinked_accounts/products.accounts_with_product<.5,'linked_component_candidate','unlinked_item_candidate'))
    if con is not None:
        priced=con.execute("""SELECT item_id,sum(gross_line_amount) exported_gross,
          count(*) FILTER(WHERE gross_line_amount>0) positive_price_pairs FROM ops_products GROUP BY 1""").fetchdf()
        products=products.merge(priced,on='item_id',how='left',validate='one_to_one')
        products.loc[~products.workflow_marker_candidate&(products.positive_price_pairs.fillna(0)==0),'item_role_candidate']='nonpriced_item_candidate'
    products.to_csv(run/'catalog_item_roles.csv',index=False)
    commercial=products.loc[products.sufficient_support&products.item_role_candidate.eq('unlinked_item_candidate')&
                            (products.excess_per_1000>0)].sort_values(['direct_per_1000','excess_per_1000'],ascending=False).head(12)
    commercial.to_csv(run/'commercial_candidate_shortlist.csv',index=False)
    briefs=pd.DataFrame()
    if len(boot):
        briefs=boot.merge(products,on='item_id',how='left',validate='one_to_one',suffixes=('_bootstrap',''))
        # Complete-month persistence uses the observed global source interval.
        start,end=pd.Timestamp(quality.first_day),pd.Timestamp(quality.last_day)
        observed=months.copy();observed['period_month']=pd.to_datetime(observed.period_month)
        full=(observed.period_month>=start)&((observed.period_month+pd.offsets.MonthEnd(0))<=end)
        usable=observed.loc[full&(observed.accounts>=30)].copy()
        usable['positive']=usable.excess_per_1000>0
        persistence=usable.groupby('item_id').agg(supported_complete_months=('positive','size'),positive_complete_months=('positive','sum')).reset_index()
        briefs=briefs.merge(persistence,on='item_id',how='left',validate='one_to_one')
        briefs[['supported_complete_months','positive_complete_months']]=briefs[['supported_complete_months','positive_complete_months']].fillna(0)
        briefs['evidence_level']=np.where(briefs.directly_linked_accounts>0,'direct_link_present_not_root_cause','cooccurrence_only')
        briefs['linked_line_share']=1-briefs.unlinked_accounts/briefs.accounts_with_product
        briefs['alternative_explanations']='Producto acompañante/modificador; corrección de otra línea; mezcla de contexto; código provisional'
        briefs['next_action']=np.where(briefs.workflow_marker_candidate,
            'Verificar cuándo se añade el marcador; no usarlo como predictor previo sin establecer secuencia temporal',
            np.where(briefs.linked_line_share>.5,
            'Verificar configuración/modificador y cuenta completa antes de atribuir incidencia al producto',
            'Contrastar eventos directamente vinculados y repetición por local/mes antes de proponer cambios'))
        briefs.to_csv(run/'product_case_briefs.csv',index=False)
    else:
        pd.DataFrame(columns=['item_id','evidence_level','next_action']).to_csv(run/'product_case_briefs.csv',index=False)
    if con is not None and len(briefs):
        con.register('brief_products',briefs[['item_id']])
        con.execute("""SELECT p.item_id,a.location_id,count(*) accounts,sum(a.has_correction::INT) corrections,
            sum(p.directly_linked_correction::INT) direct_corrections,
            1000*avg(a.has_correction::DOUBLE) correction_per_1000
            FROM ops_products p JOIN brief_products b USING(item_id) JOIN ops_accounts a USING(uid)
            WHERE a.eligible GROUP BY 1,2""").fetchdf().to_csv(run/'product_location_drilldown.csv',index=False)
        con.unregister('brief_products')
    priorities=[]
    h1=hypotheses.loc[hypotheses.hypothesis_id=='H1'].iloc[0]
    marker_count=int(briefs.workflow_marker_candidate.sum()) if len(briefs) else 0
    priorities.append({'priority':1,'problem':'Separar productos comerciales de marcadores y explicar correcciones',
        'evidence':str(h1.effect)+f'; {marker_count} de las fichas son candidatos a marcadores de proceso',
        'decision':'reformular' if marker_count else h1.decision,'next_action':'Homologar roles del catálogo y revisar candidatos comerciales con vínculo directo',
        'business_decision':'Elegir producto/proceso que requiere observación o revisión de configuración',
        'missing_information':'Composición real de combos, causa de la corrección y preparación del pedido',
        'artifact':'product_case_briefs.csv; product_location_drilldown.csv'})
    generic=float(groups.loc[(groups.kind=='discount')&(groups.operational_group=='generic_authorization'),'reconciled_discount_amount'].sum())
    priorities.append({'priority':2,'problem':'Mejorar explicación de autorizaciones genéricas',
        'evidence':f'{generic:.2f} de importe descontado conciliado; no es pérdida estimada',
        'decision':'continuar' if generic>0 else 'descartar','next_action':'Definir submotivos y contrastarlos con ejemplos por local',
        'business_decision':'Acordar captura y evidencia mínima para autorizaciones',
        'missing_information':'Políticas y significado de códigos por local/período','artifact':'generic_profiles.csv; reviewed_accounts.csv'})
    priorities.append({'priority':3,'problem':'Homologar comparación operativa entre locales',
        'evidence':f'Cambio máximo de ranking: {locations.absolute_rank_change.max():.0f} posiciones' if len(locations) else 'Sin locales comparables',
        'decision':'reformular','next_action':'Validar diccionario en locales con mayores cambios antes de comparar desempeño',
        'business_decision':'Elegir definiciones comunes y segmentación para seguimiento',
        'missing_information':'Diferenciar prácticas de captura de diferencias operativas reales','artifact':'location_coding.csv; reason_dictionary.csv'})
    pd.DataFrame(priorities).to_csv(run/'next_priorities.csv',index=False)
    lines=['# Hallazgos de operación comercial','',
       f"Fuentes: {quality.first_day} a {quality.last_day}; {int(quality.accounts):,} cuentas incluidas y {int(quality.eligible_accounts):,} conciliadas.",
       '',f"Venta bruta conciliada: {quality.gross:,.2f}. Descuento conciliado: {quality.discount:,.2f}. Importes del sistema; moneda no verificada.",
       '',f"Cuentas con corrección según diccionario provisional: {int(quality.correction_accounts):,}. Cuentas con descuento y anulación: {int(quality.discount_and_void_accounts):,}.",
       '',f"Diccionario: {len(dictionary):,} combinaciones de tipo/local/código/nombre; {(dictionary.review_status=='confirmed').sum():,} confirmadas operativamente.",
       '',f"Entre las fichas destacadas hay {marker_count} candidatos a marcadores de proceso. Su asociación puede ser consecuencia de una corrección; no tratarlos como artículos defectuosos o predictores previos sin validar secuencia.",
       '', '## Preguntas y decisiones','']
    for h in hypotheses.to_dict('records'):
        lines.extend([f"### {h['hypothesis_id']}. {h['question']}",'',f"**{h['decision']}**. {h['effect']}",'',
            f"Interpretación: {h['decision_reason']}. Alternativas: {h['alternative_explanations']}.",'',
            f"Acción: {h['next_action']}.",''])
    lines.extend(['## Revisión y límites','',
       f"Se contrastaron {review.get('reviewed',0)} cuentas con los JSON originales. Esto verifica coherencia de fuente, no cumplimiento de políticas ni causas.",
       '', 'Los productos se asocian por presencia en cuentas; solo parte de los eventos identifica un artículo. No sumar los importes por producto ni sumar anulaciones y descuentos como pérdida.',
       '', 'Los modelos son explicativos y usan una muestra uniforme con errores agrupados. Selección de productos y bootstrap son exploratorios; no hay validación independiente ni efecto causal demostrado.',
       '', 'El volumen de aperturas no mide carga simultánea, personal disponible, espera o preparación. No se propone una reducción de personal a partir de esta asociación.',
       '', '## Prioridades siguientes',''])
    for p in priorities:
        lines.extend([f"{p['priority']}. **{p['problem']}**: {p['next_action']}. Evidencia pendiente: {p['missing_information']}.",''])
    (run/'findings.md').write_text('\n'.join(lines),encoding='utf-8')
    (run/'findings_manifest.json').write_text(json.dumps({'report_source_sha256':report_hash,
        'run':str(run.resolve()),'no_feedback_into_models':True},indent=2),encoding='utf-8')
    return briefs,pd.DataFrame(priorities)
