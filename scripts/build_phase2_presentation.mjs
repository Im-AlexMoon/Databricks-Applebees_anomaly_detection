/** Editable eight-slide delivery from an accepted Databricks export, never local estimates.
 * Run with the bundled artifact-tool runtime. See docs/phase2/README.md.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';

const root=process.cwd();
const input=path.resolve(process.argv[2]??'');
const runId=process.argv[3];
if(!process.argv[2]||!runId)throw Error('Usage: build_phase2_presentation.mjs presentation_data.json RUN_ID [revision]');
const revision=process.argv[4]??'v1';
if(!/^[A-Za-z0-9_-]+$/.test(revision))throw Error('Invalid revision name');
const build=path.join(root,'.cache/phase2_deck');
const out=path.join(root,'outputs/phase2_delivery',revision);
await fs.mkdir(build,{recursive:true});await fs.mkdir(out,{recursive:true});
const runtime=process.env.CODEX_PRESENTATION_RUNTIME??path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies');
const skill=process.env.CODEX_PRESENTATION_SKILL??path.join(process.env.USERPROFILE,'.codex/plugins/cache/openai-primary-runtime/presentations/26.1024.11018/skills/presentations');
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
process.env.PATH=path.join(runtime,'bin/override')+path.delimiter+path.join(runtime,'bin/fallback')+path.delimiter+process.env.PATH;
const nodeModules=path.join(build,'node_modules');
try{await fs.access(nodeModules);}catch{await fs.symlink(path.join(runtime,'node/node_modules'),nodeModules,'junction');}
const require=createRequire(path.join(build,'loader.cjs'));
const {Presentation,PresentationFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font=resolvePresentationFont();
const fontPolicy={basis:'design',families:[font]};
const coverPath=path.join(build,'cover.jpg');
try{await fs.access(coverPath);}catch{
 const response=await fetch('https://assets.simpleviewinc.com/simpleview/image/upload/c_fill%2Ch_396%2Cq_75%2Cw_704/v1/crm/easternkentucky/Applebees_027FE738-5056-A36A-0A750410E4201862-027fe5dc5056a36_027fe794-5056-a36a-0a2b7e20ac7a054c.jpg');
 if(!response.ok)throw Error('Could not download the contextual cover photograph');
 await fs.writeFile(coverPath,new Uint8Array(await response.arrayBuffer()));
}
const d=JSON.parse(await fs.readFile(input,'utf8'));
const validation=JSON.parse(await fs.readFile(path.join(path.dirname(input),'validation.json'),'utf8'));
if(validation.status!=='passed'||d.batch_version!==validation.manifest_version)throw Error('Expected accepted matching Databricks evidence');
const m=d.quality;
const n=v=>Number(v??0);
const fmt=v=>n(v).toLocaleString('es-MX',{maximumFractionDigits:0});
const pct=(v,digits=2)=>(100*v).toFixed(digits)+'%';
const money=v=>n(v).toLocaleString('es-MX',{minimumFractionDigits:2,maximumFractionDigits:2})+' UM';
const jobUrl=`https://dbc-8785bcd0-4e22.cloud.databricks.com/jobs/335175700298250/runs/${runId}`;
const source=`Fuente: Databricks Gold, batch ${d.batch_version}. Snapshot ${d.snapshot_id}. Ejecución ${jobUrl}. SQL: presentation_data.json / query_definitions. ${m.first_day} a ${m.last_day}.`;
const p=Presentation.create({slideSize:{width:1280,height:720}});
const c={ink:'#172B28',muted:'#536862',green:'#165943',red:'#A92B35',paper:'#FAF9F5',line:'#D9E0D8'};
function text(s,value,x,y,w,h,size=23,bold=false,color=c.ink){
 const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=value;sh.text.style={typeface:font,fontSize:size,bold,color,wrap:'word',insets:{top:0,bottom:0,left:0,right:0}};
 return sh;
}
function slide(title,notes){const s=p.slides.add();s.background.fill=c.paper;text(s,title,64,44,1152,110,43,true);s.speakerNotes.text=notes+'\n\n'+source;return s;}
function foot(s,value){text(s,value,64,652,1152,44,18,false,c.muted);}
function box(s,title,detail,x,y,w=245,h=145){
 const a=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:'#EDF2EC',line:{fill:c.line,width:1}});
 text(s,title,x+14,y+16,w-28,34,25,true,c.green);text(s,detail,x+14,y+60,w-28,h-65,22);
 return a;
}
function connect(s,a,b,fromSide='right',toSide='left'){s.shapes.connect(a,b,{kind:'elbow',fromSide,toSide,line:{fill:c.green,width:2},tail:{type:'triangle',width:'med',length:'med'}});}
function chart(s,categories,series,options={}){
 series=series.map(s=>({...s,values:s.values.map(v=>Number(v.toFixed(6)))}));
 const ch=s.charts.add('bar',{position:{left:64,top:190,width:850,height:390},categories,series,hasLegend:series.length>1,
  legend:{position:'bottom',textStyle:{fontSize:19}},
  barOptions:{direction:'bar',grouping:'clustered',gapWidth:70},
  dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0',textStyle:{fontSize:19}},
  xAxis:{textStyle:{fontSize:19},numberFormatCode:'0.0',majorGridlines:{fill:c.line,width:1}},
  yAxis:{textStyle:{fontSize:20}},chartFill:c.paper,plotAreaFill:c.paper,...options});
 applyPresentationChartFont(ch,{fontFamily:font});return ch;
}

// 1. Cover photo provides context, never evidence about the analyzed locations.
{
 const s=p.slides.add();s.background.fill=c.paper;
 text(s,'Control y explicación\nde descuentos',64,66,1120,158,58,true);
 text(s,'Pipeline de Applebee’s en Databricks',64,252,580,76,32,false,c.green);
 text(s,`${fmt(m.valid_accounts)} cuentas\n${fmt(m.source_days)} días de información\n${m.first_day} a ${m.last_day}`,64,369,540,145,27);
 s.images.add({blob:new Uint8Array(await fs.readFile(path.join(build,'cover.jpg'))),contentType:'image/jpeg',alt:'Restaurante Applebee’s en Corbin, Kentucky. Imagen contextual ajena a la muestra.',fit:'cover',position:{left:665,top:265,width:550,height:310}});
 foot(s,'Fase 2. Ingeniería, calidad y valor del dato. Interpretaciones operativas provisionales.');
 s.speakerNotes.text='0:00–0:40. Nuestro objetivo es explicar cómo los descuentos y las correcciones aparecen en los datos y permitir su revisión. No estimamos fraude ni pérdidas. Foto contextual, no es evidencia de una ubicación del dataset. Foto: Explore Kentucky Wildlands, https://www.explorekywildlands.com/listing/applebee%E2%80%99s-neighborhood-bar-%26-grill/2804/ .\n'+source;
}
// 2. Native editable architecture.
{
 const s=slide('De archivos privados a tablas para decidir','0:40–1:35. Describir origen, transporte sin filtrado analítico y capas. El catálogo de empleados omite nombre y nacimiento. Cada cuenta conserva archivo y posición de origen.');
 const a=box(s,'Fuentes','ZIP histórico\n4 catálogos\nMotivos versionados',64,220);
 const b=box(s,'Bronze','Auto Loader\nJSON original\nUID y SHA256',365,220);
 const e=box(s,'Silver','Cuenta y eventos\nRoles separados\nConciliación',666,220);
 const g=box(s,'Gold','Calidad\nMotivos y semanas\nProductos',967,220);
 connect(s,a,b);connect(s,b,e);connect(s,e,g);
 const q=box(s,'Cuarentena','Fuente y causa\nRevisión explícita',666,450,245,125);connect(s,e,q,'bottom','top');
 text(s,'Un manifiesto identifica la población de cada ejecución.',64,472,530,85,28,true);
 foot(s,'Unity Catalog Volume para entrada privada. Tablas Delta y vistas materializadas para consulta.');
}
// 3. Native diagram explaining the transformation grain.
{
 const s=slide('Una cuenta conserva su importe al cruzar eventos','1:35–2:25. Ejemplo sintético: dos artículos, tres descuentos y dos pagos. Un cruce directo produce doce combinaciones. Agregamos cada tabla hija por cuenta antes de unir. Los puestos del empleado permanecen aparte.');
 const a=box(s,'Artículos','2 filas',64,205,260,110);
 const b=box(s,'Descuentos','3 filas',64,345,260,110);
 const e=box(s,'Pagos','2 filas',64,485,260,110);
 const r=box(s,'Agregados por cuenta','Una fila por hijo',450,340,315,130);
 connect(s,a,r);connect(s,b,r);connect(s,e,r);
 const z=box(s,'Cuenta final','1 fila\nImportes conservados',900,340,315,130);connect(s,r,z);
 text(s,'Ejemplo sintético',450,206,720,42,25,true,c.muted);
 text(s,'La clave combina localización, fecha operativa e Id.',450,510,730,85,27,true);
 foot(s,'Vendedor desde artículos vendidos. Aplicador y gerente se conservan como roles independientes.');
}
// 4. Native editable workflow and actual run evidence.
{
 const s=slide('Cuatro tareas controlan la publicación del resultado','2:25–3:20. Mostrar la ejecución aceptada. Preflight valida huellas y casos sintéticos. El pipeline transforma. La aceptación compara contra el manifiesto y la cohorte local. La última tarea solo exporta después del éxito.');
 const labels=[['preflight','SHA256\nPruebas Spark'],['update_pipeline','Bronze, Silver\ny Gold'],['Validación','Conteos e importes\nReferencia local'],['Exportación','Agregados\npara presentar']];
 const shapes=labels.map((v,i)=>box(s,v[0],v[1],64+i*301,225,245,160));shapes.slice(1).forEach((v,i)=>connect(s,shapes[i],v));
 text(s,`Ejecución ${runId}\nAceptación: ${validation.status}`,64,443,1120,87,29,true,c.green);
 text(s,'Una ejecución simultánea. Horario pausado. Reintento acotado.',64,560,1120,50,25);
 foot(s,'El fallo de una expectativa no revierte todas las tablas. La tarea de aceptación controla la exportación.');
}
// 5. Actual quantitative quality.
{
 const s=slide('La calidad se verifica con población e importes','3:20–4:15. Enseñar las reglas de aceptación. Los datos sintéticos no entran en las tablas históricas. Describir las cuentas que no cumplen la población monetaria y conservar su presencia para análisis descriptivo. Liquid clustering declara localización y fecha, sin afirmar mejoras de rendimiento no medidas.');
 text(s,fmt(m.valid_accounts),64,190,535,80,64,true,c.green);
 text(s,'cuentas válidas\n'+fmt(m.quarantine_accounts)+' en cuarentena',64,284,535,85,27);
 text(s,pct(n(m.reconciled_evaluable_accounts)/n(m.evaluable_accounts),3),680,190,530,80,64,true,c.green);
 text(s,`conciliación entre ${fmt(m.evaluable_accounts)}\ncuentas evaluables`,680,284,530,85,27);
 text(s,`${fmt(m.eligible_accounts)} cuentas para importes\nTolerancia contable: 0.02 UM\nReferencia local y Databricks: misma cohorte`,64,429,1120,143,27);
 foot(s,'También se comprueban hijos, duplicados, roles, catálogos y horarios. Bronze conserva el payload.');
}
// 6. Editable financial chart with explicit population and unit.
{
 const rows=d.datasets.reason_groups.filter(r=>r.kind==='discount').reduce((a,r)=>{a[r.operational_group]=(a[r.operational_group]??0)+n(r.reconciled_discount_amount);return a;},{});
 const names={pending:'Pendiente',promotion_candidate:'Promoción candidata',generic_authorization:'Autorización genérica',correction:'Corrección',employee_benefit_candidate:'Beneficio empleado',collection_candidate:'Ajuste de cobro candidato',other:'Otros'};
 const entries=Object.entries(rows).sort((a,b)=>b[1]-a[1]).slice(0,6);
 const s=slide('Los motivos cambian la interpretación del descuento','4:15–5:15. Importes exclusivamente de cuentas conciliadas. La taxonomía de motivos es provisional. Las autorizaciones genéricas merecen mejor explicación operativa. El dinero descontado no equivale a pérdida. Las anulaciones se reportan por separado.');
 chart(s,entries.map(([k])=>names[k]??k.replaceAll('_',' ')),[{name:'Miles de UM',values:entries.map(([,v])=>v/1000),fill:c.green}]);
 text(s,money(m.eligible_discount),940,220,275,120,35,true,c.green);
 text(s,'Descuento conciliado\n\n'+pct(n(m.accounts_with_discount)/n(m.valid_accounts))+' de cuentas\ncon descuento',940,366,275,170,24);
 foot(s,'Gráfico en miles de unidades monetarias del exportado. Grupos provisionales. Población conciliada.');
}
// 7. Editable comparison chart: direct evidence versus co-occurrence.
{
 const rows=d.datasets.product_examples.slice(0,3);
 const s=slide('La presencia de un producto requiere contexto','5:15–6:10. Las etiquetas DON T MAKE o PRIORITY TCKT pueden registrar el flujo de una corrección. Su asociación no permite concluir que el producto causó el problema. Comparar presencia en cuenta con enlaces de descuento por línea o void por ItemId.');
 chart(s,rows.map(r=>String(r.item_name).slice(0,26)),[
  {name:'Coocurrencia',values:rows.map(r=>1000*n(r.correction_accounts)/n(r.accounts)),fill:c.green},
  {name:'Enlace directo',values:rows.map(r=>1000*n(r.direct_accounts)/n(r.accounts)),fill:c.red}],{position:{left:64,top:190,width:850,height:405}});
 text(s,'Tasas por mil\ncuentas con\nel producto',940,220,275,130,28,true);
 text(s,'Marcadores,\nmodificadores\ny productos tienen\nroles distintos.',940,404,275,160,24);
 foot(s,'Top 3 de ejemplos con al menos 100 cuentas. Coocurrencia y enlace directo no implican causalidad.');
}
// 8. Operational next steps and demonstration.
{
 const s=slide('El siguiente paso es validar motivos y procesos','6:10–7:00. Concluir valor del pipeline y pasar a tres minutos de demo. 7:00–10:00 mostrar job, DAG, Gold y evidencia incremental. 10:00–12:00 preguntas. Una revisión operativa independiente deberá verificar las explicaciones y medir resultados antes/después.');
 text(s,'1',64,198,60,65,43,true,c.green);text(s,'Validar la taxonomía con operación',151,205,1030,50,31,true);text(s,'Confirmar motivos genéricos, correcciones y promociones.',151,264,1030,60,25);
 text(s,'2',64,352,60,65,43,true,c.green);text(s,'Revisar cuentas con trazabilidad',151,359,1030,50,31,true);text(s,'Contrastar excepciones, roles y explicaciones del ticket.',151,418,1030,60,25);
 text(s,'3',64,506,60,65,43,true,c.green);text(s,'Medir mejoras con datos posteriores',151,513,1030,50,31,true);text(s,'Comparar tasas e importes con volumen y cobertura equivalentes.',151,572,1030,60,25);
 foot(s,'Demo: job y dependencias, DAG, calidad, Gold e incremento sin duplicación.');
}
// Render draft and validate native objects before writing the final output.
const draft=path.join(build,`candidate-${revision}.pptx`);
await(await PresentationFile.exportPptx(p)).save(draft);
await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:path.join(out,'Applebees_Fase2.pptx'),
 pythonExecutable:path.join(runtime,'python/python.exe'),integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:8,requiredNativeChartOwnerSlides:[6,7],requiredNativeTableOwnerSlides:[],
 materializeLiteralChartWorkbooks:true,fontPolicy,verifyArtifactToolImport:true,
 receiptPath:path.join(build,`validation-${revision}.json`)});
for(const [i,s] of p.slides.items.entries()){
 const png=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
// Raster PDF from the inspected slide renders. PPTX retains native editable objects.
// The bundled PDF exporter currently cannot resolve its Helvetica font module on Windows.
const pdfResult=spawnSync(path.join(runtime,'python/python.exe'),[path.join(root,'scripts/render_phase2_pdf.py'),
 '--images',build,'--output',path.join(out,'Applebees_Fase2.pdf')],{encoding:'utf8'});
if(pdfResult.status!==0)throw Error(pdfResult.stderr||pdfResult.stdout||'PDF rendering failed');
await fs.writeFile(path.join(out,'procedencia.json'),JSON.stringify({input,runId,batch:d.batch_version,snapshot:d.snapshot_id,source:'Databricks Gold',font,slides:8},null,2));
await fs.writeFile(path.join(out,'notas_presentador.md'),p.slides.items.map((s,i)=>`## Diapositiva ${i+1}\n\n${s.speakerNotes.text}`).join('\n\n'));
console.log(JSON.stringify({out,slides:p.slides.items.length,font,batch:d.batch_version}));
