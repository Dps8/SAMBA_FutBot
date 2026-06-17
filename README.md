# SAMBA FutBot

Pipeline reproducible para la categoria profesional del reto Copa FutBotMX,
capitulo Vision por Computadora. El objetivo es segmentar y rastrear campo,
robots y balon en videos de futbol robotico usando SAM 3, y convertir esas
detecciones en metricas, eventos, mapas tacticos y videos demo.

La ruta actual no depende solo de un prompt a SAM 3. Para la camara superior se
usa una estrategia hibrida:

1. SAM 3 segmenta campo y robots.
2. SAM 3 detecta porterias azul/amarilla cuando `--goals` esta activo.
3. SAM 3 tambien intenta detectar la pelota por prompts semanticos.
4. Un detector HSV/forma aporta una pista configurable de color.
5. Para porterias, el rango HSV puede recalibrarse desde las cajas que SAM 3
   encuentre en el propio video.
6. Se descartan manchas de color dentro de robots o fuera de contexto.
7. Se fusionan candidatos de SAM 3 y color.
8. Se filtran robots duplicados por puntaje, tamano, IoU, contencion y
   distancia entre centros.
9. Se refina la trayectoria de la pelota con consistencia temporal.
10. Se asignan IDs de tracking y equipo por color dominante del robot.
11. Se calculan metricas, posesion por equipo, eventos, homografia, reportes y QA.

## Estado Actual

- Categoria de trabajo: profesional.
- Modelo base: `facebook/sam3`.
- Variante recomendada para vista superior: `process-top-camera`.
- Campo oficial usado para homografia: `2.43 m x 1.82 m`.
- Pelota oficial considerada: pelota naranja tipo golf, `42 mm`; el color se
  maneja como perfil configurable, no como unica fuente de deteccion.
- Resultados parciales principales: `outputs/review/2026-05-27/18abril_top_camera`.

## Estructura Del Repositorio

```text
config/
  default.yml                         Configuracion principal: prompts, SAM3, tracking y analisis.
  top_camera_homography_template.yml  Plantilla de homografia con medidas oficiales.

data/
  raw/                                Videos crudos descargados o copiados.
  frames/                             Frames de calibracion/debug.
  manifests/                          Indices generados del Drive.

docs/
  RETO.md                             Resumen local del reto.
  RESULTS.md                          Resultados parciales y observaciones.
  TECHNICAL_WALKTHROUGH.md            Explicacion tecnica extendida del repo.

outputs/
  detections/                         Detecciones JSONL.
  tracks/                             Detecciones con track_id.
  metrics/                            Resumenes numericos JSON.
  events/                             Eventos candidatos JSON.
  events/*event-summary.json          Marcador candidato y conteos deportivos.
  field_analysis/                     Homografia, CSVs y mapas tacticos.
  qa/                                 Reportes automaticos de calidad.
  qa/*                                Incluyen claim_readiness para saber que metricas son defendibles.
  pseudolabels/                       Manifiestos de candidatos para fine-tuning posterior.
  videos/                             Videos demo renderizados.
  review/                             Resultados revisados por fecha/camara.

src/samba_futbot/
  Codigo Python del paquete y CLI.

tests/
  Pruebas unitarias por modulo.
```

## Instalacion

Recomendado: Python 3.12, GPU NVIDIA, CUDA compatible con PyTorch y acceso a los
checkpoints de SAM 3 en Hugging Face.

Instalacion base:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Dependencias para SAM 3:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-sam3.txt
huggingface-cli login
```

Si no instalas el paquete con `pip install -e .`, puedes correr los comandos con:

```powershell
$env:PYTHONPATH="src"
python -m samba_futbot.cli --help
```

## Configuracion Importante

Archivo principal: `config/default.yml`.

Puntos clave:

- `project.drive_root_id`: carpeta publica de Google Drive con los videos.
- `sam3.model_id`: modelo a usar, por defecto `facebook/sam3`.
- `sam3.prompts.field`: prompts para campo verde.
- `sam3.prompts.robots`: prompts para robots.
- `sam3.prompts.ball`: prompts semanticos para pelota.
- `ball_detection.sam3_enabled`: activa la fuente SAM3 para pelota en camara superior.
- `ball_detection.color_enabled`: activa la fuente de color/forma.
- `ball_detection.default_profile`: perfil cromatico por defecto, hoy `orange`.
- `ball_detection.profiles`: rangos HSV configurables para pelota naranja,
  blanca, amarilla u otros perfiles.
- `team_detection.enabled`: activa clasificacion de robots por equipo.
- `team_detection.palette`: colores RGB esperados para equipos `blue` y
  `yellow`.
- `team_detection.max_color_distance`: distancia maxima contra la paleta para
  aceptar un pixel como evidencia de equipo.
- `team_detection.min_saturation`, `min_value`, `min_pixels`: filtros para que
  la clasificacion de equipo use pixeles cromaticos suficientes y no el fondo
  verde/zonas oscuras del robot.
- `goal_detection.color_enabled`: activa fallback cromatico para porterias
  azul/amarilla cuando SAM3 no las encuentra por prompt.
- `goal_detection.adaptive_color`: si SAM3 encuentra `goal_blue` o
  `goal_yellow`, toma pixeles dentro de esas cajas y ajusta el HSV usado por el
  fallback del mismo video.
- `goal_detection.broad_profiles`: rangos amplios por familia de color. Sirven
  para buscar "azules posibles" y "amarillos posibles" sin depender del tono
  exacto.
- `goal_detection.spatial_gate_from_seeds`: si hay cajas semilla de SAM3,
  limita la busqueda cromatica a regiones cercanas a esas coordenadas para
  evitar objetos externos del mismo color.
- `goal_detection.require_seed_for_color`: si esta activo, una clase de
  porteria necesita al menos una caja semilla de SAM3 antes de aceptar
  detecciones cromaticas de esa clase. El default actual es `false` para que la
  porteria azul pueda recuperarse por color aunque SAM3 no la haya propuesto.
- `goal_detection.require_field_overlap`: exige que la porteria aceptada este
  sobre o tocando el campo detectado. Con `max_per_frame_per_class: 1`, el
  pipeline conserva como maximo una porteria azul y una amarilla por frame.
- `goal_detection.infer_missing_opposite`: si en un frame hay campo y solo se
  detecta una porteria de color, crea la porteria opuesta por simetria respecto
  al eje central del campo y la marca con `source: goal_geometry`. Esta apagado
  por defecto porque puede crear una porteria falsa junto a la porteria real si
  la geometria de campo no esta limpia.
- `robot_filter.enabled`: activa el filtro conservador para camara superior.
  Reduce cajas repetidas de robots usando varios criterios combinados: area
  minima, area maxima relativa al frame, IoU, contencion, distancia minima entre
  centros y maximo por frame. El default actual conserva hasta 4 robots por
  frame y protege candidatos cercanos al balon, porque esos robots suelen ser
  los mas importantes para posesion, distancia al balon y analisis de jugada.
- `--robot-recovery-box-expand-x-px`,
  `--robot-recovery-box-expand-top-px` y
  `--robot-recovery-box-expand-bottom-px`: expanden las cajas del fallback HSV
  de robots oscuros. Esto compensa que el color oscuro detecte a veces solo una
  parte del robot, especialmente cerca del balon.
- En los renders, los colores se asignan por objeto antes que por equipo:
  pelota naranja, robots rojo/blanco y porterias con su color real. Las
  porterias inferidas por geometria no se dibujan en el video demo. Las labels
  no muestran equipo por defecto; usa `--show-team-labels` solo cuando el
  reporte de calidad de equipos sea defendible.
- `analysis.possession_radius_px`: distancia robot-pelota para contar posesion.
- `analysis.in_play_field_margin_px`: margen para decidir si la pelota esta en juego.
- `analysis.ball_border_margin_px`: filtro contra falsos positivos en bordes.
- Los reportes QA generan `claim_readiness`, una matriz de evidencia para
  `ball_tracking`, `metric_speed_trajectory`, `team_possession`,
  `goal_scoring` y `shot_pressure`. Esto separa lo que ya se puede defender ante
  jurado de lo que todavia debe revisarse visualmente.
- Un evento `goal_candidate` mantiene `goal_scoring` en revision. El claim solo
  pasa a listo cuando existe evidencia emitida como `goal_confirmed`; el
  marcador mostrado antes de eso siempre es candidato.
- `analysis.goal_confirmation` valida temporalmente el gol: descarta porterias
  inferidas por geometria, exige una pelota que venga desde fuera, movimiento
  hacia la porteria y permanencia dentro durante varios frames. Sus umbrales se
  pueden ajustar en `config/default.yml`.

Para validar eventos existentes sin repetir SAM3:

```powershell
python -m samba_futbot.cli validate-goals `
  --tracks "outputs\tracks\clip-tracks.jsonl" `
  --events "outputs\events\clip-events.json" `
  --out "outputs\events\clip-validated-events.json" `
  --config config/default.yml
```

Plantilla de homografia: `config/top_camera_homography_template.yml`.

Contiene medidas oficiales:

- Campo: `2.43 m x 1.82 m`.
- Circulo central: `0.60 m`.
- Area de penalizacion: `0.25 m x 0.80 m`.
- Porteria: `0.60 m x 0.10 m`.

Antes de reportar distancias metricas finales hay que reemplazar `image_points`
por las cuatro esquinas reales del campo en el video analizado.

## Uso Basico

Ver ayuda:

```powershell
samba-futbot --help
```

O sin instalacion editable:

```powershell
$env:PYTHONPATH="src"
python -m samba_futbot.cli --help
```

Ver metadata de un video:

```powershell
python -m samba_futbot.cli video-info --video "data\raw\video.mov"
```

Extraer frames para inspeccion:

```powershell
python -m samba_futbot.cli sample-frames `
  --video "data\raw\video.mov" `
  --out-dir "data\frames\video" `
  --every 5 `
  --max-frames 12
```

Exportar candidatos de pseudo-etiquetas desde detecciones SAM3, para preparar
fine-tuning sin entrenar todavia:

```powershell
python -m samba_futbot.cli export-pseudolabels `
  --detections "outputs\detections\clip\detections.jsonl" `
  --out "outputs\pseudolabels\clip-pseudolabel-candidates.json" `
  --classes "robots,ball,goal_blue,goal_yellow" `
  --min-score 0.60 `
  --min-area 20 `
  --require-mask
```

Exportar frames/crops auditables para revision humana o fine-tuning posterior:

```powershell
python -m samba_futbot.cli export-frame-dataset `
  --video "outputs\review\2026-05-27\18abril_top_camera\clips\IMG_9938_f001799_10s.mp4" `
  --detections "outputs\review\2026-05-27\18abril_top_camera\runs\tracks\IMG_9938_f001799_10s-top-hybrid-ball-v1-in-play-tracks.jsonl" `
  --out-dir "outputs\datasets\IMG_9938_f001799_10s" `
  --classes "robots,ball,goal_blue,goal_yellow" `
  --min-score 0.60 `
  --split-strategy by-video
```

Este comando escribe `manifest.json`, frames completos y crops por clase. El
split por defecto es `by-video` para evitar fuga de informacion entre frames
casi identicos del mismo clip. Si se necesita una exploracion rapida de un solo
video, `--split-strategy by-frame` reparte por frame, pero no debe usarse para
reportar validacion final.

Convertir ese manifest a formatos de entrenamiento:

```powershell
python -m samba_futbot.cli merge-frame-datasets `
  --manifests "outputs\datasets\clip_a\manifest.json,outputs\datasets\clip_b\manifest.json" `
  --out "outputs\datasets\top_camera_merged\manifest.json" `
  --split-strategy by-source-balanced

python -m samba_futbot.cli export-coco `
  --manifest "outputs\datasets\top_camera_merged\manifest.json" `
  --out-dir "outputs\datasets\top_camera_merged_coco" `
  --image-root "outputs\datasets\top_camera_merged"
```

`merge-frame-datasets --split-strategy by-source-balanced` mantiene cada video
en un solo split, pero balancea fuentes completas entre `train` y `val` cuando
hay pocos clips. Es la opcion recomendada para preparar pseudo-etiquetas
auditables y fine-tuning sin introducir detectores externos no permitidos.
Cuando el manifest conserva `mask_path` y `mask_index`, `export-coco` agrega
segmentaciones RLE COCO y un bloque de auditoria con mascaras referenciadas,
exportadas y fallidas. Las cajas se mantienen si una mascara individual no se
puede leer.

Crear un subconjunto determinista centrado en pelota para experimentos de
balance:

```bash
PYTHONPATH=src python -m samba_futbot.cli balance-coco \
  --annotations "/data/coco/annotations/train.json" \
  --out "/data/coco/annotations/train-ball-balanced.json" \
  --focus-classes ball \
  --negative-ratio 1.0 \
  --seed 123
```

`--focus-only` elimina las otras categorias y crea un dataset especialista.
Debe usarse solo como experimento: una imagen sin anotacion de pelota no es un
negativo confiable cuando las etiquetas son pseudo-etiquetas. El JSON registra
esta limitacion como `negative_semantics`; para resultados finales, los
negativos y mascaras de pelota deben revisarse manualmente.

Preparar una configuracion de adaptacion SAM3 desde el YAML oficial instalado:

```bash
PYTHONPATH=src python -m samba_futbot.cli prepare-sam3-finetune \
  --template "/opt/sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml" \
  --out "/opt/sam3/sam3/train/configs/samba_futbot/robots_ball.yaml" \
  --data-root "/data/top_camera_merged" \
  --train-json "/data/top_camera_merged_coco/annotations/train.json" \
  --val-json "/data/top_camera_merged_coco/annotations/val.json" \
  --experiment-dir "/runs/samba_futbot/robots_ball" \
  --bpe-path "/opt/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz" \
  --epochs 3 \
  --train-limit 64 \
  --val-limit 64 \
  --resolution 1008
```

El generador activa segmentacion, perdidas mask/Dice, decodificacion RLE en
train/val, prompts por categoria y adaptacion de las cabezas de segmentacion y
scoring. El backbone queda congelado porque full fine-tuning de 840 M
parametros no cabe de forma estable en una GPU de 16 GB.

Instalar primero las dependencias oficiales de entrenamiento dentro del entorno
que contiene SAM3:

```bash
cd /opt/sam3
python -m pip install -e ".[train]"
```

Ejecutar con el lanzador compatible con autograd:

```bash
PYTHONPATH="src:/opt/sam3" python scripts/run_sam3_finetune.py \
  -c configs/samba_futbot/robots_ball.yaml \
  --use-cluster 0 \
  --num-gpus 1
```

Los checkpoints, configuraciones resueltas, logs, TensorBoard y predicciones
COCO quedan dentro de `--experiment-dir`. Ver `docs/SAM3_FINETUNING.md` para
el contrato de evaluacion antes de promover un checkpoint.

Comparar el modelo base y el checkpoint adaptado sobre exactamente las imagenes
presentes en ambos archivos de predicciones:

```bash
PYTHONPATH=src python -m samba_futbot.cli compare-sam3-finetune \
  --ground-truth "/data/top_camera_merged_coco/annotations/val.json" \
  --baseline "/runs/samba_futbot/baseline/dumps/coco_predictions_segm.json" \
  --candidate "/runs/samba_futbot/robots_ball/dumps/coco_predictions_segm.json" \
  --out "/runs/samba_futbot/comparison/comparison.json" \
  --report-out "/runs/samba_futbot/comparison/comparison.md"
```

La comparacion recalcula COCO AP/AR global y por clase. Restringe el ground
truth a los IDs realmente inferidos, evitando que un `val-limit` reduzca
artificialmente las metricas por evaluar imagenes que no tuvieron prediccion.

Antes de entrenar o adaptar, auditar el manifest:

```powershell
python -m samba_futbot.cli dataset-quality `
  --manifest "outputs\datasets\top_camera_merged\manifest.json" `
  --out "outputs\datasets\top_camera_merged\quality.json" `
  --report-out "outputs\datasets\top_camera_merged\quality.md" `
  --low-score-threshold 0.60
```

Ese paso marca cajas invalidas, detecciones de baja confianza, frames sin
detecciones, rutas duplicadas, repeticiones del mismo `video + frame_index` y
videos compartidos entre splits. Esto evita fuga temporal antes de cualquier
adaptacion compatible con SAM.

Crear el manifest curado que se usara para adaptacion:

```powershell
python -m samba_futbot.cli curate-dataset `
  --manifest "outputs\datasets\top_camera_merged\manifest.json" `
  --out "outputs\datasets\top_camera_merged\curated-manifest.json" `
  --report-out "outputs\datasets\top_camera_merged\curation-report.json" `
  --classes "ball,robots,goal_blue,goal_yellow" `
  --min-score 0.60 `
  --deduplicate-source-frames
```

La deduplicacion conserva, para cada video y frame, la variante con mayor
cantidad de cajas validas y mejor evidencia de confianza. El reporte registra
que imagen se conservo y cuales se descartaron, de modo que la reduccion del
dataset queda auditable.

Preparar un paquete de revision humana de pelota:

```bash
PYTHONPATH=src python -m samba_futbot.cli select-ball-review \
  --manifest "outputs/review/2026-06-15/mask_training_v2/merged_mask_manifest.json" \
  --out "outputs/review/2026-06-15/ball_review/ball-review-v2-dense.json" \
  --report-out "outputs/review/2026-06-15/ball_review/ball-review-v2-dense-report.json" \
  --positive-frames 40 \
  --negative-frames 40 \
  --source-group-mode original-video \
  --min-frame-gap 2
```

El comando separa frames con pelota detectada (`verify_mask`) y frames sin
pseudo-etiqueta de pelota (`verify_absence`). Los segundos son negativos
candidatos, no negativos reales: deben revisarse manualmente antes de usarse
como ausencia confirmada. `original-video` agrupa clips como
`IMG_9933_f000000_10s` y `IMG_9933_f008995_10s` bajo la misma grabacion
original para reducir fuga temporal.

Auditar y exportar esa revision despues de editarla:

```bash
PYTHONPATH=src python -m samba_futbot.cli audit-ball-review \
  --review "outputs/review/2026-06-15/ball_review/ball-review-v2-dense.json" \
  --out "outputs/review/2026-06-15/ball_review/ball-review-v2-dense-audit.json" \
  --report-out "outputs/review/2026-06-15/ball_review/ball-review-v2-dense-audit.md"

PYTHONPATH=src python -m samba_futbot.cli export-reviewed-ball \
  --review "outputs/review/2026-06-15/ball_review/ball-review-v2-dense.json" \
  --out "outputs/review/2026-06-15/ball_review/reviewed-ball-manifest.json" \
  --report-out "outputs/review/2026-06-15/ball_review/reviewed-ball-report.json" \
  --split-strategy by-source-balanced \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

`export-reviewed-ball` falla por defecto si queda cualquier frame pendiente.
Los positivos deben tener anotaciones humanas en `annotations`; los negativos
solo entran al manifest si `ball_absent_verified` es `true`. El split
`by-source-balanced` mantiene todos los frames de una misma grabacion original
en train, val o test, sin mezclarla entre particiones. El resumen separa
`mask_ready_annotations` de `bbox_only_annotations`: las segundas sirven para
COCO bbox, pero no bastan para un fine-tuning de segmentacion SAM3.

Congelar un holdout para anotacion humana independiente:

```powershell
python -m samba_futbot.cli select-holdout `
  --manifest "outputs\datasets\top_camera_merged\curated-manifest.json" `
  --out "outputs\datasets\top_camera_merged\human-holdout.json" `
  --report-out "outputs\datasets\top_camera_merged\human-holdout-report.json" `
  --max-frames 24 `
  --preferred-split val `
  --seed 2026
```

La plantilla no copia detecciones ni cajas pseudo-etiquetadas. Solo conserva
las rutas de imagen, el estado `pending` y clases esperadas como metadato de
estratificacion. El reporte incluye una huella SHA-256 para repetir exactamente
la misma seleccion al comparar modelos.

Indexar Drive:

```powershell
python -m samba_futbot.cli index-drive `
  --config config/default.yml `
  --out data\manifests\drive_index.json
```

Descargar videos:

```powershell
python -m samba_futbot.cli download-all `
  --manifest data\manifests\drive_index.json `
  --out-dir data\raw
```

## Pipeline Recomendado Para Camara Superior

Este es el flujo principal para clips desde arriba, como los de:

```text
data/raw/Meta_Glasses/18abril/Camara_superior
```

Ejemplo:

```powershell
python -m samba_futbot.cli process-top-camera `
  --config config/default.yml `
  --video "outputs\review\2026-05-27\18abril_top_camera\clips\IMG_9938_f001799_10s.mp4" `
  --results-dir "outputs\review\2026-05-27\18abril_top_camera\runs" `
  --render
```

Que hace internamente:

1. Lee el video con OpenCV.
2. Ejecuta SAM 3 para `field,robots,goal_blue,goal_yellow`; los prompts de
   porteria incluyen cajas, postes, tableros, mesas y azul oscuro.
3. Ejecuta SAM 3 para `ball` con prompts semanticos.
4. Detecta pelota por color/forma usando el perfil HSV configurado.
5. Si SAM3 encontro porterias, calibra el HSV real desde esas cajas.
6. Agrega porterias por color azul/amarillo si el fallback cromatico esta
   activo.
7. Si solo aparece una porteria, infiere la opuesta por geometria del campo.
8. Fusiona campo, robots, porterias y candidatos de pelota.
9. Refina la pelota con `refine-ball`.
10. Genera tracks y asigna equipo de robots por color.
11. Clasifica estado de juego (`in_play`, `dead_ball`,
   `human_intervention`) y eventos externos como robot retirado o detenido.
12. Filtra metricas, eventos y homografia para contar solo frames `in_play`.
13. Calcula metricas y posesion por equipo.
14. Detecta eventos candidatos, incluyendo goles visuales si hay porteria.
15. Renderiza dos demos MP4: narrativa del partido y analisis tecnico.
16. Genera QA automatico.

Parametros utiles:

- `--sam3-ball / --no-sam3-ball`: activa o desactiva la fuente SAM3 para pelota.
- `--color-ball / --no-color-ball`: activa o desactiva la fuente HSV/forma.
- `--ball-window-size`, `--ball-step`, `--ball-start`: ventanas/anclas para
  SAM3 pelota.
- `--ball-threshold`: umbral SAM3 para pelota.
- `--ball-color-profile`: perfil cromatico (`orange`, `white`, `yellow`, etc.).
- `--ball-hsv-lower` y `--ball-hsv-upper`: limites HSV manuales `H,S,V`.
- `--orange-min-area`: area minima para la fuente de color.
- `--orange-max-area`: area maxima permitida para pelota.
- `--orange-max-per-frame`: candidatos maximos antes del refinamiento.
- `--goals / --no-goals`: activa o desactiva deteccion visual de porterias
  azul/amarilla.
- `--color-goals / --no-color-goals`: activa o desactiva fallback HSV para
  porterias azul/amarilla. Cuando `goal_detection.adaptive_color` esta activo,
  este fallback aprende el rango real desde las cajas semilla detectadas por
  SAM3.
- `--refine-max-jump-px`: salto maximo tolerado entre frames.
- `--refine-preferred-area`: area esperada de la pelota.
- `--trail-length`: longitud del rastro visual.
- `--max-seconds`: limita duracion renderizada.
- `--render-narrative / --no-render-narrative`: genera el video limpio de
  narracion del partido, con posesion, equipos y eventos como goles, tiros,
  cambios de posesion o colisiones.
- `--render-analysis / --no-render-analysis`: genera el video tecnico con
  cajas, scores, distancia de cada robot al balon, velocidad del balon,
  trayectoria y una probabilidad heuristica de presion de tiro hacia porteria.
- `--analysis-freeze`: solo para el video `analysis`; congela frames relevantes
  para explicar eventos importantes como tiros, goles candidatos, pases,
  intercepciones o colisiones.
- `--freeze-seconds`, `--freeze-cooldown-frames`, `--freeze-max-events` y
  `--freeze-event-types`: controlan duracion, separacion y tipos de eventos que
  pueden generar pausas analiticas.
- `--generate-game-state / --no-generate-game-state`: genera o desactiva los
  JSON de estado de juego dentro del pipeline.
- `--filter-by-game-state / --no-filter-by-game-state`: usa solo frames
  `in_play` para metricas, eventos deportivos y analisis de campo. Los tracks
  completos se conservan y, si el filtro esta activo, se crea tambien
  `*-in-play-tracks.jsonl`.
- `--game-state-out`, `--external-events-out` y `--game-segments-out`: rutas
  opcionales para los artefactos de estado de juego.
- `--game-state-missing-ball-frames`, `--robot-removed-after-frames`,
  `--robot-disabled-after-frames` y `--stationary-threshold-px`: tolerancias
  para pelota fuera/ausente, robot retirado y robot inmovil.
- `--no-render`: procesa sin generar video demo.
- `--no-qa`: desactiva QA automatico.
- `--run-report-out`: ruta opcional para el reporte Markdown integral de la
  corrida. Si no se indica, el pipeline escribe en `outputs\reports`.
- `--run-manifest-out`: ruta opcional para el manifiesto JSON reproducible de
  la corrida. Si no se indica, el pipeline escribe en `outputs\reports`. Incluye
  timestamp UTC, comando, argumentos, runtime, artefactos, huella Git local y
  SHA256 del codigo/config usado.

## Pipeline Con Homografia

Para calcular trayectoria en metros y mapa tactico:

```powershell
python -m samba_futbot.cli process-top-camera `
  --config config/default.yml `
  --video "outputs\review\2026-05-27\18abril_top_camera\clips\IMG_9938_f001799_10s.mp4" `
  --results-dir "outputs\review\2026-05-27\18abril_top_camera\runs" `
  --field-calibration config\top_camera_homography_template.yml `
  --render
```

Esto agrega:

- JSON de analisis de cancha.
- CSV de trayectoria de pelota.
- CSV de robots proyectados.
- CSV de control territorial por zona.
- PNG de mapa tactico con trayectoria de pelota, ocupacion por zonas, control
  territorial, mapa de calor de robots por equipo y robots coloreados por
  equipo.
- Candidatos reglamentarios: pelota fuera de campo, entradas a porteria,
  robots en area de penalizacion.
- Ocupacion de robots por equipo, zona, area reglamentaria y tercio
  defensivo/medio/ofensivo relativo al lado que defiende cada equipo.
- Indice de presion ofensiva por equipo, calculado como proporcion de muestras
  de robots en tercio ofensivo relativo.
- Control territorial por zona: para cada celda de la grilla se reporta lider,
  margen y proporcion de muestras por equipo.

Para generar solo el analisis desde tracks existentes:

```powershell
python -m samba_futbot.cli field-analysis `
  --tracks "outputs\tracks\clip-tracks.jsonl" `
  --calibration config\top_camera_homography_template.yml `
  --video "data\raw\video.mov" `
  --config config\default.yml `
  --out "outputs\field_analysis\clip-field-analysis.json" `
  --csv-out "outputs\field_analysis\clip-trajectory.csv" `
  --robot-csv-out "outputs\field_analysis\clip-robots.csv" `
  --zone-control-csv-out "outputs\field_analysis\clip-zone-control.csv" `
  --map-out "outputs\field_analysis\clip-field-map.png" `
  --fps 30
```

`--video` es opcional, pero conviene usarlo cuando los tracks vienen de una
corrida antigua sin `team`: recalcula `blue`/`yellow` desde la imagen antes de
proyectar robots al campo.

Para reasignar y auditar equipos sin repetir SAM3:

```powershell
python -m samba_futbot.cli assign-teams `
  --video "data\raw\video.mov" `
  --tracks "outputs\tracks\clip-tracks.jsonl" `
  --out "outputs\tracks\clip-tracks-with-teams.jsonl" `
  --config config\default.yml

python -m samba_futbot.cli team-quality `
  --tracks "outputs\tracks\clip-tracks-with-teams.jsonl" `
  --out "outputs\qa\clip-team-quality.json" `
  --report-out "outputs\qa\clip-team-quality.md"
```

`team-quality` mide cobertura, cambios de equipo dentro del mismo track,
tracks ambiguos y colapso hacia un solo color. Esta auditoria debe aprobarse
antes de defender posesion o control territorial por equipo.

Validar una calibracion antes de confiar en distancias y velocidades metricas:

```powershell
python -m samba_futbot.cli calibration-check `
  --calibration config\top_camera_homography_template.yml `
  --video "outputs\review\2026-05-27\18abril_top_camera\clips\IMG_9938_f001799_10s.mp4" `
  --out "outputs\field_analysis\calibration-quality.json"
```

El resultado incluye `status`, error de reproyeccion, area y cobertura del
poligono, orden/convexidad de esquinas, relacion de aristas, angulos comprimidos
y puntos fuera del frame. Estas comprobaciones detectan calibraciones muy
sesgadas aunque los cuatro puntos usados para ajustar la homografia tengan
error de reproyeccion cercano a cero.

## Comandos Individuales

Ejecutar SAM 3 directo:

```powershell
python -m samba_futbot.cli run-sam3 `
  --config config/default.yml `
  --video "data\raw\video.mov" `
  --out "outputs\detections\video"
```

Ejecutar SAM 3 por ventanas:

```powershell
python -m samba_futbot.cli run-sam3-sweep `
  --config config/default.yml `
  --video "data\raw\video.mov" `
  --out "outputs\detections\video-field-robots" `
  --prompt-frames "0,300,600" `
  --classes "field,robots"
```

Detectar pelota por color/forma:

```powershell
python -m samba_futbot.cli detect-orange-ball `
  --video "data\raw\video.mov" `
  --out "outputs\detections\video-orange-ball.jsonl" `
  --context-detections "outputs\detections\video-field-robots\detections.jsonl" `
  --color-profile orange
```

Aunque el comando conserva el nombre `detect-orange-ball` por compatibilidad,
puede cambiar el perfil HSV con `--color-profile`, `--hsv-lower` y
`--hsv-upper`. En `process-top-camera`, los equivalentes son
`--ball-color-profile`, `--ball-hsv-lower` y `--ball-hsv-upper`. La deteccion
final recomendada no sale de una sola fuente: sale de fusionar SAM3 pelota +
color/forma + refinamiento temporal.

Refinar pelota:

```powershell
python -m samba_futbot.cli refine-ball `
  --detections "outputs\detections\video\detections.jsonl" `
  --out "outputs\detections\video\detections-refined.jsonl"
```

Asignar tracks:

```powershell
python -m samba_futbot.cli track `
  --detections "outputs\detections\video\detections-refined.jsonl" `
  --out "outputs\tracks\video-tracks.jsonl"
```

Metricas:

```powershell
python -m samba_futbot.cli metrics `
  --tracks "outputs\tracks\video-tracks.jsonl" `
  --out "outputs\metrics\video-metrics.json" `
  --fps 30
```

Eventos:

```powershell
python -m samba_futbot.cli events `
  --tracks "outputs\tracks\video-tracks.jsonl" `
  --out "outputs\events\video-events.json" `
  --summary-out "outputs\events\video-event-summary.json"
```

Resumen de eventos desde un archivo existente:

```powershell
python -m samba_futbot.cli event-summary `
  --events "outputs\events\video-events.json" `
  --out "outputs\events\video-event-summary.json"
```

Analisis situacional avanzado:

```powershell
python -m samba_futbot.cli situation-analysis `
  --tracks "outputs\tracks\video-tracks.jsonl" `
  --out "outputs\events\video-situations.json" `
  --frame-width 1080
```

Este JSON reporta por frame distancia robot-balon, estado de posesion
`controlled/disputed/free`, riesgo de perder el balon y probabilidades
heuristicas de pase, tiro y mantener posesion.

Estado de juego y eventos externos:

```powershell
python -m samba_futbot.cli game-state `
  --tracks "outputs\tracks\video-tracks.jsonl" `
  --out "outputs\events\video-game-state.json" `
  --events-out "outputs\events\video-external-events.json" `
  --segments-out "outputs\events\video-game-segments.json"
```

Este comando marca frames/segmentos `in_play`, `dead_ball` y
`human_intervention`, y genera candidatos de `robot_removed` y
`robot_disabled`. Es una primera capa heuristica para separar juego real de
pausas, intervenciones y mantenimiento antes de defender metricas finales.

`process-video` y `process-top-camera` ya ejecutan esta capa automaticamente por
defecto. En cada corrida se agregan artefactos como:

- `events/*-game-state.json`: estados por frame, segmentos y eventos externos.
- `events/*-game-segments.json`: segmentos compactos para inspeccion rapida.
- `events/*-external-events.json`: intervenciones humanas, pausas y robots
  retirados/detenidos.
- `tracks/*-in-play-tracks.jsonl`: detecciones filtradas para metricas y
  analisis cuando `--filter-by-game-state` esta activo.

Luego puedes usar ese JSON para filtrar comandos posteriores:

```powershell
python -m samba_futbot.cli metrics `
  --tracks "outputs\tracks\video-tracks.jsonl" `
  --game-state "outputs\events\video-game-state.json" `
  --out "outputs\metrics\video-in-play-metrics.json"
```

`events` y `field-analysis` tambien aceptan `--game-state`, de modo que tiros,
posesion, velocidad metrica y trayectoria pueden calcularse solo con frames
marcados como `in_play`.

Render demo narrativo:

```powershell
python -m samba_futbot.cli render-demo `
  --video "data\raw\video.mov" `
  --tracks "outputs\tracks\video-tracks.jsonl" `
  --events "outputs\events\video-events.json" `
  --out "outputs\videos\video-narrative-demo.mp4" `
  --style narrative `
  --max-seconds 120
```

Render demo de analisis:

```powershell
python -m samba_futbot.cli render-demo `
  --video "data\raw\video.mov" `
  --tracks "outputs\tracks\video-tracks.jsonl" `
  --events "outputs\events\video-events.json" `
  --out "outputs\videos\video-analysis-demo.mp4" `
  --style analysis `
  --analysis-freeze `
  --max-seconds 120
```

El estilo `narrative` evita saturar la pantalla y sirve para presentar lo que
ocurre en el partido. El estilo `analysis` muestra evidencia tecnica: equipos,
confianza, distancia robot-balon, velocidad de pelota, trayectoria y presion de
tiro estimada.

Reporte Markdown:

```powershell
python -m samba_futbot.cli summarize-run `
  --title "Clip camara superior" `
  --metrics "outputs\metrics\clip-metrics.json" `
  --events "outputs\events\clip-events.json" `
  --field-analysis "outputs\field_analysis\clip-field-analysis.json" `
  --qa "outputs\qa\clip-qa.json" `
  --field-map "outputs\field_analysis\clip-field-map.png" `
  --demo "outputs\videos\clip-demo.mp4" `
  --out "outputs\reports\clip-report.md"
```

QA automatico:

```powershell
python -m samba_futbot.cli qa-run `
  --metrics "outputs\metrics\clip-metrics.json" `
  --events "outputs\events\clip-events.json" `
  --field-analysis "outputs\field_analysis\clip-field-analysis.json" `
  --max-unknown-team-ratio 0.35 `
  --out "outputs\qa\clip-qa.json" `
  --report-out "outputs\qa\clip-qa.md"
```

Para ordenar varias corridas por calidad y escoger que revisar primero:

```powershell
python -m samba_futbot.cli qa-index `
  --root "outputs\review" `
  --out "outputs\review\qa-index.json" `
  --report-out "outputs\review\qa-index.md"
```

Para elegir candidatos de demo final por claims listos:

```powershell
python -m samba_futbot.cli showcase-index `
  --root "outputs\review" `
  --out "outputs\review\showcase-index.json" `
  --report-out "outputs\review\showcase-index.md" `
  --required-claims "ball_tracking,team_possession" `
  --limit 12
```

Para comparar una corrida baseline contra una variante o modelo adaptado:

```powershell
python -m samba_futbot.cli compare-qa `
  --baseline "outputs\qa\baseline-qa.json" `
  --candidate "outputs\qa\candidate-qa.json" `
  --out "outputs\qa\baseline-vs-candidate.json" `
  --report-out "outputs\qa\baseline-vs-candidate.md"
```

El comparador evalua score, coberturas, saltos de pelota, incertidumbre de
equipos, homografia y claims ganados/perdidos. Debe usarse para decidir si una
variante de prompts o un fine-tuning realmente mejora al baseline.

Para crear un reporte Markdown final desde un batch procesado:

```powershell
python -m samba_futbot.cli submission-report `
  --batch-root "outputs\review\2026-06-08\top_camera_batch" `
  --training-root "outputs\review\2026-06-08\training_datasets" `
  --out "outputs\review\2026-06-08\SUBMISSION_EVIDENCE.md" `
  --top 4
```

Ese reporte consolida candidatos de showcase, rutas de videos narrativo y
analitico, QA, resumen de batch, capa tactica con distancias/posesion y estado
del dataset preparado para adaptacion compatible con SAM.

## Resultados Generados

Estado de avance y estimado de cierre:

- `docs/PROJECT_STATUS.md`
- `docs/REMOTE_TEST_CHECKLIST.md`

Cada corrida puede producir:

- `detections.jsonl`: una deteccion por linea, con clase, score, caja, mascara y frame.
- `detections-refined.jsonl`: detecciones despues de elegir trayectoria de pelota.
- `tracks.jsonl`: detecciones con IDs temporales.
- `metrics.json`: cobertura, fragmentacion, velocidades y conteos.
  Incluye posesion por equipo, dominancia de posesion y racha mas larga de
  posesion.
- `events.json`: tiros, pases, posesion, colisiones o goles candidatos.
- `event-summary.json`: marcador candidato por equipo, goles por porteria,
  pases, intercepciones, tiros y colisiones.
- `field-analysis.json`: coordenadas metricas, zonas, velocidad en m/s y reglas.
- `trajectory.csv`: trayectoria de pelota en metros.
- `robots.csv`: posiciones proyectadas de robots.
- `zone-control.csv`: lider, margen y proporcion de control por celda.
- `field-map.png`: mapa tactico de la cancha con pelota, zonas, control
  territorial y robots por equipo.
- `demo.mp4`: video con overlays, trails, posesion y evento reciente cuando se
  proporciona `events.json`.
- `qa.json` / `qa.md`: evaluacion automatica de calidad, incluyendo cobertura
  de pelota/campo, saltos, reglas e incertidumbre de equipos.
- `qa-index.json` / `qa-index.md`: ranking de corridas QA encontradas bajo una
  carpeta de resultados, incluyendo incertidumbre de equipos.
- `quality.json` / `quality.md`: auditoria de dataset exportado antes de
  adaptar/entrenar, con cajas invalidas, scores bajos, duplicados por
  video/frame y ejemplos a revisar.
- `curated-manifest.json` / `curation-report.json`: dataset filtrado y
  deduplicado junto con la trazabilidad de cada descarte.
- `human-holdout.json` / `human-holdout-report.json`: plantilla independiente
  para anotacion humana y huella reproducible de seleccion.
- COCO `annotations/*.json`: cajas y, cuando existen, mascaras RLE auditadas.
- `report.md`: reporte integral de corrida con metricas, eventos, homografia,
  QA, demo y mapa tactico cuando esos artefactos existen.
- `manifest.json`: timestamp UTC, comando, argumentos, runtime, huella Git
  local, SHA256 del codigo/config, rutas de artefactos y resumenes clave de la
  corrida.

Para mejorar legibilidad de videos ya procesados sin repetir SAM3, `render-demo`
puede reusar tracks/eventos y dibujar overlays semitransparentes:

```powershell
python -m samba_futbot.cli render-demo `
  --video "outputs\review\2026-05-27\18abril_top_camera\clips\IMG_9938_f001799_10s.mp4" `
  --tracks "outputs\review\2026-05-27\18abril_top_camera\runs\tracks\IMG_9938_f001799_10s-top-fusion-hsv-v2-refined-tracks.jsonl" `
  --events "outputs\review\2026-06-08\top_camera_batch\events\IMG_9938_f001799_10s-top-fusion-hsv-v2-refined-events.json" `
  --out "outputs\review\2026-06-15\visual_refresh\videos\IMG_9938_f001799_10s-readable-analysis-mask-demo.mp4" `
  --style analysis `
  --analysis-freeze `
  --freeze-seconds 3.5 `
  --mask-overlay `
  --mask-alpha 0.38 `
  --label-scale 0.9 `
  --box-thickness 4 `
  --visual-hold-frames 18
```

Cuando una deteccion trae `mask_path` y `mask_index`, el overlay usa la mascara
real; si no existe una mascara valida, rellena la caja con el color de la clase.
`visual-hold-frames` solo conserva tracks durante perdidas breves: no inventa
robots cuando el archivo de tracks ya no tiene detecciones durante muchos
segundos.

Cuando SAM3 pierde robots en vista superior, se puede crear un preview de
recuperacion por color/forma sin usar detectores externos:

```powershell
python -m samba_futbot.cli detect-dark-robots `
  --video "outputs\review\2026-05-27\18abril_top_camera\clips\IMG_9938_f001799_10s.mp4" `
  --out "outputs\review\2026-06-15\robot_recovery\IMG_9938_f001799_10s-dark-robots-lower-half-mergedparts.jsonl" `
  --field-detections "outputs\review\2026-05-27\18abril_top_camera\runs\detections\IMG_9938_f001799_10s-field-robots-sweep-clipped\detections.jsonl" `
  --min-area 800 `
  --max-area 18000 `
  --min-circularity 0.30 `
  --hsv-upper "179,255,125" `
  --min-center-y-ratio 0.38 `
  --merge-distance-px 42 `
  --max-per-frame 4
```

Este paso es un post-procesamiento auditable para recuperar blobs oscuros de
robots sobre el campo. Debe revisarse visualmente antes de defender posesion o
conteo de robots, porque sombras y soportes de porteria pueden parecer robots
si no se restringe la zona o la forma.

Tambien puede activarse dentro del pipeline de camara superior:

```powershell
python -m samba_futbot.cli process-top-camera `
  --config config/default.yml `
  --video "outputs\review\2026-05-27\18abril_top_camera\clips\IMG_9938_f001799_10s.mp4" `
  --results-dir "outputs\review\2026-06-15\final_top_camera_robot_recovery" `
  --suffix "top-final-robot-recovery-v1" `
  --robot-color-recovery `
  --robot-recovery-min-area 800 `
  --robot-recovery-min-circularity 0.30 `
  --robot-recovery-hsv-upper "179,255,125" `
  --robot-recovery-min-center-y-ratio 0.38 `
  --robot-recovery-merge-distance-px 42 `
  --robot-recovery-max-per-frame 4 `
  --render-analysis `
  --analysis-freeze `
  --mask-overlay `
  --label-scale 0.82 `
  --box-thickness 4
```

La opcion integrada escribe `*-dark-robots.jsonl` junto a las otras detecciones
y lo incluye antes de tracking/QA. Mantenerla apagada por defecto permite
comparar baseline SAM3 contra la variante recuperada.

## Pruebas

En Windows, si `pytest` no esta instalado, se puede usar `unittest`:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests
```

En un entorno con dependencias de desarrollo:

```powershell
pytest
```

## Documentacion Extendida

- `docs/TECHNICAL_WALKTHROUGH.md`: explicacion larga del codigo y despliegue.
- `docs/RESULTS.md`: resultados parciales procesados.
- `docs/SAM3_FINETUNING.md`: contrato de datos, ejecucion y evaluacion para
  adaptar SAM3 con el repositorio oficial.

## Licencia

Apache-2.0. Ver `LICENSE`.

Este proyecto usa o esta preparado para usar SAM 3 de Meta, Hugging Face,
OpenCV, NumPy, pandas y Pillow. Hay que respetar las licencias de dependencias,
checkpoints y datos fuente.
