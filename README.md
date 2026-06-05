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
8. Se refina la trayectoria de la pelota con consistencia temporal.
9. Se asignan IDs de tracking y equipo por color dominante del robot.
10. Se calculan metricas, posesion por equipo, eventos, homografia, reportes y QA.

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
  PROFESSIONAL_STRATEGY.md            Estrategia tecnica para categoria profesional.
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
- `goal_detection.require_seed_for_color`: modo conservador. Si esta activo,
  una clase de porteria necesita al menos una caja semilla de SAM3 antes de
  aceptar detecciones cromaticas de esa clase.
- `goal_detection.require_field_overlap`: exige que la porteria aceptada este
  sobre o tocando el campo detectado. Con `max_per_frame_per_class: 1`, el
  pipeline conserva como maximo una porteria azul y una amarilla por frame.
- `analysis.possession_radius_px`: distancia robot-pelota para contar posesion.
- `analysis.in_play_field_margin_px`: margen para decidir si la pelota esta en juego.
- `analysis.ball_border_margin_px`: filtro contra falsos positivos en bordes.

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
2. Ejecuta SAM 3 para `field,robots,goal_blue,goal_yellow`.
3. Ejecuta SAM 3 para `ball` con prompts semanticos.
4. Detecta pelota por color/forma usando el perfil HSV configurado.
5. Si SAM3 encontro porterias, calibra el HSV real desde esas cajas.
6. Agrega porterias por color azul/amarillo si el fallback cromatico esta
   activo.
7. Fusiona campo, robots, porterias y candidatos de pelota.
8. Refina la pelota con `refine-ball`.
9. Genera tracks y asigna equipo de robots por color.
10. Calcula metricas y posesion por equipo.
11. Detecta eventos candidatos, incluyendo goles visuales si hay porteria.
12. Renderiza demo MP4.
13. Genera QA automatico.

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
  territorial y robots coloreados por equipo.
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

Validar una calibracion antes de confiar en distancias y velocidades metricas:

```powershell
python -m samba_futbot.cli calibration-check `
  --calibration config\top_camera_homography_template.yml `
  --video "outputs\review\2026-05-27\18abril_top_camera\clips\IMG_9938_f001799_10s.mp4" `
  --out "outputs\field_analysis\calibration-quality.json"
```

El resultado incluye `status`, error de reproyeccion, area del poligono de campo
en pixeles y puntos fuera del frame.

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

Render demo:

```powershell
python -m samba_futbot.cli render-demo `
  --video "data\raw\video.mov" `
  --tracks "outputs\tracks\video-tracks.jsonl" `
  --events "outputs\events\video-events.json" `
  --out "outputs\videos\video-demo.mp4" `
  --max-seconds 120
```

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

## Resultados Generados

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
- `report.md`: reporte integral de corrida con metricas, eventos, homografia,
  QA, demo y mapa tactico cuando esos artefactos existen.
- `manifest.json`: timestamp UTC, comando, argumentos, runtime, huella Git
  local, SHA256 del codigo/config, rutas de artefactos y resumenes clave de la
  corrida.

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
- `docs/PROFESSIONAL_STRATEGY.md`: estrategia de innovacion para la categoria.
- `docs/RESULTS.md`: resultados parciales procesados.

## Licencia

Apache-2.0. Ver `LICENSE`.

Este proyecto usa o esta preparado para usar SAM 3 de Meta, Hugging Face,
OpenCV, NumPy, pandas y Pillow. Hay que respetar las licencias de dependencias,
checkpoints y datos fuente.
