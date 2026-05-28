# SAMBA FutBot Technical Walkthrough

Este documento explica como funciona el proyecto, que hace cada archivo
principal, como interpretar la configuracion y como desplegar/probar el
pipeline en Windows o en una maquina con GPU.

## 1. Objetivo Del Sistema

SAMBA FutBot es un pipeline reproducible para la categoria profesional del reto
FutBotMX Vision por Computadora. El objetivo tecnico es:

- descargar o indexar videos del Drive oficial;
- ejecutar SAM 3 sobre videos de futbol robotico;
- segmentar campo, robots y pelota;
- reparar/consolidar tracking;
- calcular metricas;
- detectar eventos deportivos basicos;
- renderizar videos demo con overlays y trails;
- dejar resultados reproducibles en `outputs/`.

La idea profesional diferenciadora es usar SAM 3 con contexto temporal:

- prompts por clase;
- ensamble de prompts para pelota naranja;
- ventanas ancladas en frames especificos;
- clips fisicos por ventana para controlar VRAM;
- tracking nativo de SAM 3 mas reparacion IoU;
- post-procesamiento geometrico y metricas deportivas.

## 2. Estructura General

```text
config/default.yml          Configuracion principal del pipeline.
docs/RETO.md                Resumen local de requisitos del reto.
docs/PROFESSIONAL_STRATEGY.md Estrategia profesional.
docs/RESULTS.md             Tabla de videos ya procesados.
src/samba_futbot/           Codigo Python del paquete.
tests/                      Pruebas unitarias.
outputs/                    Resultados generados.
data/                       Manifest, videos descargados y frames de debug.
```

`outputs/` contiene artefactos generados. Algunos estan versionados porque ya
los subimos como resultados; los `.mp4` se manejan con Git LFS.

## 3. Configuracion Principal

Archivo: `config/default.yml`

```yaml
project:
  name: SAMBA FutBot
  category: Profesional
  drive_root_id: 1TF7-P4rAwPmHFw_TjmNfFU3ORxqnp8CD
```

- `name`: nombre interno del proyecto.
- `category`: categoria del reto.
- `drive_root_id`: ID de la carpeta publica de Google Drive con videos.

```yaml
data:
  manifest: data/manifests/drive_index.json
  raw_dir: data/raw
  frame_dir: data/frames
```

- `manifest`: archivo JSON donde se guarda el indice del Drive.
- `raw_dir`: carpeta local para videos crudos descargados.
- `frame_dir`: carpeta para frames de calibracion/debug.

```yaml
sam3:
  model_id: facebook/sam3
  backend: official
  max_frames: 300
  prompt_frame_index: 0
  stride: 2
  threshold: 0.45
  mask_threshold: 0.5
  use_fa3: false
  offload_video_to_cpu: true
  offload_state_to_cpu: true
```

- `model_id`: modelo base. Se usa `facebook/sam3` porque fue el baseline mas
  estable. `facebook/sam3.1` queda para experimentos.
- `backend`: backend de ejecucion. `official` usa la implementacion oficial de
  SAM 3.
- `max_frames`: tamano base de ventana.
- `prompt_frame_index`: frame donde se inserta el prompt cuando se usa
  `run-sam3`.
- `stride`: reservado para backend Transformers.
- `threshold`: umbral de score general.
- `mask_threshold`: umbral de mascara para backend Transformers.
- `use_fa3`: Flash Attention 3 apagado por compatibilidad.
- `offload_video_to_cpu`: baja video a CPU para ahorrar VRAM.
- `offload_state_to_cpu`: baja estado de tracking a CPU para ahorrar VRAM.

```yaml
  prompts:
    field:
      - green soccer field
    robots:
      - robot
    ball:
      - small orange ball
      - orange ball
      - pelota naranja
```

- `field`: prompt para detectar la cancha.
- `robots`: prompt general para robots.
- `ball`: ensamble de prompts. El prompt mas fuerte en pruebas fue
  `small orange ball`; `pelota naranja` se deja como variante de contexto.

```yaml
tracking:
  iou_threshold: 0.25
  max_age: 12
  min_area:
    ball: 10
    robots: 40
    field: 1000
```

- `iou_threshold`: umbral IoU para asociar detecciones entre frames.
- `max_age`: cuantos frames puede sobrevivir un track sin deteccion.
- `min_area`: filtro minimo por clase, pensado para quitar ruido.

```yaml
analysis:
  fps: 30
  possession_radius_px: 90
  collision_radius_px: 55
  goal_x_margin_ratio: 0.08
```

- `fps`: FPS asumido si no se lee metadata.
- `possession_radius_px`: distancia maxima robot-pelota para posesion.
- `collision_radius_px`: distancia maxima entre robots para posible colision.
- `goal_x_margin_ratio`: margen lateral usado para zonas de gol.

```yaml
visualization:
  overlay_alpha: 0.45
  trail_length: 45
  output_fps: 30
```

- `trail_length`: longitud de trails dibujados.
- `overlay_alpha` y `output_fps`: preparados para visualizaciones futuras.

## 4. Paquete Python

### `src/samba_futbot/cli.py`

Es el punto de entrada. Define comandos de terminal con `argparse`.

Comandos principales:

- `index-drive`: indexa el Drive publico.
- `download`: descarga un archivo por ID o nombre.
- `download-all`: descarga todos los videos del manifest.
- `sample-frames`: extrae frames para inspeccion.
- `run-sam3`: corre SAM 3 una vez sobre un video.
- `run-sam3-sweep`: corre SAM 3 por ventanas/anclas.
- `merge-detections`: fusiona detecciones JSONL.
- `filter-detections`: filtra falsos positivos geometricos.
- `detect-orange-ball`: detector HSV/color/forma para pelota naranja.
- `refine-ball`: elige una trayectoria temporal coherente entre multiples
  candidatos de pelota.
- `process-video`: pipeline completo por video.
- `process-top-camera`: pipeline especializado para camara superior.
- `track`: aplica tracker IoU.
- `events`: genera eventos deportivos basicos.
- `metrics`: resume tracks.
- `render-demo`: renderiza video lado a lado.
- `video-info`: imprime metadata de video.

Flujo importante: `process-video`.

1. Lee metadata con `video_info`.
2. Calcula anchors de campo/robots y pelota.
3. Llama internamente a `run-sam3-sweep` para campo/robots.
4. Llama internamente a `run-sam3-sweep` para pelota.
5. Fusiona resultados.
6. Aplica tracker IoU.
7. Calcula metricas enriquecidas.
8. Detecta eventos deportivos candidatos.
9. Renderiza demo MP4.
10. Imprime JSON con rutas y resumen.

Este comando existe para no tener que ejecutar manualmente 5 o 6 comandos por
video.

Ejemplo:

```powershell
samba-futbot process-video `
  --config config/default.yml `
  --video "ruta\al\video.mov" `
  --results-dir outputs `
  --clip-windows `
  --render
```

Flujo importante: `process-top-camera`.

Este es el flujo recomendado para clips de camara superior donde la pelota
naranja es pequena y SAM 3 puede saltarsela en algunos frames. La estrategia es
deliberadamente hibrida:

1. Lee metadata con `video_info`.
2. Calcula anchors de campo/robots.
3. Ejecuta SAM 3 solo para `field,robots`, donde el modelo es estable.
4. Detecta la pelota naranja con HSV, area y circularidad.
5. Usa las detecciones de robots para descartar manchas naranjas dentro de
   robots.
6. Fusiona campo/robots con candidatos de pelota.
7. Aplica refinamiento temporal por programacion dinamica.
8. Corre tracking, metricas, eventos y demo.

Ejemplo recomendado para un clip de 10 segundos de camara superior:

```powershell
samba-futbot process-top-camera `
  --config config/default.yml `
  --video "outputs\review\2026-05-27\18abril_top_camera\clips\IMG_9938_f001799_10s.mp4" `
  --results-dir "outputs\review\2026-05-27\18abril_top_camera\runs" `
  --render
```

Parametros clave de esa ruta:

- `--orange-min-area 300`: descarta reflejos o marcas naranjas demasiado
  pequenas.
- `--orange-max-per-frame 6`: conserva varios candidatos antes del refinamiento.
- `--refine-max-jump-px 35`: penaliza saltos imposibles de la pelota entre
  frames.
- `--refine-preferred-area 680`: sesga la seleccion hacia el tamano esperado de
  la pelota en esta vista.

### `src/samba_futbot/config.py`

Funciones de configuracion:

- `load_config(path)`: carga YAML.
- `deep_get(data, dotted_path)`: lee claves anidadas como
  `project.drive_root_id`.

Este modulo evita meter rutas y parametros fijos dentro del codigo.

### `src/samba_futbot/drive.py`

Maneja Google Drive publico sin API key.

Funciones relevantes:

- indexa carpetas publicas;
- crea manifest local;
- descarga archivos;
- permite descargar todos los videos filtrando extensiones.

El objetivo es que la descarga de datos sea reproducible sin depender de pasos
manuales.

### `src/samba_futbot/sam3_adapter.py`

Es el adaptador entre nuestro pipeline y SAM 3.

Puntos clave:

- `flatten_prompt_config`: convierte prompts del YAML en pares
  `(clase, prompt)`.
- `run_sam3_video`: decide si usar backend oficial o Transformers.
- `_run_official_sam3`: usa el predictor oficial de SAM 3.
- `_track_count`: corrige la diferencia entre frame final absoluto y cantidad
  de frames a propagar.
- `_detections_from_processed`: convierte salidas SAM 3 a nuestro tipo
  `Detection`.

Por que importa:

- SAM 3 puede devolver objetos/masks en formatos distintos.
- Este archivo normaliza todo a una estructura propia.
- Tambien agrega compatibilidad con `prompt_frame_index`, que nos permite
  lanzar prompts desde frames donde la pelota se ve mejor.

### `src/samba_futbot/windowing.py`

Utilidades para ventanas y fusion de detecciones.

Funciones:

- `parse_int_list`: lee listas tipo `"0,150,300"`.
- `deduplicate_detections`: quita duplicados por frame/clase usando IoU.
- `merge_detection_files`: une varios JSONL y deduplica.
- `offset_detections`: corrige indices de frame cuando SAM corre sobre clips.
- `write_window_manifest`: guarda metadata de ventanas procesadas.

La funcion mas estrategica es `offset_detections`: cuando cortamos un clip
desde frame 300, SAM cree que el primer frame es 0; esta funcion vuelve a sumar
300 para que los resultados queden alineados al video original.

### `src/samba_futbot/video.py`

Utilidades de video con OpenCV.

Funciones:

- `require_cv2`: valida que OpenCV este instalado.
- `video_info`: lee FPS, frames, ancho, alto y duracion.
- `extract_video_clip`: crea un MP4 corto de una ventana.
- `sample_frames`: guarda frames cada cierto stride o segundos.

`extract_video_clip` fue clave para evitar OOM en GPU: SAM 3 cargaba el video
completo, aunque solo propagaramos una ventana logica. Al crear clips fisicos,
SAM solo carga 220 o 300 frames.

### `src/samba_futbot/types.py`

Define estructuras de datos.

`Detection` contiene:

- `frame_index`: frame original.
- `class_name`: `field`, `robots`, `ball`, etc.
- `score`: confianza.
- `box`: caja `(x1, y1, x2, y2)`.
- `prompt`: prompt que genero la deteccion.
- `object_id`: ID nativo de SAM.
- `track_id`: ID reparado por tracker.
- `team`: equipo, preparado para aliados/rivales.
- `mask_path`: ruta de mascara si se guarda.
- `area`: area de mascara.
- `extra`: metadata adicional.

`Detection.to_record()` convierte el objeto a JSON serializable.
`Detection.from_record()` reconstruye el objeto desde JSON.

`Event` representa eventos deportivos.

### `src/samba_futbot/tracking.py`

Tracker IoU simple.

Idea:

1. Agrupa detecciones por frame.
2. Para cada clase, intenta asociar detecciones nuevas con tracks activos.
3. Usa IoU entre cajas.
4. Si no encuentra match, crea track nuevo.
5. Si un track no aparece durante `max_age`, lo deja morir.

No es tan sofisticado como ByteTrack, pero es reproducible y suficiente para
reparar IDs en muchos casos.

### `src/samba_futbot/metrics.py`

Resume detecciones y tracks.

Calcula:

- frames observados;
- primer/ultimo frame;
- total de detecciones;
- numero de tracks;
- largo medio de track;
- gaps/fragmentacion;
- detecciones por clase;
- tracks unicos por clase;
- score medio;
- cobertura por clase;
- fragmentacion por clase;
- velocidad media/maxima de la pelota;
- estadisticas de area.

Se usa para `outputs/metrics/*.json`.

### `src/samba_futbot/events.py`

Eventos deportivos basicos.

Funciones:

- posesion aproximada: robot mas cercano a pelota;
- colisiones: robots cercanos;
- tiros: pelota moviendose hacia margen de gol.

Estado actual:

- funcional como base;
- todavia necesita fortalecerse para entrega profesional;
- debe combinarse con separacion aliados/rivales para pases/intercepciones.

### `src/samba_futbot/team.py`

Modulo inicial para clasificar equipo por color.

Estado:

- preparado como base;
- no es todavia el centro del pipeline.

Siguiente mejora sugerida:

- muestrear color dominante dentro de bbox o mascara;
- asignar `team=allied/rival/unknown`;
- reflejarlo en overlays y eventos.

### `src/samba_futbot/visualize.py`

Renderiza demos.

`render_demo_video`:

1. Lee tracks JSONL.
2. Agrupa detecciones por frame.
3. Abre video original.
4. Crea un video de salida con ancho doble.
5. Izquierda: original.
6. Derecha: anotado.
7. Dibuja cajas, labels y trails.

Colores actuales:

- `field`: verde.
- `robots`: amarillo/naranja.
- `ball`: naranja/azul BGR segun conversion.
- `robot_allied` y `robot_rival`: reservados para equipos.

### `src/samba_futbot/io_utils.py`

Lectura/escritura.

Funciones:

- `write_json`
- `read_json`
- `write_jsonl`
- `read_jsonl`
- `read_detections`
- `write_detections`
- `write_events`

Centralizar esto evita repetir parsing JSON en todos los modulos.

### `src/samba_futbot/geometry.py`

Funciones geometricas simples:

- distancias;
- centroides;
- utilidades para analisis espacial.

Sirve de base para posesion, colisiones y velocidad.

## 5. Resultados Y Artefactos

Carpetas relevantes:

```text
outputs/detections/   Detecciones SAM 3 fusionadas.
outputs/tracks/       Detecciones con track_id.
outputs/metrics/      Resumen cuantitativo.
outputs/events/       Eventos deportivos candidatos.
outputs/videos/       Demos MP4 con overlay.
outputs/videos/qa_frames/ Frames de control visual.
```

Los MP4 se subieron con Git LFS. Los videos crudos del Drive no se suben.

## 6. Despliegue En Windows

Desde PowerShell:

```powershell
cd D:\Repositorios\SAMBA_FutBot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .[dev]
```

Para solo usar resultados, no hace falta SAM 3. Para ejecutar SAM 3, tambien se
necesitan dependencias del entorno GPU.

Pruebas:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests
```

Ver metadata de un video:

```powershell
samba-futbot video-info --video "ruta\video.mov"
```

Procesar video completo:

```powershell
samba-futbot process-video `
  --config config/default.yml `
  --video "ruta\video.mov" `
  --results-dir outputs `
  --clip-windows `
  --render
```

Procesar clip de camara superior con la ruta recomendada actual:

```powershell
samba-futbot process-top-camera `
  --config config/default.yml `
  --video "ruta\clip_camara_superior.mp4" `
  --results-dir outputs `
  --render
```

## 7. Despliegue En Maquina GPU

La maquina GPU debe tener:

- Python 3.12;
- CUDA funcionando;
- PyTorch con CUDA;
- SAM 3 instalado;
- entorno virtual activo;
- repo/copias del proyecto.

Flujo tipico:

```bash
cd /ruta/SAMBA_FutBot
source /ruta/venv/bin/activate
PYTHONPATH=src python -m unittest discover -s tests
samba-futbot process-video \
  --config config/default.yml \
  --video "/ruta/video.mov" \
  --results-dir outputs \
  --clip-windows \
  --render
```

Para clips de camara superior, usar la variante hibrida:

```bash
samba-futbot process-top-camera \
  --config config/default.yml \
  --video "/ruta/clip_camara_superior.mp4" \
  --results-dir outputs \
  --render
```

Para lotes:

```bash
for VIDEO in /ruta/videos/*.mov; do
  samba-futbot process-video \
    --config config/default.yml \
    --video "$VIDEO" \
    --results-dir outputs \
    --clip-windows \
    --render
done
```

Para un lote de clips ya recortados desde la camara superior:

```bash
for VIDEO in /ruta/clips_camara_superior/*.mp4; do
  samba-futbot process-top-camera \
    --config config/default.yml \
    --video "$VIDEO" \
    --results-dir outputs \
    --render
done
```

## 8. Flujo Completo Reproducible

### Paso 1: indexar Drive

```powershell
samba-futbot index-drive --config config/default.yml
```

Salida:

```text
data/manifests/drive_index.json
```

### Paso 2: descargar videos

```powershell
samba-futbot download-all `
  --manifest data/manifests/drive_index.json `
  --out-dir data/raw `
  --strip-root
```

### Paso 3: procesar un video

```powershell
samba-futbot process-video `
  --video "data/raw/Meta_Glasses/.../video.mov" `
  --results-dir outputs
```

Para camara superior, primero se recomienda recortar clips utiles y luego:

```powershell
samba-futbot process-top-camera `
  --video "outputs\review\2026-05-27\18abril_top_camera\clips\IMG_9938_f001799_10s.mp4" `
  --results-dir "outputs\review\2026-05-27\18abril_top_camera\runs" `
  --render
```

### Paso 4: revisar resultados

```text
outputs/videos/*demo.mp4
outputs/metrics/*metrics.json
outputs/videos/qa_frames/*.jpg
```

## 9. Que Tenemos Cubierto Del Reto

Cubierto:

- uso de SAM 3;
- segmentacion campo/robots/pelota;
- tracking;
- metricas cuantitativas;
- demos MP4;
- resultados con Git LFS;
- procesamiento por clips para videos largos;
- prompt engineering y contexto temporal;
- flujo hibrido de camara superior con SAM3 + HSV + refinamiento temporal.

En progreso/falta reforzar:

- aliados vs rivales;
- eventos deportivos fuertes;
- visualizacion tipo dashboard/panel;
- demo final de maximo 2 minutos;
- reel publico de 30 segundos;
- README final de entrega.

## 10. Como Actualizar GitHub Manualmente

Como regla de trabajo actual: los cambios quedan primero en Windows y la subida
la haces tu.

Ver cambios:

```powershell
git status -sb
git diff --stat
```

Agregar cambios normales:

```powershell
git add docs/TECHNICAL_WALKTHROUGH.md
```

Si agregas resultados ignorados en `outputs/`, usar `-f`:

```powershell
git add -f outputs/metrics outputs/detections outputs/tracks outputs/videos/qa_frames
```

Si agregas MP4, confirmar Git LFS:

```powershell
git lfs status
git add -f outputs/videos/nombre-demo.mp4
```

Commit y push:

```powershell
git commit -m "Update technical documentation"
git push
```

## 11. Lectura Rapida Para Defender El Proyecto

Frase corta:

> SAMBA FutBot usa SAM 3 con ensamble de prompts, ventanas temporales ancladas,
> clips fisicos para controlar VRAM, tracking hibrido SAM3 + IoU y metricas
> deportivas para analizar videos de futbol robotico.

Frase tecnica:

> En lugar de ejecutar SAM 3 una sola vez sobre videos completos, el pipeline
> divide el video en ventanas de contexto, elige anchors temporales para objetos
> pequenos como la pelota, reindexa detecciones al timeline original, fusiona
> resultados con deduplicacion IoU y genera tracks/metricas reproducibles.

Frase de innovacion:

> La innovacion esta en la ingenieria de contexto y post-procesamiento alrededor
> de SAM 3: prompt ensemble, temporal anchor sweep, clipped video context,
> hybrid tracking y analisis geometrico.
