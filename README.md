# SAMBA FutBot

Pipeline reproducible para la categoria profesional del reto **Copa FutBotMX,
capitulo Vision por Computadora**. El objetivo es segmentar y rastrear campo,
robots y balon en videos de futbol robotico usando SAM 3, y producir analisis
de juego con visualizaciones y video demo.

## Enfoque

La solucion esta organizada como un pipeline de investigacion reproducible:

1. Indexar y descargar videos publicos del Drive oficial.
2. Ejecutar SAM 3/SAM 3.1 con prompts de concepto para campo, robots y balon.
3. Consolidar detecciones con tracking por identidad, clases y color de equipo.
4. Analizar posesion, pases, intercepciones, tiros y colisiones.
5. Renderizar overlays, trails, mapas de calor y un video demo de maximo 2 min.

Para la categoria profesional, la innovacion propuesta combina:

- **Prompt engineering avanzado:** ensamble de prompts por clase, filtros de area,
  prior espacial y refinamiento por puntos/cajas cuando sea necesario.
- **SAM 3.1 Object Multiplex:** por defecto se usa `facebook/sam3.1` para video,
  por su mejora de tracking multiobjeto. Cambia a `facebook/sam3` si el comite
  exige el checkpoint base.
- **Analisis deportivo:** homografia opcional a coordenadas de cancha, posesion,
  eventos y metricas operativas de continuidad de tracking.

## Instalacion

Recomendado: Python 3.12, GPU NVIDIA, CUDA 12.6 o superior, y acceso aprobado a
los checkpoints de SAM 3 en Hugging Face.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para SAM 3 en GPU:

```powershell
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-sam3.txt
huggingface-cli login
```

## Uso rapido

Indexa el Drive oficial:

```powershell
samba-futbot index-drive --config config/default.yml --out data/manifests/drive_index.json
```

Descarga un video por nombre o ID:

```powershell
samba-futbot download --manifest data/manifests/drive_index.json --name video-297_singular_display.mov --out-dir data/raw
```

Descarga todos los videos preservando la estructura del Drive:

```powershell
samba-futbot download-all --manifest data/manifests/drive_index.json --out-dir data/raw
```

Muestra frames de calibracion:

```powershell
samba-futbot sample-frames --video data/raw/video-297_singular_display.mov --out-dir data/frames/video-297 --every 5 --max-frames 12
```

Ejecuta SAM 3/SAM 3.1:

```powershell
samba-futbot run-sam3 --config config/default.yml --video data/raw/video-297_singular_display.mov --out outputs/detections/video-297
```

Consolida tracks, eventos y demo:

```powershell
samba-futbot track --detections outputs/detections/video-297/detections.jsonl --out outputs/detections/video-297/tracks.jsonl
samba-futbot events --tracks outputs/detections/video-297/tracks.jsonl --out outputs/reports/video-297-events.json
samba-futbot render-demo --video data/raw/video-297_singular_display.mov --tracks outputs/detections/video-297/tracks.jsonl --out outputs/videos/video-297-demo.mp4 --max-seconds 120
```

## Estructura

- `src/samba_futbot/`: codigo del pipeline.
- `config/default.yml`: prompts, umbrales y parametros reproducibles.
- `docs/`: resumen del reto y estrategia profesional.
- `data/`: videos y frames locales, ignorados por git.
- `outputs/`: detecciones, reportes y videos generados, ignorados por git.

## Entregables pendientes antes del 19 de junio de 2026

- Video demo final de maximo 2 minutos.
- Reel de Instagram publico de al menos 30 segundos.
- Capturas o GIFs de resultados en este README.
- Enlace al reel en este README.
- Confirmar licencia y creditos del equipo.

## Creditos

Este proyecto usa o esta preparado para usar SAM 3 de Meta, Transformers de
Hugging Face, OpenCV, NumPy y pandas. Revisa y respeta las licencias de cada
dependencia y de los checkpoints de SAM 3.

## Licencia

Apache-2.0. Ver `LICENSE`.
