# Matriz de cumplimiento de entrega

Revision final para la categoria Profesional de Vision por Computadora. La
fuente normativa es `Convocatoria_CopaFutBotMX-Meta-VF-20260429T020141.pdf`.

## 3.2.2 Categoria Profesional

| Objetivo | Estado | Evidencia |
|---|---|---|
| Innovacion sobre SAM 3 | Cumplido | Prompts de contexto, puntos/cajas, fusion semantica-color, tracker ByteTrack/IoU, geometria, eventos y QA. |
| Fine-tuning de dominio | Cumplido | 128 imagenes de validacion; AP global `0.2299 -> 0.3586` y AP robots `0.4496 -> 0.7068`. Ver `evidence/sam3-finetune-comparison.json`. |
| Metricas cuantitativas | Cumplido | COCO AP/AR, cobertura por clase, fragmentacion, velocidad, trayectoria, posesion, eventos y compuertas de calidad. |
| Reproducibilidad | Cumplido | CLI, configuracion versionada, runbook de fine-tuning, pruebas y `scripts/build_submission_videos.py`. |

## 3.5 Entregables

| Regla | Estado | Evidencia |
|---|---|---|
| 3.5.1 Campo, aliados, rivales y pelota | Cumplido con QA | SAM 3 segmenta campo/robots/pelota; la clasificacion de equipo usa apariencia y puede quedar `unknown` si no supera confianza. Las porterias azul/amarilla tienen prompt y fallback cromatico/geometrico. |
| 3.5.1 Trayectorias y eventos | Cumplido | Tracks persistentes, trayectoria de pelota, distancia robot-balon, pases, tiros, intercepciones, colisiones, goles candidatos y estados de juego. |
| 3.5.2 Visualizacion avanzada | Cumplido | Mapa de calor dinamico y acumulado sobre `IMG_9933.MOV` completo (12:56, 23,278 cuadros), mapa tactico, posesion candidata, flujo, narrativa y vista analitica. |
| 3.5.3 Demo de hasta 2 min | Cumplido | `SAMBA_FutBot-demo-final.mp4`: aproximadamente 103 s, 1920x1080, original+resultado, narrativa, analisis, multivista, heatmap completo, H.264 y sin audio. Se publica en GitHub Release v1.1.0. |
| 3.5.3 Reel de al menos 30 s | Archivo listo | `SAMBA_FutBot-reel-instagram.mp4`: aproximadamente 63 s, 1080x1920, H.264, sin audio. Falta publicar en Instagram y reemplazar el enlace provisional del README. |
| 3.5.4 README completo | Cumplido salvo enlace externo | Arquitectura, instalacion, reproduccion, entorno, resultados, capturas, licencia y creditos estan incluidos. |

## 3.6 Software de terceros

No se usa YOLO ni Ultralytics. El proyecto integra bibliotecas de proposito
general y el repositorio oficial de SAM 3; cada dependencia, rol y licencia se
declara en `THIRD_PARTY_NOTICES.md`. El pipeline, las reglas de fusion, el
post-procesamiento, las metricas, los eventos, QA y los renders son propios del
proyecto.

## 3.7 Rubrica Profesional

| Criterio | Evidencia principal |
|---|---|
| Innovacion | Prompts y contexto rotativos, fusion SAM 3 + color adaptable, fine-tuning medido, ByteTrack, recuperacion geometrica, prediccion de tiro y QA por claim. |
| Pipeline funcional | `process-video` y `process-top-camera` producen detecciones, tracks, estados, eventos, metricas, QA, mapas y doble video. |
| Rendimiento | Ventanas de 120 cuadros, limite de 16 objetos, rotacion de prompts, ejecucion por lotes y post-procesamiento sin GPU. |
| Visualizacion | Mascaras semitransparentes, cajas por objeto, confianza, IDs, distancias, trayectorias, narrativa, analisis y mapas de calor. |

## Metricas defendibles

La corrida completa de `IMG_9933.MOV` declara 23,278 cuadros (12:56), de los
cuales 23,274 son legibles: pelota en 97.2% y robots en 83.5% de cuadros. El
mapa de calor usa 23,784 muestras despues de filtros geometricos y de campo
calibrado sobre toda la duracion.
Las distancias y velocidades en metros se calculan con homografia de cancha
2.43 x 1.82 m sobre asociaciones validas. La fragmentacion del tracking de la
corrida cromatica completa impide defender posesion por equipo como verdad de
campo; se presenta como candidata con confianza y alcance explicitos.

## Unico paso externo pendiente

Publicar `SAMBA_FutBot-reel-instagram.mp4` en una cuenta publica de Instagram y
reemplazar la linea pendiente del README con la URL del reel antes del cierre.
