# Documentacion Publica

Esta carpeta contiene solamente material necesario para comprender, reproducir
y evaluar SAMBA FutBot. Las bitacoras internas, rutas de maquinas de trabajo,
selecciones preliminares de clips y explicaciones personales no forman parte de
la entrega publica.

## Documentos

| Archivo | Proposito |
|---|---|
| `RETO.md` | Resumen local de requisitos de la categoria profesional. |
| `RESULTS.md` | Resultados procesados, limitaciones conocidas y rutas de artefactos. |
| `SAM3_FINETUNING.md` | Preparacion de datos, adaptacion de SAM 3 y contrato de evaluacion. |
| `SUBMISSION_COMPLIANCE.md` | Trazabilidad contra los puntos 3.2.2, 3.5, 3.6 y 3.7. |

La instalacion, arquitectura, comandos del CLI y reproduccion de videos se
mantienen en el `README.md` principal para evitar dos manuales divergentes.

## Evidencia Final

- `submission-video-manifest-v1.3.0.json`: entradas, codec, resolucion,
  duracion y alcance de afirmaciones del demo y reel.
- `submission-video-qa-v1.3.0.json`: hashes, decodificacion completa,
  estabilizacion de porteria azul y composicion Remotion revisadas.
- `video-427-calibrated-goal-events.json`: disparo y gol confirmado mediante
  pelota rastreada, cruce dirigido, persistencia y contacto con pared trasera.
- `video-427-calibrated-goal-summary.json`: marcador confirmado y resumen de
  eventos de la jugada.
- `sam3-finetune-comparison.json`: evaluacion COCO de mascaras sobre 128
  imagenes y comparacion baseline/adaptacion.
- `full-match-IMG_9933-summary.json`: cobertura y alcance de la corrida de
  partido completo usada para el mapa de calor.
- `IMG_9933-full-metrics.json` y `IMG_9933-full-heatmap-qa-report.json`:
  metricas operativas y control de calidad del partido completo.
- `IMG_9938-ball-filter-report.json`: resultado del filtro contextual de una
  pelota por cuadro.

Los videos, tracks y mascaras son artefactos grandes y se publican en el GitHub
Release o se generan bajo `outputs/`; no se duplican dentro de `docs/`.

## Capturas

`assets/` contiene un conjunto pequeno de cuadros finales enlazados desde el
README: enfoques, analisis, ambas porterias, prediccion de pelota/robots,
heatmap, mapa tactico y validacion cuantitativa. Las capturas antiguas o no
referenciadas se excluyen para evitar confundir resultados preliminares con la
entrega vigente.

## Criterio De Publicacion

Una afirmacion se presenta como resultado cuando tiene evidencia versionada y
supera su compuerta QA. Los valores candidatos, las calibraciones manuales y
las limitaciones se rotulan explicitamente; no se convierten en resultados
automaticos por edicion del video.
