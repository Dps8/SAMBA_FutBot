# Estrategia profesional

## Hipotesis tecnica

SAM 3 ya entrega segmentacion y tracking por concepto, pero en futbol robotico
hay objetos pequenos, reflejos, oclusiones, humanos alrededor de la cancha,
pausas de juego y robots visualmente similares. La solucion debe demostrar
dominio del modelo y del dominio deportivo, no solo ejecutar un notebook.

La meta competitiva es tener un pipeline reproducible que entregue evidencia:
videos narrativos, videos tecnicos, metricas, mapas tacticos, QA y una ruta
clara hacia fine-tuning.

## Pipeline propuesto

1. **Seleccion de videos**
   - Priorizar camara superior cuando exista, porque facilita homografia, mapas
     de calor, distancia real y analisis tactico.
   - Usar videos laterales para una demo visualmente clara cuando ayuden a
     entender jugadas.
   - Procesar clips cortos primero, luego lotes mas grandes con QA index.

2. **Prompts y refinamiento**
   - Campo: `soccer field`, `green playing field`, `green robotic soccer field`.
   - Robots: `small wheeled soccer robot`, `robot soccer player`.
   - Balon: `small ball`, `golf ball`, `ball on green field`, `ball near robot`,
     `small orange ball`.
   - Porterias: `blue box`, `dark blue box`, `yellow box`, `goal frame`,
     `goal post`, `blue/yellow board`, `blue/yellow table`, `caja azul`,
     `caja amarilla`, `tabla azul`, `tabla amarilla`.
   - Fusionar SAM3 pelota con una fuente cromatica/geometrica configurable. Hoy
     el perfil por defecto es `orange`, pero el pipeline puede cambiar a
     `white`, `yellow` o rangos HSV manuales sin reescribir codigo.
   - Para porterias, si SAM3 encuentra una caja, recalibrar el HSV desde esas
     coordenadas para no depender de un color hardcodeado.
   - Aplicar reglas fisicas: una sola porteria por color en cada frame y
     siempre asociada al campo verde.
   - Si solo aparece una porteria, inferir la opuesta por simetria respecto al
     campo detectado y marcarla como geometrica, no como deteccion visual pura.
   - Usar filtros geometricos y contexto de campo/robots si SAM3 o color
     confunden logos, reflejos, bordes o piezas del robot.

3. **Tracking**
   - Preferir IDs nativos de SAM3 video cuando sean estables.
   - Completar huecos con tracker IoU y filtros temporales.
   - Reportar continuidad de tracks, fragmentacion y FPS.
   - Comparar luego contra ByteTrack o tracker nativo SAM3 si el tiempo lo
     permite.

4. **Separacion de equipos**
   - Clasificar robots por color dominante dentro de la mascara o caja.
   - Ajustar paletas por video en `config/default.yml`.
   - Asignar `team=blue/yellow/unknown` por votacion temporal de track.
   - Usar QA para medir proporcion de robots `unknown` antes de defender
     posesion por equipo.

5. **Analisis de juego**
   - Posesion: robot mas cercano al balon en coordenadas de cancha o pixeles.
   - Distancia robot-balon: en narrativa mostrar el robot mas cercano; en
     analisis mostrar distancia de cada robot al balon.
   - Homografia: convertir pelota y robots desde pixeles a metros de cancha.
   - Zonas: reportar ocupacion por grilla para diferenciar juego defensivo,
     medio y ofensivo.
   - Tercio relativo por equipo: convertir posiciones de robots en muestras
     defensivas, medias u ofensivas segun el lado que defiende cada equipo.
   - Presion ofensiva: resumir la proporcion de robots de cada equipo en su
     tercio ofensivo.
   - Control territorial: estimar equipo lider por zona de grilla a partir de
     muestras proyectadas de robots.
   - Mapa tactico: PNG con trayectoria de pelota, ocupacion por zonas, calor de
     robots por equipo, control territorial y robots coloreados.
   - Reglas oficiales: usar campo `2.43 m x 1.82 m`, circulo central de
     `0.60 m`, area de penalizacion `0.25 m x 0.80 m` y porteria de `0.60 m`.
   - Validacion de calibracion: reportar error de reproyeccion y puntos fuera
     de frame antes de presentar distancias o velocidades en metros.
   - Goles visuales: si hay `goal_blue` o `goal_yellow`, generar candidatos de
     gol cuando la pelota entra en la caja de porteria. Para resultados finales
     se prefiere homografia y validacion manual.
   - Pase: cambio de poseedor dentro del mismo equipo.
   - Intercepcion: cambio de poseedor entre equipos.
   - Tiro: velocidad del balon con direccion hacia el lado de porteria.
   - Colision: distancia pequena entre robots con convergencia de trayectorias.
   - Probabilidades tacticas: iniciar con heuristicas explicables de gol, pase,
     mantener posesion y perder posesion usando distancia robot-balon, velocidad
     y direccion del balon, zona de cancha, angulo hacia porteria, densidad de
     robots entre balon y porteria, duracion de posesion y cercania de rivales.
     Etiquetar estas probabilidades como `heuristic` hasta calibrarlas con datos.

6. **Estado de juego y eventos externos**
   - Balon fuera de juego: detectar ausencia de balon en campo, balon lejos de
     robots, baja velocidad prolongada, falta de posesion o balon fuera de los
     limites calibrados.
   - Tiempo muerto: segmento con balon no jugable, humanos/arbitro cerca del
     campo, o pausa prolongada sin dinamica de partido.
   - Robot retirado: desaparicion persistente de un track, salida del campo o
     traslado hacia fuera por humano/arbitro.
   - Robot roto o inmovil: robot con movimiento casi cero durante una ventana
     larga mientras el balon/juego continua, especialmente tras colision.
   - Intervencion de jugador/arbitro: agregar prompts/clases para `referee`,
     `human hand`, `person`, `player`, `arbitro`, `mano`; marcar presencia
     humana dentro o cerca del campo.
   - Separar segmentos no-juego de metricas finales de velocidad, posesion y
     trayectoria para no contaminar claims deportivos.

7. **Visualizacion**
   - Dos videos por corrida:
     - `narrative`: video limpio para contar el partido; incluye posesion,
       equipo, evento reciente y distancia del robot mas cercano al balon.
     - `analysis`: video tecnico; incluye cajas, confianza, equipos, distancia
       de cada robot al balon, velocidad, trayectoria, presion de tiro y
       anotaciones densas.
   - Pausas analiticas o freeze frames solo en `analysis`: congelar 1-2 segundos
     en eventos importantes y senalar trayectoria probable, probabilidad de gol,
     posible pase, riesgo de perder posesion, balon fuera de juego, robot
     inmovil, robot retirado o intervencion humana/arbitro.
   - Overlay de mascaras y cajas.
   - Trails por robot y balon.
   - Heatmap de robots por equipo y heatmap de pelota/zona.
   - Panel ligero con posesion y eventos detectados.

8. **QA, reportes y reproducibilidad**
   - QA automatico: clasificar cada corrida como `good`, `review` o `fail`
     usando cobertura de pelota, saltos imposibles, cobertura de campo/robots,
     homografia, equipos desconocidos y senales reglamentarias.
   - `claim_readiness`: marcar que claims estan listos para defender:
     `ball_tracking`, `metric_speed_trajectory`, `team_possession`,
     `goal_scoring`, `shot_pressure`.
   - Reporte reproducible: generar Markdown y manifiesto JSON por corrida con
     timestamp, runtime, branch, commit, estado local de Git, artefactos y
     resumenes.
   - Huella de codigo: incluir SHA256 de `src`, `config` y dependencias
     declaradas para auditoria.

9. **Dataset estadistico y fine-tuning**
   - Exportar candidatos de pseudo-etiquetas desde detecciones SAM3 con mascara,
     score y area confiables.
   - Crear tabla por frame/track con features tacticas: distancia robot-balon,
     velocidades, aceleracion, zona, equipo, poseedor, proximidad a porteria,
     presion rival, presencia humana y estado en juego/no-juego.
   - Calibrar probabilidades tacticas con modelos ligeros antes de prometer
     prediccion aprendida: regresion logistica, gradient boosting o HMM/estado
     temporal simple.
   - Fine-tuning de segmentacion: usar pseudo-labels curados para robots, balon,
     porterias y posibles humanos/arbitro.
   - Mantener split por video para evitar fuga temporal.
   - Evaluacion honesta: reportar si el ground truth es humano o pseudo-label;
     no mezclar consistencia contra pseudo-label con precision real.

## Estado del plan general

| Bloque | Estado | Siguiente accion |
|---|---|---|
| Ingesta/Drive/videos | Implementado parcial | Mantener rutas limpias y manifests reproducibles. |
| SAM3 por ventanas | Implementado | Afinar prompts por vista/camara. |
| Pelota hibrida SAM3 + color + refinamiento | Implementado | Mejorar oclusion y cambios de color. |
| Porterias por prompt + color + geometria | Implementado parcial | Validar mas clips y medir falsos positivos. |
| Tracking IoU | Implementado | Comparar con SAM3 nativo/ByteTrack si hay tiempo. |
| Equipos blue/yellow | Implementado parcial | Reducir `unknown` y validar por video. |
| Posesion por equipo | Implementado | Mostrar distancia al balon en ambos renders. |
| Eventos base: gol, tiro, pase, intercepcion, colision | Implementado como candidatos | Calibrar umbrales y asociar con estado en juego. |
| Estado en juego/no-juego | Parcial | Formalizar `dead_ball`, pausa e intervencion humana. |
| Robot retirado/roto/inmovil | Pendiente | Implementar heuristicas por desaparicion, salida e inmovilidad. |
| Intervencion humana/arbitro | Pendiente | Agregar prompts/clases y filtros para humanos/manos/arbitro. |
| Homografia, velocidad y trayectoria metrica | Implementado parcial | Sustituir plantilla por calibraciones reales por clip. |
| Mapas tacticos y heatmaps | Implementado parcial | Aumentar mapas por equipo/robot y resumen visual. |
| Videos narrativa/analisis | Implementado parcial | Agregar distancia en narrative y freeze frames en analysis. |
| QA y claim readiness | Implementado | Usarlo como compuerta para claims finales. |
| Pseudo-labels para fine-tuning | Implementado inicial | Exportar candidatos reales y curarlos. |
| Fine-tuning | Pendiente | Preparar dataset, split por video y experimento LoRA/adapter. |
| Reporte final/reproducibilidad | Implementado parcial | Generar evidencia final con comandos, manifests y videos. |

## Prioridad inmediata

1. Completar visualizacion: distancia al balon en ambos videos y freeze frames
   analiticos para eventos importantes.
2. Formalizar estados `in_play`, `dead_ball`, `human_intervention`,
   `robot_removed`, `robot_disabled` y excluir segmentos no-juego de velocidad,
   posesion y trayectoria final.
3. Procesar lote de clips superiores y producir QA index para escoger evidencia
   final.
4. Exportar pseudo-labels reales para robots, pelota, porterias y humanos si
   aparecen.
5. Ejecutar fine-tuning solo despues de tener pseudo-labels auditables y una
   baseline con metricas.

## Trabajo paralelo con agentes

Conviene usar agentes en paralelo si el usuario da una orden explicita para
hacerlo. No deben tocar los mismos archivos a la vez. Division recomendada:

- Agente A, datos/fine-tuning: manifiesto de pseudo-labels, criterios de
  curacion, split por video y primer plan LoRA/adapter.
- Agente B, eventos reglamentarios: candidatos de humano/arbitro, robot
  retirado, robot inmovil y balon fuera de juego.
- Agente C, visualizacion/QA: freeze frames, distancia en narrative y panel
  tecnico del analysis video.
- Agente D, evaluacion: correr lotes, revisar QA index, seleccionar clips
  buenos y documentar fallos visuales.

La ruta critica local debe quedarse con integracion, decisiones de arquitectura,
revision de cambios y pruebas completas.

## Riesgos y mitigaciones

- **Checkpoint gated:** solicitar acceso en Hugging Face y documentar el paso.
- **Videos grandes:** no subir videos al repo, solo scripts de descarga.
- **SAM 3 lento:** procesar clips cortos, `stride`, resolucion controlada y
  SAM 3.1 para multiobjeto.
- **Sin ground truth:** reportar metricas operativas y, si hay tiempo, anotar
  100 a 200 frames para IoU y precision/recall.
- **Probabilidades no calibradas:** marcar como heuristicas hasta tener dataset
  acumulado y evaluacion.
- **Eventos humanos ambiguos:** reportar como candidatos y usar evidencia visual
  en freeze frames antes de afirmar una intervencion.

## Experimentos sugeridos

- Comparar `facebook/sam3` contra `facebook/sam3.1` en el mismo clip.
- Probar ensambles de prompts contra un prompt unico.
- Comparar pelota por SAM3, pelota por color y fusion SAM3+color en la misma
  ventana de camara superior.
- Comparar tracking nativo contra tracking nativo mas reparacion IoU.
- Medir estabilidad de area, fragmentacion y porcentaje de frames con balon.
- Usar `qa-run` para ordenar variantes por score antes de revisar manualmente
  los videos renderizados.
- Exportar pseudo-labels de los mejores clips y correr un primer experimento de
  fine-tuning con split por video.
