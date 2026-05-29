# Estrategia profesional

## Hipotesis tecnica

SAM 3 ya entrega segmentacion y tracking por concepto, pero en futbol robotico
hay objetos pequenos, reflejos, oclusiones y confusiones entre robots similares.
La solucion debe demostrar dominio del modelo y del dominio deportivo, no solo
ejecutar un notebook.

## Pipeline propuesto

1. **Seleccion de videos**
   - Priorizar camara superior cuando exista, porque facilita homografia y mapas
     de calor.
   - Usar videos laterales para una demo visualmente clara.

2. **Prompts y refinamiento**
   - Campo: `soccer field`, `green playing field`.
   - Robots: `small wheeled soccer robot`, `robot soccer player`.
   - Balon: `small ball`, `golf ball`, `ball on green field`, `ball near robot`
     y la variante especifica del reglamento actual `small orange ball`.
   - Fusionar SAM3 pelota con una fuente cromatica/geométrica configurable. Hoy
     el perfil por defecto es `orange`, pero el pipeline puede cambiar a
     `white`, `yellow` o rangos HSV manuales sin reescribir codigo.
   - Para porterias, usar prompts amplios como `blue box`, `yellow box`,
     `goal frame`, `goal post`, `caja azul` y `caja amarilla`; si SAM3 encuentra
     una caja, recalibrar el HSV desde esas coordenadas para no depender de un
     color hardcodeado.
   - Agregar variantes de dominio como `blue/yellow board`, `blue/yellow table`
     y `tabla azul/amarilla`, y aplicar reglas fisicas: una sola porteria por
     color en cada frame y siempre asociada al campo verde.
   - Usar filtros geometricos y contexto de campo/robots si SAM 3 o el detector
     de color confunden logos, reflejos, bordes o piezas del robot.

3. **Tracking**
   - Preferir IDs nativos de SAM 3 video.
   - Completar huecos con tracker IoU y filtros temporales.
   - Reportar continuidad de tracks, fragmentacion y FPS.

4. **Separacion de equipos**
   - Clasificar robots por color dominante dentro de la mascara.
   - Ajustar paletas por video en `config/default.yml`.
   - La primera integracion usa paleta RGB `blue/yellow` y votacion por track
     para asignar `team` a cada robot antes de calcular eventos.

5. **Analisis de juego**
   - Posesion: robot mas cercano al balon en coordenadas de cancha o pixeles.
   - Homografia: convertir centros de pelota desde pixeles a metros de cancha.
   - Zonas: reportar ocupacion por grilla para diferenciar juego defensivo,
     medio y ofensivo.
   - Mapa tactico: PNG con trayectoria y calor por zonas para explicar el
     comportamiento sin depender del video completo.
   - Reglas oficiales: usar campo `2.43 m x 1.82 m`, circulo central de
     `0.60 m`, area de penalizacion `0.25 m x 0.80 m` y porteria de `0.60 m`
     para que las metricas sean defendibles.
   - Candidatos reglamentarios: entradas a porteria, balon fuera de campo y
     muestras de robots en area de penalizacion.
   - Goles visuales: si SAM3 detecta `goal_blue` o `goal_yellow`, generar
     candidatos de gol cuando la pelota entra en la caja de porteria. Para
     resultados finales se prefiere homografia y validacion manual.
   - Posesion por equipo: sumar frames de posesion a partir del robot poseedor
     y su equipo asignado.
   - QA automatico: clasificar cada corrida como `good`, `review` o `fail`
     usando cobertura de pelota, saltos imposibles, cobertura de campo/robots y
     senales reglamentarias.
   - Reporte reproducible: generar Markdown por corrida para convertir metricas
     tecnicas en narrativa de evaluacion.
   - Pase: cambio de poseedor dentro del mismo equipo.
   - Intercepcion: cambio de poseedor entre equipos.
   - Tiro: velocidad del balon hacia zona de gol.
   - Colision: distancia pequena entre robots con convergencia de trayectorias.

6. **Visualizacion**
   - Overlay de mascaras y cajas.
   - Trails por robot y balon.
   - Heatmap por equipo.
   - Panel lateral con posesion y eventos detectados.

## Riesgos y mitigaciones

- **Checkpoint gated:** solicitar acceso en Hugging Face y documentar el paso.
- **Videos grandes:** no subir videos al repo, solo scripts de descarga.
- **SAM 3 lento:** procesar clips cortos, `stride`, resolucion controlada y
  SAM 3.1 para multiobjeto.
- **Sin ground truth:** reportar metricas operativas y, si hay tiempo, anotar
  100 a 200 frames para IoU y precision/recall.

## Experimentos sugeridos

- Comparar `facebook/sam3` contra `facebook/sam3.1` en el mismo clip.
- Probar ensambles de prompts contra un prompt unico.
- Comparar pelota por SAM3, pelota por color y fusion SAM3+color en la misma
  ventana de camara superior.
- Comparar tracking nativo contra tracking nativo mas reparacion IoU.
- Medir estabilidad de area, fragmentacion y porcentaje de frames con balon.
- Usar `qa-run` para ordenar variantes por score antes de revisar manualmente
  los videos renderizados.
