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
   - Balon: `orange soccer ball`, `small ball`.
   - Usar prompts negativos y filtros geometricos si SAM 3 confunde publico,
     logos, reflejos o bordes.

3. **Tracking**
   - Preferir IDs nativos de SAM 3 video.
   - Completar huecos con tracker IoU y filtros temporales.
   - Reportar continuidad de tracks, fragmentacion y FPS.

4. **Separacion de equipos**
   - Clasificar robots por color dominante dentro de la mascara.
   - Ajustar paletas por video en `config/default.yml`.

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
- Comparar tracking nativo contra tracking nativo mas reparacion IoU.
- Medir estabilidad de area, fragmentacion y porcentaje de frames con balon.
