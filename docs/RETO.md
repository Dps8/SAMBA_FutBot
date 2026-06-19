# Resumen del reto FutBotMX Vision por Computadora

Fuente local: `Convocatoria_CopaFutBotMX-Meta-VF-20260429T020141.pdf`.

## Categoria profesional

La categoria profesional espera rigor tecnico avanzado. La convocatoria pide:

- Innovacion sobre SAM 3: fine-tuning, prompt engineering avanzado o integracion
  con otros modelos.
- Calidad del pipeline funcional.
- Rendimiento y resultados reproducibles.
- Metricas cuantitativas.
- Visualizacion con valor deportivo real.

## Entregables obligatorios

- Repositorio publico en GitHub.
- Codigo funcional que use SAM 3 para segmentar campo, robots aliados, robots
  rivales y balon.
- Tracking de robots y balon a lo largo del video.
- Deteccion de eventos clave como pases, tiros, intercepciones o colisiones,
  segun alcance.
- Al menos una visualizacion: mapas de calor, posesion, trails, Voronoi, grafos,
  dashboard o anotaciones narrativas.
- Video de maximo 2 minutos con original y resultado segmentado.
- Reel publico en Instagram de al menos 30 segundos, enlazado desde el README.
- README con arquitectura, instalacion, reproduccion, hardware/software,
  resultados, reel, licencia y creditos.

## Fechas clave

- Periodo de desarrollo y acompanamiento: 25 de mayo a 19 de junio de 2026.
- Limite del entregable GitHub: 19 de junio de 2026, 23:59 hora centro de Mexico.
- Evaluacion: 20 al 24 de junio de 2026.
- Resultados: 25 de junio de 2026.
- Premiacion: 26 de junio de 2026.

## Datos

Drive oficial del reto:

`https://drive.google.com/drive/folders/1TF7-P4rAwPmHFw_TjmNfFU3ORxqnp8CD`

El indexador incluido encuentra:

- `17Abril`
- `18abril`
- subcarpetas de camaras
- videos `.mov` publicos

## Regla De Gol Usada Por El Pipeline

El reglamento de juego, apartados 4.4.5 y 7.4.4, define un gol valido cuando el
balon hace contacto con la pared trasera de la porteria. Cruzar la linea de gol
por si solo no basta. Por ello, la ruta calibrada del pipeline exige:

1. pelota unica con tracking continuo;
2. cruce exterior-interior del segmento de gol;
3. persistencia dentro de la porteria;
4. contacto con la pared trasera calibrada.

Un cruce sin contacto trasero produce `goal_rejected` y no modifica el
marcador. Fuente:
[Reglas de Futbol para la Copa FutBotMX 2026](https://secihti.mx/wp-content/uploads/2026/01/Reglas_Copa_FutBotMX_v3_2026-01-21.pdf).
