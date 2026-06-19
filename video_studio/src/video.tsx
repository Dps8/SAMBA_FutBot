import React, {Fragment} from 'react';
import {
  AbsoluteFill,
  Easing,
  OffthreadVideo,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {TransitionSeries, linearTiming} from '@remotion/transitions';
import {fade} from '@remotion/transitions/fade';

type Format = 'demo' | 'reel';
type SourceSceneDefinition = {
  kind: 'source';
  title: string;
  kicker: string;
  startSeconds: number;
  durationInFrames: number;
  accent: string;
};
type CustomSceneDefinition = {
  kind: 'intro' | 'method' | 'outro';
  durationInFrames: number;
};
type SceneDefinition = SourceSceneDefinition | CustomSceneDefinition;

const FPS = 30;
const NAVY = '#0b1f3a';
const INK = '#081014';
const GOLD = '#d6ad2f';
const GREEN = '#29bf78';
const CYAN = '#38bde8';
const WHITE = '#f7fafb';
const MUTED = '#b9c6cc';

const demoScenes: SceneDefinition[] = [
  {kind: 'intro', durationInFrames: 105},
  {kind: 'method', durationInFrames: 195},
  {kind: 'source', title: 'Modulo narrativo', kicker: 'Juego, posesion y eventos', startSeconds: 11, durationInFrames: 330, accent: GREEN},
  {kind: 'source', title: 'Lectura de la jugada', kicker: 'Pausa analitica con evidencia', startSeconds: 23, durationInFrames: 120, accent: CYAN},
  {kind: 'source', title: 'Gol validado', kicker: 'Cruce, persistencia y pared trasera', startSeconds: 28, durationInFrames: 360, accent: GOLD},
  {kind: 'source', title: 'Segunda camara', kicker: 'Mascaras SAM 3 e IDs temporales', startSeconds: 40, durationInFrames: 180, accent: CYAN},
  {kind: 'source', title: 'Analisis metrico', kicker: 'Distancia en m y velocidad en m/s', startSeconds: 48, durationInFrames: 240, accent: GREEN},
  {kind: 'source', title: 'Prediccion de movimiento', kicker: 'Pelota y robots con incertidumbre explicita', startSeconds: 57, durationInFrames: 240, accent: GOLD},
  {kind: 'source', title: 'Mapa de calor', kicker: 'Partido completo, 23,274 cuadros', startSeconds: 65, durationInFrames: 270, accent: GREEN},
  {kind: 'source', title: 'Lectura tactica', kicker: 'Actividad acumulada y mapa calibrado', startSeconds: 75, durationInFrames: 240, accent: CYAN},
  {kind: 'source', title: 'Resultados', kicker: 'Estadisticas y magnitudes fisicas', startSeconds: 83, durationInFrames: 360, accent: GOLD},
  {kind: 'source', title: 'Validacion', kicker: 'QA operativo y evaluacion cuantitativa', startSeconds: 95, durationInFrames: 510, accent: GREEN},
  {kind: 'outro', durationInFrames: 120},
];

const reelScenes: SceneDefinition[] = [
  {kind: 'intro', durationInFrames: 90},
  {kind: 'method', durationInFrames: 150},
  {kind: 'source', title: 'Narrativa', kicker: 'Dos robots y una pelota unica', startSeconds: 9, durationInFrames: 300, accent: GREEN},
  {kind: 'source', title: 'Lectura de jugada', kicker: 'Pausa y evidencia visual', startSeconds: 19, durationInFrames: 120, accent: CYAN},
  {kind: 'source', title: 'Gol validado', kicker: 'El marcador cambia tras confirmar', startSeconds: 24, durationInFrames: 300, accent: GOLD},
  {kind: 'source', title: 'Segunda camara', kicker: 'Segmentacion e identidad temporal', startSeconds: 34, durationInFrames: 180, accent: CYAN},
  {kind: 'source', title: 'Analisis metrico', kicker: 'Metros y metros por segundo', startSeconds: 40, durationInFrames: 210, accent: GREEN},
  {kind: 'source', title: 'Prediccion', kicker: 'Ramas cinematicas con p heuristica', startSeconds: 47, durationInFrames: 210, accent: GOLD},
  {kind: 'source', title: 'Mapa de calor', kicker: '12:56 de partido acumulado', startSeconds: 54, durationInFrames: 210, accent: GREEN},
  {kind: 'source', title: 'Resultados', kicker: 'Juego y magnitudes fisicas', startSeconds: 61, durationInFrames: 300, accent: GOLD},
  {kind: 'source', title: 'Validacion', kicker: 'QA y evaluacion COCO', startSeconds: 71, durationInFrames: 420, accent: CYAN},
  {kind: 'outro', durationInFrames: 90},
];

const transitionFrames = {demo: 12, reel: 10};

const timelineDuration = (scenes: SceneDefinition[], transition: number) =>
  scenes.reduce((sum, scene) => sum + scene.durationInFrames, 0) -
  transition * (scenes.length - 1);

export const demoDurationInFrames = timelineDuration(demoScenes, transitionFrames.demo);
export const reelDurationInFrames = timelineDuration(reelScenes, transitionFrames.reel);

const typography: React.CSSProperties = {
  fontFamily: 'Arial, Helvetica, sans-serif',
  color: WHITE,
};

const FieldMarks: React.FC<{portrait: boolean}> = ({portrait}) => (
  <AbsoluteFill style={{opacity: 0.15, pointerEvents: 'none'}}>
    <div style={{position: 'absolute', inset: portrait ? '9% 8%' : '12% 8%', border: '3px solid white'}} />
    <div style={{position: 'absolute', left: '50%', top: portrait ? '9%' : '12%', bottom: portrait ? '9%' : '12%', width: 3, background: 'white'}} />
    <div style={{position: 'absolute', left: '50%', top: '50%', width: portrait ? 360 : 260, height: portrait ? 360 : 260, border: '3px solid white', borderRadius: '50%', transform: 'translate(-50%, -50%)'}} />
  </AbsoluteFill>
);

const Intro: React.FC<{format: Format}> = ({format}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const portrait = format === 'reel';
  const entrance = spring({frame, fps, config: {damping: 16, stiffness: 110}});
  const lineWidth = interpolate(frame, [5, 45], [0, 1], {extrapolateRight: 'clamp'});
  return (
    <AbsoluteFill style={{...typography, backgroundColor: NAVY, overflow: 'hidden'}}>
      <FieldMarks portrait={portrait} />
      <div style={{position: 'absolute', left: portrait ? 58 : 108, right: portrait ? 58 : 108, top: portrait ? 230 : 225}}>
        <div style={{height: 8, width: `${lineWidth * 100}%`, backgroundColor: GOLD, marginBottom: portrait ? 56 : 38}} />
        <div style={{fontSize: portrait ? 50 : 58, fontWeight: 700, color: GOLD, letterSpacing: 2, opacity: entrance}}>PUMAS | UNAM</div>
        <div style={{fontSize: portrait ? 90 : 108, lineHeight: 0.92, fontWeight: 900, marginTop: 18, transform: `translateY(${(1 - entrance) * 50}px)`, opacity: entrance}}>SAMBA<br />FUTBOT</div>
        <div style={{fontSize: portrait ? 34 : 30, color: MUTED, marginTop: 34, maxWidth: portrait ? 820 : 1060}}>Vision por computadora para futbol robotico</div>
      </div>
      <div style={{position: 'absolute', left: portrait ? 58 : 108, bottom: portrait ? 220 : 105, fontSize: portrait ? 25 : 22, color: MUTED, lineHeight: 1.55}}>
        Copa FutBotMX 2026 | Categoria Profesional<br />
        German Alday | Raul Garcia | Darien Pina
      </div>
    </AbsoluteFill>
  );
};

const Method: React.FC<{format: Format}> = ({format}) => {
  const frame = useCurrentFrame();
  const portrait = format === 'reel';
  const items = [
    ['01', 'Percepcion', 'SAM 3 + prompts + mascaras'],
    ['02', 'Fusion', 'Color adaptable + contexto'],
    ['03', 'Tiempo', 'ByteTrack + consistencia'],
    ['04', 'Geometria', 'Homografia + reglas'],
    ['05', 'Prediccion', 'Ramas + incertidumbre'],
    ['06', 'QA', 'Metricas + evidencia'],
  ];
  return (
    <AbsoluteFill style={{...typography, backgroundColor: INK, padding: portrait ? '170px 64px' : '120px 105px'}}>
      <div style={{fontSize: portrait ? 25 : 22, color: GREEN, fontWeight: 800}}>PIPELINE REPRODUCIBLE</div>
      <div style={{fontSize: portrait ? 61 : 66, fontWeight: 900, marginTop: 14, lineHeight: 1.04}}>Evidencia antes<br />que decoracion</div>
      <div style={{display: 'grid', gridTemplateColumns: portrait ? '1fr' : '1fr 1fr', gap: portrait ? 32 : 26, marginTop: portrait ? 72 : 55}}>
        {items.map(([number, title, detail], index) => {
          const progress = interpolate(frame, [16 + index * 10, 32 + index * 10], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});
          return (
            <div key={number} style={{display: 'grid', gridTemplateColumns: portrait ? '80px 1fr' : '72px 1fr', alignItems: 'center', borderTop: `2px solid ${index % 2 === 0 ? GREEN : GOLD}`, paddingTop: 18, opacity: progress, transform: `translateY(${(1 - progress) * 22}px)`}}>
              <div style={{fontSize: portrait ? 29 : 25, color: index % 2 === 0 ? GREEN : GOLD, fontWeight: 900}}>{number}</div>
              <div>
                <div style={{fontSize: portrait ? 35 : 30, fontWeight: 800}}>{title}</div>
                <div style={{fontSize: portrait ? 24 : 20, color: MUTED, marginTop: 5}}>{detail}</div>
              </div>
            </div>
          );
        })}
      </div>
      <div style={{position: 'absolute', left: portrait ? 64 : 105, right: portrait ? 64 : 105, bottom: portrait ? 120 : 72, fontSize: portrait ? 24 : 20, color: MUTED}}>Sin YOLO | Sin eventos inventados | Probabilidades heuristicamente rotuladas</div>
    </AbsoluteFill>
  );
};

const SourceScene: React.FC<{scene: SourceSceneDefinition; source: string; chapter: number; totalChapters: number}> = ({scene, source, chapter, totalChapters}) => {
  const frame = useCurrentFrame();
  const {durationInFrames, width, height} = useVideoConfig();
  const introOpacity = interpolate(frame, [0, 7, 24, 38], [0, 1, 1, 0], {extrapolateRight: 'clamp'});
  const panelX = interpolate(frame, [0, 18], [-70, 0], {extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)});
  const progress = interpolate(frame, [0, durationInFrames - 1], [0, 1], {extrapolateRight: 'clamp'});
  return (
    <AbsoluteFill style={{backgroundColor: INK, overflow: 'hidden'}}>
      <OffthreadVideo
        src={staticFile(source)}
        trimBefore={Math.round(scene.startSeconds * FPS)}
        muted
        style={{width, height, objectFit: 'contain'}}
      />
      <AbsoluteFill style={{boxShadow: 'inset 0 0 90px rgba(0,0,0,0.28)', pointerEvents: 'none'}} />
      <div style={{...typography, position: 'absolute', top: height * 0.13, left: 0, padding: height > width ? '28px 52px 30px 68px' : '24px 70px 26px 105px', backgroundColor: 'rgba(8,16,20,0.92)', borderRight: `8px solid ${scene.accent}`, opacity: introOpacity, transform: `translateX(${panelX}px)`}}>
        <div style={{fontSize: height > width ? 22 : 18, color: scene.accent, fontWeight: 900, letterSpacing: 2}}>CAPITULO {String(chapter).padStart(2, '0')}</div>
        <div style={{fontSize: height > width ? 43 : 38, fontWeight: 900, marginTop: 7}}>{scene.title}</div>
        <div style={{fontSize: height > width ? 24 : 20, color: MUTED, marginTop: 7}}>{scene.kicker}</div>
      </div>
      <div style={{position: 'absolute', left: 0, right: 0, bottom: 0, height: height > width ? 10 : 7, backgroundColor: 'rgba(255,255,255,0.16)'}}>
        <div style={{width: `${progress * 100}%`, height: '100%', backgroundColor: scene.accent}} />
      </div>
      <div style={{...typography, position: 'absolute', right: height > width ? 34 : 46, bottom: height > width ? 34 : 28, fontSize: height > width ? 18 : 15, color: 'rgba(255,255,255,0.72)', fontWeight: 700}}>{chapter}/{totalChapters}</div>
    </AbsoluteFill>
  );
};

const Outro: React.FC<{format: Format}> = ({format}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const portrait = format === 'reel';
  const reveal = spring({frame, fps, config: {damping: 18, stiffness: 95}});
  return (
    <AbsoluteFill style={{...typography, backgroundColor: NAVY, alignItems: 'center', justifyContent: 'center', textAlign: 'center'}}>
      <FieldMarks portrait={portrait} />
      <div style={{opacity: reveal, transform: `scale(${0.94 + reveal * 0.06})`}}>
        <div style={{fontSize: portrait ? 29 : 24, color: GOLD, fontWeight: 900, letterSpacing: 3}}>PUMAS | UNIVERSIDAD NACIONAL AUTONOMA DE MEXICO</div>
        <div style={{fontSize: portrait ? 76 : 86, fontWeight: 900, marginTop: 26}}>Vision que se puede auditar</div>
        <div style={{fontSize: portrait ? 27 : 25, color: MUTED, marginTop: 32, lineHeight: 1.6}}>SAM 3 + contexto + tracking + geometria<br />Codigo, pruebas y evidencia reproducible</div>
        <div style={{width: portrait ? 500 : 620, height: 7, backgroundColor: GREEN, margin: '46px auto 0'}} />
      </div>
      <div style={{position: 'absolute', bottom: portrait ? 145 : 72, fontSize: portrait ? 23 : 20, color: MUTED}}>Copa FutBotMX 2026 | Categoria Profesional</div>
    </AbsoluteFill>
  );
};

const Scene: React.FC<{scene: SceneDefinition; format: Format; chapter: number; totalChapters: number}> = ({scene, format, chapter, totalChapters}) => {
  if (scene.kind === 'intro') return <Intro format={format} />;
  if (scene.kind === 'method') return <Method format={format} />;
  if (scene.kind === 'outro') return <Outro format={format} />;
  if (scene.kind === 'source') {
    return <SourceScene scene={scene} source={format === 'demo' ? 'source-demo.mp4' : 'source-reel.mp4'} chapter={chapter} totalChapters={totalChapters} />;
  }
  return null;
};

export const ProfessionalCut: React.FC<{format: Format}> = ({format}) => {
  const scenes = format === 'demo' ? demoScenes : reelScenes;
  const transition = transitionFrames[format];
  const totalChapters = scenes.filter((scene) => scene.kind === 'source').length;
  return (
    <TransitionSeries>
      {scenes.map((scene, index) => (
        <Fragment key={`${scene.kind}-${index}`}>
          {index > 0 ? (
            <TransitionSeries.Transition
              presentation={fade()}
              timing={linearTiming({durationInFrames: transition})}
            />
          ) : null}
          <TransitionSeries.Sequence durationInFrames={scene.durationInFrames}>
            <Scene
              scene={scene}
              format={format}
              chapter={scenes.slice(0, index + 1).filter((item) => item.kind === 'source').length}
              totalChapters={totalChapters}
            />
          </TransitionSeries.Sequence>
        </Fragment>
      ))}
    </TransitionSeries>
  );
};
