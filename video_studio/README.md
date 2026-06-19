# SAMBA FutBot Video Studio

Edicion reproducible en Remotion para el demo horizontal y el reel vertical.
El estudio no recalcula detecciones ni eventos: remonta los videos verificados
de `submission_v1_3_base` y conserva sus afirmaciones.

```powershell
cd video_studio
pnpm install
pnpm media:prepare
pnpm render:demo
pnpm render:reel
```

Los MP4 preparados bajo `public/` y los renders bajo `outputs/` son artefactos
locales ignorados por Git. El codigo del timeline, las duraciones y los textos
si quedan versionados.
