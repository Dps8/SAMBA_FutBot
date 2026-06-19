import React from 'react';
import {Composition} from 'remotion';
import {
  ProfessionalCut,
  demoDurationInFrames,
  reelDurationInFrames,
} from './video';

export const VideoRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="ProfessionalDemo"
        component={ProfessionalCut}
        durationInFrames={demoDurationInFrames}
        fps={30}
        width={1920}
        height={1080}
        defaultProps={{format: 'demo' as const}}
      />
      <Composition
        id="ProfessionalReel"
        component={ProfessionalCut}
        durationInFrames={reelDurationInFrames}
        fps={30}
        width={1080}
        height={1920}
        defaultProps={{format: 'reel' as const}}
      />
    </>
  );
};
