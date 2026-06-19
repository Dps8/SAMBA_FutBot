import {copyFile, mkdir} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const studio = resolve(here, '..');
const repo = resolve(studio, '..');
const source = resolve(repo, 'outputs', 'review', '2026-06-19', 'submission_v1_3_base');
const publicDir = resolve(studio, 'public');

await mkdir(publicDir, {recursive: true});
await Promise.all([
  copyFile(resolve(source, 'SAMBA_FutBot-demo-final.mp4'), resolve(publicDir, 'source-demo.mp4')),
  copyFile(resolve(source, 'SAMBA_FutBot-reel-instagram.mp4'), resolve(publicDir, 'source-reel.mp4')),
]);

console.log(`Prepared verified media in ${publicDir}`);
