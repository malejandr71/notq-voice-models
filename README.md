# Notq voice models

Offline text-to-speech packs for the [Notq](https://github.com/malejandr71) app, run on the phone by
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 1.13.8. The files are published as GitHub Release assets
and downloaded by the app at run time (Wi-Fi by default), checked by size and MD5.

| Pack | Model | Voices | Size |
|---|---|---|---|
| `en` | Kokoro v1.0, int8 (`kokoro-int8-multi-lang-v1_0` from sherpa-onnx), English voices only, US-English lexicon | af_heart, af_bella, af_sarah, bf_emma, am_michael, am_fenrir, bm_george | ~119 MB |
| `de` | Kokoro-82M German fine-tune (`crane-local-ai/Kokoro-82M-v1.0-German-ONNX`), sherpa-onnx metadata added, dynamic int8 | df_kerstin | ~121 MB |

Each pack also carries the minimal eSpeak NG data for its language (phoneme tables + one dictionary).

## How the files are made

`build.py` downloads the sources, checks each against a pinned SHA-256, and builds the packs. The
workflow in `.github/workflows/release.yml` runs it once per release tag and uploads `dist/*` plus
`manifest.json` (file path inside the pack, asset name, size, SHA-256, MD5). A published release is
never rebuilt; a new model version gets a new tag.

Asset names are `<pack>-<path>` with `/` written as `__`
(e.g. `de-espeak-ng-data__lang__gmw__de` is `espeak-ng-data/lang/gmw/de` in the German pack).

## Licences

- Kokoro-82M and Kokoro v1.0 voices: Apache-2.0 (hexgrad/Kokoro-82M).
- sherpa-onnx conversion: Apache-2.0 (k2-fsa).
- German fine-tune: Apache-2.0 (kikiri-tts / crane-local-ai); training data Kerstin 1.0, CC0-1.0.
- eSpeak NG data: GPL-3.0-or-later.
- `build.py` and the workflow: MIT.
