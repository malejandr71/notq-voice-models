#!/usr/bin/env python3
"""
Builds the Notq offline voice packs (Kokoro for sherpa-onnx 1.13.8) from their published sources.

  en: Kokoro v1.0 int8 (k2-fsa/sherpa-onnx tts-models/kokoro-int8-multi-lang-v1_0), English voices only
      (af_heart, af_bella, af_sarah, bf_emma, am_michael, am_fenrir, bm_george), US-English lexicon.
  de: Kokoro-82M German fine-tune, voice df_kerstin (crane-local-ai/Kokoro-82M-v1.0-German-ONNX,
      trained with kikiri-tts on Kerstin 1.0, CC0), sherpa-onnx metadata added, dynamic int8 quantisation.

Each pack also carries the minimal eSpeak NG data for its language. Every input is checked against a
pinned SHA-256. Output: dist/<pack>-<path with / replaced by __> plus dist/manifest.json.

usage: python3 build.py [workdir]
"""
import hashlib, json, os, shutil, sys, tarfile, urllib.request
import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

WORK = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'work')
DIST = os.path.abspath('dist')
SRC = {
  'int8v10': ('https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_0.tar.bz2',
              '4c3052abaa60943a341f193888cf6abd68787dae6ab8ae5c925a706caa247e4e'),
  'de_model': ('https://huggingface.co/crane-local-ai/Kokoro-82M-v1.0-German-ONNX/resolve/main/onnx/model.onnx',
               'f541f868499a8186803c72166a09469821e971dbb1e4cd8eb7dcf593f49da0b1'),
  'de_voice': ('https://huggingface.co/crane-local-ai/Kokoro-82M-v1.0-German-ONNX/resolve/main/voices/df_kerstin.bin',
               'f96576f656c58221b42ce46c74398dc2a6ffc03b8408abe530d85ebd717ca062'),
}
EN_ALL = ('af_alloy,af_aoede,af_bella,af_heart,af_jessica,af_kore,af_nicole,af_nova,af_river,af_sarah,af_sky,am_adam,'
          'am_echo,am_eric,am_fenrir,am_liam,am_michael,am_onyx,am_puck,am_santa,bf_alice,bf_emma,bf_isabella,bf_lily,'
          'bm_daniel,bm_fable,bm_george,bm_lewis').split(',')
EN_KEEP = ['af_heart', 'af_bella', 'af_sarah', 'bf_emma', 'am_michael', 'am_fenrir', 'bm_george']
ESPEAK_COMMON = ['phontab', 'phonindex', 'phondata', 'intonations']

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()

def fetch(key):
    url, digest = SRC[key]
    path = os.path.join(WORK, 'src', key + ('.tar.bz2' if url.endswith('.tar.bz2') else ''))
    if not (os.path.exists(path) and sha256(path) == digest):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    actual = sha256(path)
    if actual != digest: raise SystemExit(f'{key}: sha256 {actual} != {digest}')
    return path

def set_meta(model, values):
    existing = {p.key: p for p in model.metadata_props}
    for k, v in values.items():
        if k in existing: existing[k].value = v
        else:
            p = model.metadata_props.add(); p.key = k; p.value = v

def speakers(names):
    return {'n_speakers': str(len(names)), 'speaker_names': ','.join(names),
            'speaker2id': ','.join(f'{n}->{i}' for i, n in enumerate(names)),
            'id2speaker': ','.join(f'{i}->{n}' for i, n in enumerate(names))}

def main():
    os.makedirs(WORK, exist_ok=True)
    shutil.rmtree(DIST, ignore_errors=True); os.makedirs(DIST)
    tar = fetch('int8v10')
    base = os.path.join(WORK, 'kokoro-int8-multi-lang-v1_0')
    if not os.path.isdir(base):
        with tarfile.open(tar) as t: t.extractall(WORK, filter='data')
    espeak = os.path.join(base, 'espeak-ng-data')
    packs = {'en': {}, 'de': {}}

    # English: Kokoro v1.0 int8 with seven English voices.
    en = os.path.join(WORK, 'en'); shutil.rmtree(en, ignore_errors=True); os.makedirs(en)
    m = onnx.load(os.path.join(base, 'model.int8.onnx'))
    set_meta(m, {**speakers(EN_KEEP), 'language': 'English',
                 'comment': 'Kokoro v1.0 int8 (sherpa-onnx kokoro-int8-multi-lang-v1_0), English voices only'})
    onnx.save(m, os.path.join(en, 'model.int8.onnx'))
    v = np.fromfile(os.path.join(base, 'voices.bin'), dtype=np.float32).reshape(-1, 510, 256)
    np.ascontiguousarray(np.stack([v[EN_ALL.index(n)] for n in EN_KEEP]).astype('<f4')).tofile(os.path.join(en, 'voices.bin'))
    for f in ['tokens.txt', 'lexicon-us-en.txt']: shutil.copy(os.path.join(base, f), en)
    for f in ESPEAK_COMMON + ['en_dict', 'lang/gmw/en', 'lang/gmw/en-US']:
        os.makedirs(os.path.dirname(os.path.join(en, 'espeak-ng-data', f)), exist_ok=True)
        shutil.copy(os.path.join(espeak, f), os.path.join(en, 'espeak-ng-data', f))

    # German: Kerstin fine-tune, metadata for sherpa-onnx, dynamic int8 quantisation.
    de = os.path.join(WORK, 'de'); shutil.rmtree(de, ignore_errors=True); os.makedirs(de)
    m = onnx.load(fetch('de_model'))
    del m.metadata_props[:]
    set_meta(m, {'model_type': 'kokoro', 'language': 'German', 'has_espeak': '1', 'sample_rate': '24000', 'version': '2',
                 'voice': 'de', 'style_dim': '510,1,256', **speakers(['df_kerstin']),
                 'model_url': 'https://huggingface.co/crane-local-ai/Kokoro-82M-v1.0-German-ONNX',
                 'comment': 'Kokoro-82M German fine-tune (kikiri-tts), voice df_kerstin (Kerstin 1.0, CC0-1.0)'})
    fp32 = os.path.join(WORK, 'de-model.fp32.onnx')
    onnx.save(m, fp32)
    quantize_dynamic(fp32, os.path.join(de, 'model.int8.onnx'), weight_type=QuantType.QUInt8)
    shutil.copy(fetch('de_voice'), os.path.join(de, 'voices.bin'))
    shutil.copy(os.path.join(base, 'tokens.txt'), de)
    for f in ESPEAK_COMMON + ['de_dict', 'lang/gmw/de']:
        os.makedirs(os.path.dirname(os.path.join(de, 'espeak-ng-data', f)), exist_ok=True)
        shutil.copy(os.path.join(espeak, f), os.path.join(de, 'espeak-ng-data', f))

    manifest = {}
    for pack, root in [('en', en), ('de', de)]:
        files = []
        for dirpath, _, names in os.walk(root):
            for n in sorted(names):
                rel = os.path.relpath(os.path.join(dirpath, n), root).replace(os.sep, '/')
                asset = f'{pack}-' + rel.replace('/', '__')
                shutil.copy(os.path.join(root, rel), os.path.join(DIST, asset))
                p = os.path.join(DIST, asset)
                files.append({'file': rel, 'asset': asset, 'size': os.path.getsize(p), 'sha256': sha256(p),
                              'md5': hashlib.md5(open(p, 'rb').read()).hexdigest()})
        manifest[pack] = sorted(files, key=lambda f: f['file'])
    with open(os.path.join(DIST, 'manifest.json'), 'w') as f: json.dump(manifest, f, indent=2)
    for pack, files in manifest.items():
        print(pack, len(files), 'files', sum(f['size'] for f in files), 'bytes')

if __name__ == '__main__':
    main()
