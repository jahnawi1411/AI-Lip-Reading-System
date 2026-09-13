"""Local visual-only Auto-AVSR adapter; no audio or transcript enters inference."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor/auto_avsr'
WEIGHTS = ROOT / 'models/auto-avsr/vsr_trlrs2lrs3vox2avsp_base.pth'


def face_anchors(frame, landmarks):
    height, width = frame.shape[:2]
    points = np.asarray([(p.x * width, p.y * height) for p in landmarks])
    anchors = np.stack([points[[33,133]].mean(axis=0), points[[362,263]].mean(axis=0),
                        points[1], points[[13,14]].mean(axis=0)])
    if not np.isfinite(anchors).all():
        raise ValueError('Invalid facial landmarks')
    return anchors, (0, 0, width, height)


def prepare_sentence(path, start=0.0, duration=None):
    """Preserve timing at 25 FPS; align face, then crop 96 and center-crop 88."""
    from preprocessing import LipDetector
    if not np.isfinite(start) or not 0 <= start <= 60:
        raise ValueError('Start must be between 0 and 60 seconds.')
    if duration is not None and (not np.isfinite(duration) or not 0 < duration <= 16):
        raise ValueError('Duration must be positive and at most 16 seconds.')
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not np.isfinite(fps) or not 1 <= fps <= 120:
            raise ValueError('Video could not be decoded with a valid frame rate.')
        first = int(np.ceil(start * fps))
        stop = int(np.ceil((start + duration) * fps)) if duration is not None else None
        images = []
        index = 0
        while stop is None or index < stop:
            ok, frame = cap.read()
            if not ok: break
            if frame.shape[0] * frame.shape[1] > 3840 * 2160:
                raise ValueError('Video exceeds 4K resolution.')
            if index >= first:
                ratio = min(1.0, 640 / frame.shape[1], 480 / frame.shape[0])
                if ratio < 1:
                    frame = cv2.resize(frame, (round(frame.shape[1]*ratio), round(frame.shape[0]*ratio)))
                images.append(frame)
                if len(images) / fps > 16:
                    raise ValueError('Choose a speaking interval of at most 16 seconds.')
            index += 1
        if stop is not None and index < stop:
            raise ValueError('Selected interval extends beyond the video.')
    finally:
        cap.release()
    if len(images) < 2:
        raise ValueError('Too few readable frames.')
    indices = np.minimum(np.rint(np.arange(max(2, round(len(images)/fps*25))) * fps/25).astype(int), len(images)-1)
    # Detect once per unique source frame, preserving source-time timestamps.
    anchors = {}
    with LipDetector(cropper=face_anchors) as detector:
        for i in sorted(set(indices.tolist())):
            anchors[i] = detector.detect(images[i], round(i*1000/fps))[0]
    points = np.stack([anchors[i] for i in indices])
    reference = np.load(VENDOR / 'preparation/detectors/mediapipe/20words_mean_face.npy', allow_pickle=False)
    stable = np.stack([reference[36:42].mean(axis=0), reference[42:48].mean(axis=0),
                       reference[31:36].mean(axis=0), reference[48:68].mean(axis=0)])
    crops = []
    for i, source_index in enumerate(indices):
        margin = min(6, i, len(points)-1-i)
        smoothed = points[i-margin:i+margin+1].mean(axis=0)
        smoothed += points[i].mean(axis=0) - smoothed.mean(axis=0)
        matrix, _ = cv2.estimateAffinePartial2D(smoothed, stable, method=cv2.LMEDS)
        if matrix is None or not np.isfinite(matrix).all():
            raise ValueError('Face alignment failed.')
        warped = cv2.warpAffine(images[source_index], matrix, (256,256))
        mouth = smoothed[3] @ matrix[:,:2].T + matrix[:,2]
        x,y = np.rint(mouth).astype(int)
        if not 48 <= x <= 208 or not 48 <= y <= 208:
            raise ValueError('Mouth is outside the aligned image.')
        rgb = cv2.cvtColor(warped[y-48:y+48,x-48:x+48], cv2.COLOR_BGR2RGB)
        crops.append(rgb[4:92,4:92])
    values = np.stack(crops).astype(np.float32)/255
    gray = values @ np.asarray([.2989,.587,.114], np.float32)
    from input_quality import assess_crops
    quality = assess_crops(np.clip(gray*255,0,255).astype(np.uint8))
    return ((gray[:,None]-.421)/.165).astype(np.float32), {
        'preprocessing':'auto-avsr-tasks-alignment-v1', 'frames':len(indices),
        'source_fps':fps, 'target_fps':25, 'duration_seconds':len(images)/fps,
        'start_seconds':first/fps, 'audio_used':False, 'input_quality':quality,
        'warnings':['Face Landmarker anchors replace the upstream face detector; this adaptation needs representative evaluation.']}


def recognize_sentence(path, start=0.0, duration=None, beam_size=10):
    import torch
    if not 1 <= beam_size <= 40:
        raise ValueError('Beam size must be between 1 and 40.')
    sys.path.insert(0, str(VENDOR)) if str(VENDOR) not in sys.path else None
    from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.scorers.length_bonus import LengthBonus
    torch.set_num_threads(4)
    started = time.perf_counter()
    frames, report = prepare_sentence(path,start,duration)
    units = (VENDOR/'spm/unigram/unigram5000_units.txt').read_text(encoding='utf-8').splitlines()
    tokens = ['<blank>'] + [line.split()[0] for line in units] + ['<eos>']
    # Only tensor state is accepted: never fall back to unrestricted pickle loading.
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=True, mmap=True)
    model = E2E(len(tokens), 'video').eval()
    model.load_state_dict(state, strict=True)
    del state
    provenance = json.loads((WEIGHTS.parent/'PROVENANCE.json').read_text(encoding='utf-8'))
    scorers = model.scorers()
    scorers.update(lm=None,length_bonus=LengthBonus(len(tokens)))
    search = BatchBeamSearch(beam_size=beam_size,vocab_size=len(tokens),
        weights={'decoder':.9,'ctc':.1,'lm':0.,'length_bonus':0.},scorers=scorers,
        sos=len(tokens)-1,eos=len(tokens)-1,token_list=tokens,pre_beam_score_key='decoder')
    with torch.inference_mode():
        features = model.frontend(torch.from_numpy(frames)[None])
        encoded,_ = model.encoder(model.proj_encoder(features),None)
        hypotheses = search(encoded.squeeze(0))
    if not hypotheses:
        text = ''
    else:
        ids = hypotheses[0].yseq.tolist()[1:]
        text = ''.join(tokens[i] for i in ids if i not in {0,len(tokens)-1}).replace('\u2581',' ').strip()
    result = {'text':text, 'usable_text':text or None, 'status':'decoded' if text else 'no_speech_decoded',
        'engine':'auto-avsr','domain':'English sentences (experimental)','audio_used':False,
        'review_required':True,'confidence':None,'model_score':None,'confidence_status':'uncalibrated',
        'weights_sha256':provenance['sha256'],'model_revision':provenance['revision'],
        'message':'Experimental visual-only English candidate. Review every word, especially names.',
        'decoder':'hybrid_ctc_attention','beam_size':beam_size,'input_report':report,
        'input_quality':report['input_quality'],
        'warnings':report['warnings']+report['input_quality']['warnings']+['General-speech accuracy has not been established on this application.'],
        'elapsed_seconds':time.perf_counter()-started}
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--start',type=float,default=0.)
    parser.add_argument('--duration',type=float)
    parser.add_argument('--beam-size',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Choose a new output filename.')
    result=recognize_sentence(args.video,args.start,args.duration,args.beam_size)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
