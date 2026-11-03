"""Reproducible public regression evaluation; never trains on reference labels."""
import argparse
import hashlib
import json
import time
import statistics
from pathlib import Path

from recognition import recognize_file, error_counts, weight_fingerprint


def benchmark(manifest, weights=None, mode=None, decoder='greedy', engine='original'):
    manifest = Path(manifest)
    data = json.loads(manifest.read_text(encoding='utf-8'))
    rows = data['samples']
    if engine == 'auto-avsr':
        from general_bridge import WEIGHTS
        if weights is not None and Path(weights).resolve() != WEIGHTS.resolve():
            raise ValueError('The English model uses its verified installed checkpoint.')
        weights = WEIGHTS
        mode = 'mediapipe'
    if weights is None:
        raise ValueError('Provide weights for the selected engine.')
    seen = set()
    for row in rows:
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Every sample requires a nonempty reference transcript.')
        if row['sha256'] in seen:
            raise ValueError('Duplicate video content in manifest.')
        seen.add(row['sha256'])
    if not rows:
        raise ValueError('No samples')
    results = []
    fingerprint = weight_fingerprint(weights)
    for row in rows:
        path = (manifest.parent / row['video']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Sample hash mismatch: {path.name}')
        started = time.perf_counter()
        try:
            prediction = recognize_file(path, mode=mode or row['mode'], weights=None if engine == 'auto-avsr' else Path(weights), decoder=decoder, engine=engine)
            text = prediction['text']
            result = {'text': text, 'status': prediction['status']}
        except (ValueError, OSError, RuntimeError) as exc:
            # Count rejected clips as empty hypotheses, so failures cannot improve WER.
            text = ''
            result = {'text': '', 'status': 'input_failed', 'error': str(exc)}
        result.update(speaker=row.get('speaker', 'unverified'), video=row['video'], reference=row['text'], label_source=row['label_source'],
                      seconds=time.perf_counter() - started, **error_counts(row['text'], text))
        results.append(result)
        print(json.dumps(result), flush=True)
    if weight_fingerprint(weights) != fingerprint:
        raise RuntimeError('Weights changed during evaluation')
    return {'purpose': data['purpose'], 'training_overlap': data['training_overlap'],
            'speaker_identity_verified': data['speaker_identity_verified'],
            'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'weights': str(weights), 'weights_sha256': fingerprint, 'decoder': 'hybrid_ctc_attention' if engine == 'auto-avsr' else decoder, 'engine': engine,
            'mode': mode or 'manifest', 'clips': len(results),
            'wer': sum(r['word_errors'] for r in results) / sum(r['reference_words'] for r in results),
            'cer': sum(r['character_errors'] for r in results) / sum(r['reference_characters'] for r in results),
            'exact_match_rate': sum(r['exact_match'] for r in results) / len(results),
            'input_failures': sum(r['status'] == 'input_failed' for r in results),
            'first_request_seconds': results[0]['seconds'],
            'subsequent_mean_seconds': statistics.mean(r['seconds'] for r in results[1:]) if len(results) > 1 else None,
            'median_seconds': statistics.median(r['seconds'] for r in results),
            'audio_used': False,
            'mean_seconds': sum(r['seconds'] for r in results) / len(results), 'results': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/public_regression/manifest.json'))
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--mode', choices=['grid', 'mediapipe'])
    parser.add_argument('--decoder', choices=['greedy', 'beam'], default='greedy')
    parser.add_argument('--engine', choices=['original', 'lipnet-reference', 'auto-avsr'], default='original')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output filename.')
    result = benchmark(args.manifest, args.weights, args.mode, args.decoder, args.engine)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"WER {result['wer']:.1%}; CER {result['cer']:.1%}; exact {result['exact_match_rate']:.1%}")
