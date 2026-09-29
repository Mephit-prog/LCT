"""Private stdin/stdout protocol for one long-lived PaddleOCR process (not a server).

Only local, preinstalled model directories are accepted. stdout is reserved for
bounded JSON lines; model output and diagnostics are never forwarded to the API.
"""
import base64
import contextlib
import io
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

MAX_INPUT = 17 * 1024 * 1024  # base64 of MAX_CROP_BYTES plus framing
MAX_OUTPUT = 2 * 1024 * 1024


def main():
    det = os.environ['WINE_PADDLE_DET_DIR']
    rec = os.environ['WINE_PADDLE_REC_DIR']
    if not Path(det).is_dir() or not Path(rec).is_dir():
        raise ValueError('local Paddle model directories required')
    expected = json.loads(os.environ['WINE_OCR_EXPECTED_PACKAGES'])
    if any(importlib.metadata.version(name) != version for name, version in expected.items()):
        raise ValueError('OCR runtime version mismatch')
    # Paddle may print during import/initialization/prediction. Do not corrupt IPC.
    with contextlib.redirect_stdout(sys.stderr):
        from paddleocr import PaddleOCR
        engine = PaddleOCR(lang='ru', ocr_version='PP-OCRv5',
                           text_detection_model_dir=det, text_recognition_model_dir=rec,
                           use_doc_orientation_classify=False, use_doc_unwarping=False,
                           use_textline_orientation=False)

    def predict(path):
        with contextlib.redirect_stdout(sys.stderr):
            pages = engine.predict(str(path))
            lines = []
            for page in pages:
                data = page.json
                content = data.get('res', data)
                lines.extend(x for x in content['rec_texts'] if isinstance(x, str))
            return '\n'.join(lines)

    # A real inference, not merely an import check; no request-time weight fetches.
    with tempfile.TemporaryDirectory(prefix='wine-paddle-') as tmp:
        path = Path(tmp) / 'roi.png'
        sample = Image.new('RGB', (320, 80), 'white')
        ImageDraw.Draw(sample).text((10, 10), 'WINE 2024', fill='black')
        sample.save(path)
        predict(path)
        sys.stdout.write('{"ready":true}\n')
        sys.stdout.flush()
        while True:
            wire = sys.stdin.buffer.readline(MAX_INPUT + 1)
            if not wire:
                return
            if len(wire) > MAX_INPUT or not wire.endswith(b'\n'):
                return  # corrupt protocol: parent will kill/restart this worker
            try:
                request = json.loads(wire)
                if request.get('op') != 'recognize':
                    raise ValueError('invalid operation')
                payload = base64.b64decode(request['image'], validate=True)
                if len(payload) > 12 * 1024 * 1024:
                    raise ValueError('image too large')
                # ROI is already validated and sanitized by roi.py. Decode once more
                # in the isolated process, never pass arbitrary input paths to Paddle.
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    if image.format != 'PNG' or image.width * image.height > 6_000_000:
                        raise ValueError('invalid crop')
                    image.convert('RGB').save(path, 'PNG')
                text = predict(path)
                reply = {'text': text}
            except Exception:
                reply = {'error': 'inference_failed'}
            output = (json.dumps(reply, ensure_ascii=False) + '\n').encode('utf-8')
            if len(output) > MAX_OUTPUT:
                output = b'{"error":"output_too_large"}\n'
            sys.stdout.buffer.write(output)
            sys.stdout.buffer.flush()
            path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
