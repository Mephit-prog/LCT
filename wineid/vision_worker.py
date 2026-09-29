"""Private, bounded CLIP IPC worker; only started by IsolatedVisionEncoder."""
import base64
import binascii
import io
import json
import sys

from PIL import Image

from .fusion_vision import AdaptedEncoder
from .jina_clip import unit_vectors
from .roi import MAX_PIXELS
from .vision_ipc import MAX_REQUEST


def serve(encoder):
    """One canary then lossless image batches; never log private input."""
    try:
        with Image.new('RGB', (32, 32), 'white') as canary:
            unit_vectors(encoder.encode_images([canary]), 1, encoder.config['dimensions'])
        sys.stdout.write(json.dumps({'ready': True, 'config': encoder.config,
                                     'device': encoder.device}) + '\n')
        sys.stdout.flush()
        while True:
            line = sys.stdin.buffer.readline(MAX_REQUEST + 1)
            if not line:
                break
            if len(line) > MAX_REQUEST or not line.endswith(b'\n'):
                break
            images = []
            try:
                message = json.loads(line)
                if (not isinstance(message, dict) or set(message) != {'images'}
                        or not isinstance(message['images'], list)
                        or not 1 <= len(message['images']) <= 3):
                    break
                for encoded in message['images']:
                    if not isinstance(encoded, str):
                        raise ValueError('invalid image')
                    raw = base64.b64decode(encoded, validate=True)
                    with Image.open(io.BytesIO(raw)) as source:
                        if source.width * source.height > MAX_PIXELS:
                            raise ValueError('image too large')
                        images.append(source.convert('RGB'))
                vectors = encoder.encode_images(images)
                sys.stdout.write(json.dumps({'vectors': vectors.tolist()},
                                            allow_nan=False, separators=(',', ':')) + '\n')
                sys.stdout.flush()
            except (ValueError, TypeError, KeyError, OSError, binascii.Error):
                # A malformed request terminates the worker; the parent fails
                # closed and recovers asynchronously. No input/stderr is logged.
                break
            finally:
                for image in images:
                    image.close()
    finally:
        encoder.close()


def main():
    adapter, device = sys.argv[1:3]
    serve(AdaptedEncoder(adapter=adapter or None, device=device or None,
                         allow_remote_code=True))


if __name__ == '__main__':
    main()
