"""Isolated OCR invocation; invoked by LocalOCR, not an HTTP entry point."""
import contextlib
import subprocess
import sys
from pathlib import Path


def paddle(image):
    # Paddle's initialization may write progress to stdout; reserve stdout for JSON.
    with contextlib.redirect_stdout(sys.stderr):
        from paddleocr import PaddleOCR
        engine = PaddleOCR(lang='ru', ocr_version='PP-OCRv5',
                           use_doc_orientation_classify=False, use_doc_unwarping=False,
                           use_textline_orientation=False)
        result = engine.predict(str(image))
    lines = []
    for page in result:
        data = page.json
        content = data.get('res', data)
        lines.extend(x for x in content.get('rec_texts', []) if isinstance(x, str))
    return '\n'.join(lines)


def mineru(image, output):
    # Pipeline (not VLM) and explicit Cyrillic; consult pinned MinerU CLI version
    # when preparing the server image. OCR mode forces OCR even on born-digital PDF.
    executable = Path(sys.executable).parent / 'mineru'
    if not executable.is_file():
        raise FileNotFoundError('mineru executable not installed in OCR environment')
    subprocess.run([str(executable), '-p', str(image), '-o', str(output), '-b', 'pipeline',
                    '-m', 'ocr', '-l', 'cyrillic'], check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    files = sorted(output.rglob('*.md'))
    if len(files) != 1:
        raise ValueError('expected one MinerU markdown output')
    if files[0].stat().st_size > 2 * 1024 * 1024:
        raise ValueError('MinerU markdown output too large')
    return files[0].read_text(encoding='utf-8')


def main():
    backend, source, target = sys.argv[1:]
    image, output = Path(source), Path(target)
    if backend == 'paddle':
        text = paddle(image)
    elif backend == 'mineru':
        text = mineru(image, output)
    else:
        raise ValueError('unknown backend')
    sys.stdout.write(text)


if __name__ == '__main__':
    main()
