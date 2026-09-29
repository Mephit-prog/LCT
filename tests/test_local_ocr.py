import io
import time

import pytest
from PIL import Image

from wineid.local_ocr import LocalOCR
from wineid.ocr import MistralOCR, configured_ocr
from wineid.roi import ROI


def roi():
    image = Image.new('RGB', (40, 20), 'white')
    data = io.BytesIO()
    image.save(data, 'PNG')
    return ROI('r', 'hash', (40, 20), (0, 0, 40, 20), 'manual', 'image/png', data.getvalue())


def test_provider_selection():
    assert configured_ocr({'WINE_OCR_PROVIDER': 'none', 'MISTRAL_API_KEY': 'x'}) is None
    assert isinstance(configured_ocr({'WINE_OCR_PROVIDER': 'mistral', 'MISTRAL_API_KEY': 'x'}), MistralOCR)
    assert configured_ocr({'WINE_OCR_PROVIDER': 'mineru_local'}).model == 'mineru-pipeline-cyrillic'
    assert configured_ocr({'WINE_OCR_PROVIDER': 'mineru', 'MINERU_TOKEN': 'test'}).model == 'mineru-cloud-pipeline-cyrillic'
    with pytest.raises(ValueError):
        configured_ocr({'WINE_OCR_PROVIDER': 'mineru'})
    assert configured_ocr({'WINE_OCR_PROVIDER': 'paddle'}).model == 'PP-OCRv5-ru'
    with pytest.raises(ValueError):
        configured_ocr({'WINE_OCR_PROVIDER': 'mistral'})
    with pytest.raises(ValueError):
        configured_ocr({'WINE_OCR_PROVIDER': 'bad'})


def test_local_success_and_empty(monkeypatch):
    from wineid import local_ocr
    monkeypatch.setattr(local_ocr, '_execute', lambda *a, **kw: 'Привет 2024')
    result = LocalOCR('mineru').recognize(roi(), time.monotonic() + 5)
    assert (result.status, result.raw_text, result.model) == ('ok', 'Привет 2024', 'mineru-pipeline-cyrillic')
    monkeypatch.setattr(local_ocr, '_execute', lambda *a, **kw: '')
    assert LocalOCR('paddle').recognize(roi(), time.monotonic() + 5).status == 'no_text'


def test_local_reuses_oriented_png_without_sync_reencoding(monkeypatch):
    from wineid import local_ocr
    image = roi()
    def execute(cmd, payload, deadline):
        if payload is None:
            with open(cmd[-2], 'rb') as source:
                assert source.read() == image.crop
        else:
            assert payload is image.crop
        return 'Привет'
    monkeypatch.setattr(local_ocr, '_execute', execute)
    for backend in ('tesseract', 'paddle'):
        assert LocalOCR(backend).recognize(image, time.monotonic() + 3).status == 'ok'


def test_local_failure_and_timeout(monkeypatch):
    from wineid import local_ocr
    def fail(*args, **kwargs):
        raise OSError('bad')
    monkeypatch.setattr(local_ocr, '_execute', fail)
    result = LocalOCR('tesseract').recognize(roi(), time.monotonic() + 5)
    assert result.status == 'provider_error' and result.error_code == 'local_ocr_error'
    result = LocalOCR('paddle').recognize(roi(), time.monotonic() - 1)
    assert result.status == 'timeout' and result.raw_text is None


def test_worker_mineru_contract(tmp_path, monkeypatch):
    from wineid import local_ocr_worker as worker
    exe = tmp_path / 'mineru'; exe.touch()
    monkeypatch.setattr(worker.sys, 'executable', str(tmp_path / 'python'))
    def run(cmd, **kwargs):
        assert cmd[-6:] == ['-b', 'pipeline', '-m', 'ocr', '-l', 'cyrillic']
        out = tmp_path / 'out' / 'roi'; out.mkdir(parents=True)
        (out / 'roi.md').write_text('Текст', encoding='utf8')
    monkeypatch.setattr(worker.subprocess, 'run', run)
    assert worker.mineru(tmp_path / 'roi.png', tmp_path / 'out') == 'Текст'
