import hashlib
import io
import json
import os
import subprocess
import time

import pytest
from PIL import Image

from wineid.catalog import Wine, normalize
from wineid.ocr import MockOCR, OCRResult
from wineid.ocr_eval import evaluate, prepare_manifest
from wineid.local_ocr import LocalOCR
from wineid.roi import full_frame_roi
from wineid.search import Index


def sample(tmp_path):
    image = tmp_path / 'query.png'
    buf = io.BytesIO()
    Image.new('RGB', (120, 80), 'white').save(buf, 'PNG')
    image.write_bytes(buf.getvalue())
    wine = Wine('a', 'a', 'Алиготе', 'Завод', normalize('Алиготе'),
                normalize('Завод'), (2,), {}, 'sha')
    row = {'query_id': 'q1', 'image_path': 'query.png',
           'image_sha256': hashlib.sha256(buf.getvalue()).hexdigest(),
           'source_group': 'shoot-1', 'split': 'validation', 'bbox': [0, 0, 120, 80],
           'transcription': 'АлИГОтё', 'gt_status': 'known', 'gt_slug': 'a',
           'gt_source': 'human'}
    manifest = tmp_path / 'queries.jsonl'
    manifest.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
    return manifest, Index([wine]), row


def test_ocr_report_counts_failures_and_private_output(tmp_path):
    manifest, index, row = sample(tmp_path)
    class NoText:
        model = 'fixture'
        def recognize(self, roi, deadline):
            return OCRResult(roi.roi_id, 'no_text', '', '', self.model, 1)
    out = tmp_path / 'private.jsonl'
    metrics = evaluate(manifest, index, NoText(), output=out)
    assert metrics['count'] == metrics['known'] == 1
    assert metrics['cer']['value'] == metrics['wer']['value'] == 1
    assert metrics['recall_at_5'] == {'ocr': 0, 'manual_text_oracle': 1, 'denominator': 1}
    assert metrics['usable_text'] == {'numerator': 0, 'denominator': 1}
    if os.name != 'nt':  # Windows st_mode does not expose POSIX 0600 bits
        assert out.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        evaluate(manifest, index, NoText(), output=out)
    assert evaluate(manifest, index, MockOCR('Алиготе'))['cer']['value'] == 0
    row['transcription'] = ''
    manifest.write_text(json.dumps(row) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='transcription'):
        prepare_manifest(manifest, index)


def test_manifest_audited_before_split_and_image_read(tmp_path):
    manifest, index, row = sample(tmp_path)
    invalid = {**row, 'query_id': 'q2', 'split': 'test', 'source_group': 'shoot-1'}
    manifest.write_text('\n'.join(json.dumps(r) for r in (row, invalid)))
    with pytest.raises(ValueError, match='across splits'):
        prepare_manifest(manifest, index)
    manifest.write_text(json.dumps({**row, 'image_path': '../outside.png'}))
    with pytest.raises(ValueError, match='unsafe'):
        prepare_manifest(manifest, index)


@pytest.mark.skipif(os.environ.get('WINE_SMOKE_OCR') != '1', reason='opt-in local rus+eng CLI canary')
def test_real_tesseract_canary():
    # Run only on the target server with installed, pinned rus+eng traineddata.
    lang = subprocess.run(['tesseract', '--list-langs'], capture_output=True, text=True, check=True)
    if not {'rus', 'eng'} <= set(lang.stdout.splitlines() + lang.stderr.splitlines()):
        pytest.skip('rus+eng traineddata not installed')
    worker = LocalOCR('tesseract')
    worker.start()
    assert worker.ready
    buf = io.BytesIO()
    Image.new('RGB', (200, 100), 'white').save(buf, 'PNG')
    result = worker.recognize(full_frame_roi(buf.getvalue(), time.monotonic() + 3),
                              time.monotonic() + 3)
    assert result.status in ('ok', 'no_text')
