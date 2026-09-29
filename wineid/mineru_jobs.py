"""Durable, bounded document queue for the opt-in MinerU Precision API.

Use `python -m wineid.mineru_jobs --help`. This is NOT an OCR provider for
`/api/recognize`; no user image or PDF is ever uploaded on wine HTTP requests.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
from pathlib import Path

from .mineru_cloud import (MinerUBackend, MinerUConfig, MinerUError, OcrBackend, check_file,
                           digest_file, match_result, prepare, safe_url,
                           DOWNLOAD_HOSTS, unpack_result, verify_callback)

SCHEMA = '''CREATE TABLE IF NOT EXISTS jobs (
  key TEXT PRIMARY KEY, data_id TEXT NOT NULL, config_sha TEXT NOT NULL,
  path TEXT NOT NULL, filename TEXT NOT NULL, source_pages TEXT,
  callback_mode INTEGER NOT NULL, state TEXT NOT NULL,
  batch_id TEXT, upload_url TEXT, reserved_at REAL,
  zip_url TEXT, error_code TEXT, next_at REAL NOT NULL DEFAULT 0,
  leased_until REAL NOT NULL DEFAULT 0, updated REAL NOT NULL
)'''
SOURCES = '''CREATE TABLE IF NOT EXISTS sources (
  source_path TEXT NOT NULL, source_sha TEXT NOT NULL, source_pages TEXT NOT NULL,
  job_key TEXT NOT NULL REFERENCES jobs(key),
  PRIMARY KEY (source_path, source_pages, job_key)
)'''


class JobStore:
    def __init__(self, work_dir, *, max_active=100):
        if type(max_active) is not int or not 1 <= max_active <= 10000:
            raise ValueError('invalid queue capacity')
        self.root = Path(work_dir).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        # POSIX mode bits express privacy; Windows security lives in ACLs and
        # stat() always reports 0o666, so the strict check is POSIX-only.
        if os.name != 'nt' and stat.S_IMODE(self.root.stat().st_mode) & 0o077:
            raise ValueError('work directory must be private (chmod 700)')
        self.max_active = max_active
        self.db_path = self.root / 'jobs.sqlite3'
        fd = os.open(self.db_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        if os.name != 'nt' and stat.S_IMODE(self.db_path.stat().st_mode) & 0o077:
            raise ValueError('queue database must be private (chmod 600)')
        self._threads = threading.local()
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute(SCHEMA)
        self.db.execute(SOURCES)

    @property
    def db(self):
        # ASGI callbacks run in another thread from the CLI bootstrap; each
        # thread gets its own connection. BEGIN IMMEDIATE serializes claims.
        conn = getattr(self._threads, 'db', None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute('PRAGMA busy_timeout=10000')
            self._threads.db = conn
        return conn

    def close(self):
        conn = getattr(self._threads, 'db', None)
        if conn is not None:
            conn.close()
            self._threads.db = None

    def enqueue(self, paths, config: MinerUConfig):
        """Atomically enqueue all parts; files must persist until finished."""
        parts = []
        for original in paths:
            original = Path(original).resolve(strict=True)
            source_sha = digest_file(original)
            for path, pages in prepare(original, self.root / 'spool'):
                path = check_file(path)
                data_id = digest_file(path)
                key = hashlib.sha256((data_id + ':' + config.sha256).encode()).hexdigest()
                parts.append((key, data_id, path, original, source_sha, pages))
            if digest_file(original) != source_sha:
                raise ValueError('source changed during PDF preparation')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            active = self.db.execute("SELECT count(*) FROM jobs WHERE state NOT IN ('done','failed','needs_review','daily_quota')").fetchone()[0]
            keys = [part[0] for part in parts]
            present = {}
            for key in set(keys):
                row = self.db.execute('SELECT key,callback_mode FROM jobs WHERE key=?', (key,)).fetchone()
                if row is not None:
                    present[key] = row
            if any(bool(row['callback_mode']) != bool(config.callback) for row in present.values()):
                raise ValueError('existing job has a different callback mode')
            if active + len(set(keys) - present.keys()) > self.max_active:
                raise MinerUError('queue_full')
            for key, data_id, path, original, source_sha, pages in parts:
                self.db.execute('''INSERT OR IGNORE INTO jobs
                    (key,data_id,config_sha,path,filename,source_pages,callback_mode,state,updated)
                    VALUES (?,?,?,?,?,?,?,'queued',?)''',
                    (key, data_id, config.sha256, str(path), path.name,
                     json.dumps(pages), bool(config.callback), time.time()))
                self.db.execute('''INSERT OR IGNORE INTO sources
                    (source_path,source_sha,source_pages,job_key) VALUES (?,?,?,?)''',
                    (str(original), source_sha, json.dumps(pages), key))
            self.db.execute('COMMIT')
            return keys
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def get(self, key):
        row = self.db.execute('SELECT * FROM jobs WHERE key=?', (key,)).fetchone()
        return dict(row) if row else None

    def summary(self):
        return [dict(key=r['key'], data_id=r['data_id'], state=r['state'],
                     error_code=r['error_code'], source_pages=json.loads(r['source_pages']),
                     result_dir=str(self.root / 'results' / r['key']) if r['state'] == 'done' else None)
                for r in self.db.execute('SELECT * FROM jobs ORDER BY updated DESC')]

    def update(self, key, **fields):
        allowed = {'state', 'batch_id', 'upload_url', 'zip_url', 'reserved_at',
                   'error_code', 'next_at', 'leased_until'}
        if not fields or set(fields) - allowed:
            raise ValueError('invalid job update')
        fields['updated'] = time.time()
        cols = ','.join(f'{k}=?' for k in fields)
        self.db.execute(f'UPDATE jobs SET {cols} WHERE key=?', (*fields.values(), key))

    def claim(self, *, callback_mode, callback_stale=1800):
        """One step per invocation. Stale in-flight POST is NEVER replayed."""
        now = time.time()
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute("UPDATE jobs SET state='needs_review',error_code='reservation_interrupted' "
                            "WHERE state='reserving' AND updated < ?", (now - 180,))
            row = self.db.execute('''SELECT * FROM jobs WHERE next_at<=? AND leased_until<=? AND callback_mode=?
                AND (state IN ('queued','uploading','ready') OR
                    (state='submitted' AND (?=0 OR updated<?)))
                ORDER BY CASE state WHEN 'ready' THEN 0 WHEN 'uploading' THEN 1
                                   WHEN 'queued' THEN 2 ELSE 3 END, updated LIMIT 1''',
                (now, now, int(callback_mode), int(callback_mode), now - callback_stale)).fetchone()
            if row:
                self.db.execute('UPDATE jobs SET state=?, leased_until=?, updated=? WHERE key=?',
                                ('reserving' if row['state'] == 'queued' else row['state'],
                                 now + 600, now, row['key']))
            self.db.execute('COMMIT')
            return dict(row) if row else None
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def apply_result(self, batch_id, data_id, result, *, hosts=DOWNLOAD_HOSTS):
        """Durable callback/poll state transition keyed by *both* IDs."""
        row_result = match_result({'batch_id': batch_id, 'extract_result': [result]},
                                  batch_id, data_id, hosts=hosts)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT * FROM jobs WHERE batch_id=? AND data_id=?',
                                  (batch_id, data_id)).fetchone()
            if row is None:
                raise MinerUError('unknown_callback_job')
            if row['state'] in ('done', 'failed', 'daily_quota', 'needs_review'):
                self.db.execute('COMMIT')
                return
            if row_result['state'] == 'done':
                safe_url(row_result['full_zip_url'], hosts)
                self.update(row['key'], state='ready', zip_url=row_result['full_zip_url'],
                            upload_url=None, next_at=0)
            elif row_result['state'] == 'failed':
                self.update(row['key'], state='failed', error_code='extract_failed', upload_url=None)
            elif row['state'] == 'submitted':
                self.update(row['key'], next_at=time.time() + 10)
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise


class JobWorker:
    def __init__(self, store: JobStore, backend: OcrBackend, *, poll_interval=10):
        if not 1 <= poll_interval <= 3600:
            raise ValueError('invalid poll interval')
        self.store, self.backend, self.poll_interval = store, backend, poll_interval

    def once(self):
        job = self.store.claim(callback_mode=bool(self.backend.config.callback))
        if not job:
            return None
        key, state = job['key'], job['state']
        try:
            if state in ('uploading', 'submitted', 'ready') and job['reserved_at'] and (
                    time.time() - job['reserved_at'] > 24 * 3600):
                raise MinerUError('job_timeout')
            if state == 'queued':
                path = check_file(job['path'])
                if digest_file(path) != job['data_id'] or job['config_sha'] != self.backend.config.sha256:
                    raise MinerUError('input_or_config_changed')
                # Reserving is committed BEFORE the non-idempotent POST. If the
                # process dies, the job needs manual reconciliation, not a new POST.
                batch, url = self.backend.reserve(job['filename'], job['data_id'])
                self.store.update(key, state='uploading', batch_id=batch,
                                  upload_url=url, reserved_at=time.time())
                state = 'uploading'
                job['upload_url'], job['reserved_at'] = url, time.time()
            if state == 'uploading':
                if time.time() - job['reserved_at'] > 23 * 3600:
                    raise MinerUError('signed_upload_url_expired')
                path = check_file(job['path'])
                if digest_file(path) != job['data_id']:
                    raise MinerUError('input_or_config_changed')
                self.backend.upload(job['upload_url'], path)
                if self.store.get(key)['state'] == 'uploading':
                    self.store.update(key, state='submitted', upload_url=None,
                                      next_at=time.time() + self.poll_interval)
            elif state == 'submitted':
                result = self.backend.poll(job['batch_id'], job['data_id'])
                self.store.apply_result(job['batch_id'], job['data_id'], result,
                                        hosts=self.backend.download_hosts)
                if result['state'] not in ('done', 'failed') and self.store.get(key)['state'] == 'submitted':
                    self.store.update(key, next_at=time.time() + self.poll_interval)
            elif state == 'ready':
                archive = self.store.root / 'results' / (key + '.zip')
                archive.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                self.backend.download(job['zip_url'], archive)
                unpack_result(archive, self.store.root / 'results' / key,
                              data_id=job['data_id'], config_sha=job['config_sha'])
                self.store.update(key, state='done', zip_url=None)
        except MinerUError as exc:
            if exc.code == 'daily_quota':
                self.store.update(key, state='daily_quota', error_code=exc.code)
            elif exc.code == 'queue_full' and state == 'queued':
                self.store.update(key, state='queued', error_code=exc.code, next_at=time.time() + 60)
            elif exc.retryable and state in ('uploading', 'submitted', 'ready'):
                self.store.update(key, state=state, error_code=exc.code, next_at=time.time() + 60)
            else:
                self.store.update(key, state='needs_review', error_code=exc.code)
        finally:
            self.store.update(key, leased_until=0)
        return self.store.get(key)['state']


def create_callback_app(store: JobStore, *, uid: str, seed: str):
    """Signed callback: ACK only after durable DB update. No zip download here."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    if not uid or not seed or len(seed) < 32:
        raise ValueError('MINERU_UID and a 32+ character MINERU_CALLBACK_SEED required')
    app = FastAPI(title='MinerU callback (private ingress)')

    @app.post('/callbacks/mineru')
    async def callback(request: Request):
        if request.headers.get('content-type', '').split(';')[0].strip() != 'application/json':
            return JSONResponse({'status': 'invalid_callback'}, status_code=400)
        length = request.headers.get('content-length', '')
        if length and (not length.isdecimal() or int(length) > 70000):
            return JSONResponse({'status': 'invalid_callback'}, status_code=413)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 70000:
                return JSONResponse({'status': 'invalid_callback'}, status_code=413)
        try:
            content = verify_callback(json.loads(body), uid=uid, seed=seed)
            if not isinstance(content, dict):
                raise MinerUError('invalid_callback')
            data = content.get('data', content)
            batch = data['batch_id']
            results = data['extract_result']
            if not isinstance(batch, str) or not isinstance(results, list) or not 1 <= len(results) <= 50:
                raise MinerUError('invalid_callback')
            # One file per batch in this adapter; do not ack unknown results.
            if len(results) != 1:
                raise MinerUError('invalid_callback')
            store.apply_result(batch, results[0]['data_id'], results[0])
        except (MinerUError, KeyError, ValueError, TypeError, sqlite3.Error):
            return JSONResponse({'status': 'invalid_callback'}, status_code=400)
        return {'status': 'ok'}
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True, help='private persistent directory, not web-accessible')
    sub = parser.add_subparsers(dest='command', required=True)
    enqueue = sub.add_parser('enqueue', help='prepare and queue files locally; no network')
    enqueue.add_argument('paths', nargs='+')
    sub.add_parser('status', help='states only; never prints OCR text or signed URLs')
    work = sub.add_parser('work', help='explicitly allow egress to MinerU Precision API')
    work.add_argument('--allow-upload', action='store_true', required=True)
    work.add_argument('--loop', action='store_true', help='keep polling queue; Ctrl-C to stop')
    serve = sub.add_parser('callback', help='run signed callback receiver; use private ingress/TLS')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8081)
    for cmd in (enqueue, work):
        cmd.add_argument('--model', choices=('pipeline', 'vlm'), default='pipeline')
        cmd.add_argument('--language', choices=('cyrillic', 'east_slavic'), default='cyrillic')
        cmd.add_argument('--no-ocr', action='store_true')
        cmd.add_argument('--formula', action='store_true')
        cmd.add_argument('--no-table', action='store_true')
        cmd.add_argument('--callback-url', default=None, help='public HTTPS endpoint; requires MINERU_CALLBACK_SEED')
    args = parser.parse_args()
    store = JobStore(args.work_dir)
    try:
        if args.command == 'status':
            print(json.dumps(store.summary(), ensure_ascii=False))
            return
        if args.command == 'callback':
            import uvicorn
            app = create_callback_app(store, uid=os.environ.get('MINERU_UID'),
                                      seed=os.environ.get('MINERU_CALLBACK_SEED'))
            uvicorn.run(app, host=args.host, port=args.port)
            return
        config = MinerUConfig(model_version=args.model, language=args.language,
                               is_ocr=not args.no_ocr, enable_formula=args.formula,
                               enable_table=not args.no_table, callback=args.callback_url,
                               seed=os.environ.get('MINERU_CALLBACK_SEED') if args.callback_url else None)
        if args.command == 'enqueue':
            print(json.dumps({'keys': store.enqueue(args.paths, config)}))
        elif args.command == 'work':
            token = os.environ.get('MINERU_TOKEN')
            backend = MinerUBackend(token, config)
            try:
                while True:
                    state = JobWorker(store, backend).once()
                    print(json.dumps({'processed_state': state}))
                    if not args.loop:
                        break
                    time.sleep(5)
            finally:
                backend.close()
    finally:
        store.close()


if __name__ == '__main__':
    main()
