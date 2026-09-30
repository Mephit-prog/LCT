from build_windows_bundle import assemble


def test_bundle_single_backend_and_no_local_env(tmp_path):
    source = tmp_path / 'src'
    for name in ('wineid', 'artifacts', 'web', 'local'):
        (source / name).mkdir(parents=True)
    (source / 'wineid' / 'api.py').write_text('test')
    (source / 'artifacts' / 'classes-lora-canonical.npz').write_bytes(b'weight')
    for name in ('strapi_output0709.csv', 'producer_aliases.txt', 'constraints.txt',
                 'requirements-api.txt', 'requirements-clip.txt', 'requirements-vision.txt',
                 'RUN_GUIDE.md'):
        (source / name).write_text(name)
    (source / 'web' / 'package-lock.json').write_text('{}')
    (source / 'web' / '.env').write_text('VITE_TOKEN=do-not-ship')
    (source / 'web' / 'node_modules').mkdir()
    (source / 'web' / 'node_modules' / 'old.js').write_text('old')
    for name in ('.vite', 'coverage', 'dist'):
        (source / 'web' / name).mkdir()
        (source / 'web' / name / 'generated.js').write_text('generated')
    (source / 'web' / 'tsconfig.app.tsbuildinfo').write_text('cache')
    (source / 'web' / 'debug.log').write_text('debug')
    (source / 'local' / 'run-local.ps1').write_text('test')
    (source / 'local' / 'logs').mkdir()
    (source / 'local' / 'logs' / 'backend.log').write_text('secret')
    out = tmp_path / 'bundle'
    assemble(source, out)
    assert (out / 'backend' / 'wineid' / 'api.py').read_text() == 'test'
    assert (out / 'backend' / 'artifacts' / 'classes-lora-canonical.npz').read_bytes() == b'weight'
    assert (out / 'web' / 'package-lock.json').is_file()
    assert (out / 'local' / 'run-local.ps1').is_file()
    assert not (out / 'web' / '.env').exists()
    assert not (out / 'web' / 'node_modules').exists()
    assert not (out / 'local' / 'logs').exists()
    for name in ('.vite', 'coverage', 'dist', 'tsconfig.app.tsbuildinfo', 'debug.log'):
        assert not (out / 'web' / name).exists()
    assert not (out / 'wineid').exists()
    # In-place assembly must not copy source trees onto themselves.
    assemble(source, source)
    assert (source / 'backend' / 'wineid' / 'api.py').read_text() == 'test'
