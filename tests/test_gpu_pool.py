import concurrent.futures
from unittest.mock import patch
from podcast_server import store
from podcast_server.supervisor import Supervisor, gpu_inventory


def test_inventory_and_busy_cards(monkeypatch):
    replies = iter(['0, GPU-a, 81919\n1, GPU-b, 70000\n', 'GPU-b, 1234\n'])
    with patch('subprocess.check_output', side_effect=lambda *a, **kw: next(replies)):
        cards = gpu_inventory()
    assert cards == [(0, 'GPU-a', 81919, False), (1, 'GPU-b', 70000, True)]
    monkeypatch.setenv('PODCAST_GPU_POOL', 'auto')
    with patch('podcast_server.supervisor.gpu_inventory', return_value=cards):
        assert Supervisor().available('cuda:0')
        assert not Supervisor().available('cuda:1')


def test_concurrent_claims_are_unique(tmp_path, monkeypatch):
    monkeypatch.setattr(store.config, 'DATA', tmp_path)
    store.init()
    with store.connect() as c:
        for n in range(16):
            c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)', (str(n), '', '', 'queued', n, n, 0, None, ''))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _: store.claim(), range(16)))
    assert len({r['id'] for r in rows}) == 16
    assert all(r['state'] == 'queued' for r in rows)
