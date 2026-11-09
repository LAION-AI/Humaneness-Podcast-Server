"""No model mocks in production; these tests exercise the durable job contract."""
import json,time
import pytest
from podcast_server import config,store
from podcast_server.schema import Episode
from .test_contract import example

@pytest.fixture
def db(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'DATA',tmp_path);monkeypatch.setattr(store,'disk_ok',lambda:True);store.init();return tmp_path

def test_capability_idempotency_and_conflict(db):
    e=Episode.model_validate(example());r,reused=store.create(e,'client','same-key');assert not reused
    r2,reused=store.create(e,'client','same-key');assert reused and r==r2
    assert store.auth(r['job_id'],r['job_token']) and not store.auth(r['job_id'],'wrong')
    e.title='Changed'
    with pytest.raises(ValueError):store.create(e,'client','same-key')
    assert store.auth('../passwd','x') is None

def test_restart_requeues_and_retains_checkpoint(db):
    r,_=store.create(Episode.model_validate(example()),'client');job=r['job_id'];assert store.claim()['id']==job
    (store.folder(job)/'checkpoint').write_text('committed data');store.init()
    assert store.auth(job,r['job_token'])['state']=='queued';assert (store.folder(job)/'checkpoint').read_text()=='committed data'
    assert store.claim()['attempts']==1

def test_queue_limit_and_terminal_retention(db,monkeypatch):
    monkeypatch.setattr(config,'MAX_QUEUE',1);e=Episode.model_validate(example());r,_=store.create(e,'client')
    with pytest.raises(OverflowError):store.create(e,'other')
    store.update(r['job_id'],'completed');store.create(e,'other')
    with store.connect() as c:c.execute('UPDATE jobs SET updated=? WHERE id=?',(time.time()-100*3600,r['job_id']))
    store.cleanup();assert not store.folder(r['job_id']).exists()
