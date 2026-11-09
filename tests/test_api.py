import pytest
from fastapi.testclient import TestClient
from podcast_server import config,store
from podcast_server.app import app
from .test_contract import example

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'DATA',tmp_path);monkeypatch.setattr(store,'disk_ok',lambda:True)
    monkeypatch.setattr(config,'assets',lambda:{**{k:str(tmp_path) for k in ('model','codec','asr','aligner')},'voices':{'reference_s1':{'label':'fixture','sha256':'a'},'reference_s2':{'label':'fixture','sha256':'b'}}})
    store.init()
    # No lifespan = no GPU worker. This exercises the public HTTP contract only.
    yield TestClient(app)

def test_submit_poll_isolation_cancel(client):
    response=client.post('/v1/jobs',json=example(),headers={'Idempotency-Key':'abc'});assert response.status_code==202
    r=response.json();url='/v1/jobs/'+r['job_id'];headers={'Authorization':'Bearer '+r['job_token']}
    assert client.get(url).status_code==404
    assert client.get(url,headers=headers).json()['state']=='queued'
    assert client.get(url+'/audio',headers=headers).status_code==409
    assert client.post(url+'/cancel',headers=headers).status_code==200
    assert client.get(url,headers=headers).json()['state']=='cancelled'
    assert client.delete(url,headers=headers).status_code==200
    assert client.get(url,headers=headers).status_code==404

def test_validation_and_oversize(client):
    x=example();x['speakers']['host']='unknown'
    assert client.post('/v1/validate',json=x).status_code==422
    assert client.post('/v1/jobs',content=b'x'*(512*1024+1)).status_code==413
    assert client.get('/openapi.json').status_code==200
