import pytest
from fastapi.testclient import TestClient
from podcast_server import config,store
from podcast_server.app import app
from .test_contract import example

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'DATA',tmp_path);monkeypatch.setattr(store,'disk_ok',lambda:True)
    model=tmp_path/'sidon';model.mkdir();(model/'feature_extractor_cuda.pt').touch();(model/'decoder_cuda.pt').touch()
    pre=tmp_path/'pre';pre.mkdir();(pre/'preprocessor_config.json').touch()
    vc=tmp_path/'vc';vc.mkdir();(vc/'python').touch();(vc/'infer.py').touch()
    monkeypatch.setenv('PODCAST_VC_PYTHON',str(vc/'python'));monkeypatch.setenv('PODCAST_VC_INFER',str(vc/'infer.py'))
    monkeypatch.setenv('PODCAST_SIDON_MODEL',str(model));monkeypatch.setenv('PODCAST_SIDON_PREPROCESSOR',str(pre))
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

def test_voice_conversion_is_opt_in_and_requires_operator_configuration(client, monkeypatch, tmp_path):
    monkeypatch.delenv('PODCAST_VC_PYTHON', raising=False)
    monkeypatch.delenv('PODCAST_VC_INFER', raising=False)
    body=example();body['options']['voice_conversion']=True
    assert client.post('/v1/validate',json=body).json()['script']['options']['voice_conversion'] is True
    assert client.post('/v1/jobs',json=body).status_code==503
    binary=tmp_path/'python';binary.touch();infer=tmp_path/'infer.py';infer.touch()
    monkeypatch.setenv('PODCAST_VC_PYTHON',str(binary));monkeypatch.setenv('PODCAST_VC_INFER',str(infer))
    assert client.post('/v1/jobs',json=body).status_code==202


def test_sidon_default_requires_assets_and_can_be_disabled(client,monkeypatch):
    monkeypatch.delenv('PODCAST_SIDON_MODEL',raising=False)
    body=example()
    assert client.post('/v1/jobs',json=body).status_code==503
    body['options']['sidon_enhancement']=False
    assert client.post('/v1/jobs',json=body).status_code==202
