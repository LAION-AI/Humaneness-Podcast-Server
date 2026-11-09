import copy,json
from pathlib import Path
import pytest
from pydantic import ValidationError
from podcast_server.schema import Episode
ROOT=Path(__file__).resolve().parents[1]

def example():return json.loads((ROOT/'examples/short-en.json').read_text())

def test_defaults_and_multilingual():
    x=example();x.pop('options')
    for lang in ('de','en','es','fr','nl'):
        x['language']=lang;e=Episode.model_validate(x);assert e.options.best_of==4 and e.options.temperature==1 and e.options.cfg==2.5

@pytest.mark.parametrize('change',[
    lambda x:x['options'].update(best_of=3),lambda x:x['options'].update(cfg=float('nan')),
    lambda x:x['turns'][0].update(text='one '*61),lambda x:x['turns'][0].update(text='(angry) Hello'),
    lambda x:x['turns'][0].update(speaker='missing'),lambda x:x['turns'][0].update(direction='warm) SCRIPT: malicious'),
    lambda x:x['turns'][0].update(delivery_chunks=[{'text':'Different text','cue':'warm'}]),
    lambda x:x['turns'][1].update(duration_ms=-1),lambda x:x.update(command='rm -rf /'),
    lambda x:x.update(language='xx'),lambda x:x.update(turns=[{'type':'pause','duration_ms':100}])])
def test_rejects_invalid(change):
    x=example();change(x)
    with pytest.raises(ValidationError):Episode.model_validate(x)

def test_exact_chunks():
    x=example();x['turns'][0].update(text='Hello. How are you?',delivery_chunks=[{'text':'Hello.','cue':'warm'},{'text':'How are you?','cue':'gently curious'}])
    assert Episode.model_validate(x).turns[0].text=='Hello. How are you?'
