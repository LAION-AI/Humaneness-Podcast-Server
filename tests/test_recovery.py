import json
from pathlib import Path
from podcast_server.worker import invalidate_candidate

class Reporter:
    def __init__(self):self.events=[]
    def event(self,kind,**data):self.events.append((kind,data))

def test_only_invalid_candidate_removed_and_retries_bounded(tmp_path):
    import pytest
    good=tmp_path/'good.safetensors';good.write_bytes(b'unchanged')
    bad=tmp_path/'bad.safetensors';bad.write_bytes(b'bad');bad.with_suffix('.json').write_text('{}');r=Reporter()
    invalidate_candidate(bad,r,2,1)
    assert good.read_bytes()==b'unchanged' and not bad.exists()
    assert json.loads((tmp_path/'regenerations.json').read_text())['2-1']==1
    invalidate_candidate(bad,r,2,1)
    with pytest.raises(RuntimeError):invalidate_candidate(bad,r,2,1)
    assert len(r.events)==2
