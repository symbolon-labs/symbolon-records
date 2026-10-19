"""No model inference or final data: mechanical tests for confirmation guards."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import final_cf as f


def test_components_pin_crossfamily_and_both_runtimes():
    comp=f.components()
    assert {'llama_runtime','qwen_runtime','bounded','statistics','base','text','text_cf.py','final_cf.py'}<=set(comp)
    assert all(len(x['sha256'])==64 and Path(x['path']).is_file() for x in comp.values())


def test_exclusive_preserves_first_result(tmp_path):
    dest=tmp_path/'result.json'; f.exclusive(dest,{'first':True})
    with pytest.raises(FileExistsError): f.exclusive(dest,{'first':False})
    assert json.loads(dest.read_text())=={'first':True}


def test_final_score_requires_freeze(tmp_path):
    with patch.object(f,'ROOT',tmp_path):
        with pytest.raises(FileNotFoundError): f.read_freeze()
    assert not (tmp_path/'run_final').exists()


def test_preregistered_checkpoint_selection():
    def m(ce): return {'conditions':{'correct':{'full_vocab_ce':ce}}}
    assert f.analysis.select_checkpoint([(0,m(.01)),(50,m(2)),(100,m(1)),(150,m(1))])==100
