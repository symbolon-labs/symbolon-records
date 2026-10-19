import json
import pytest
import qualify as q


def test_pinned_revisions():
    assert all(len(revision)==40 for _,revision in q.MODELS.values())
    assert len(q.SMOL_WEIGHT)==64


def test_comparison_fails_on_wrong_predictions_or_scores():
    a=[{'prediction':0,'answer_log_probabilities':[-1.,-2.,-3.,-4.]}]
    assert q.compare(a,a)['pass_check']
    assert not q.compare(a,[{**a[0],'prediction':1}])['pass_check']
    assert not q.compare(a,[{**a[0],'answer_log_probabilities':[-1.1,-2.,-3.,-4.]}])['pass_check']


def test_exclusive_receipts(tmp_path):
    p=tmp_path/'receipt.json'; q.exclusive(p,{'first':True})
    with pytest.raises(FileExistsError): q.exclusive(p,{'first':False})
    assert json.loads(p.read_text())=={'first':True}
