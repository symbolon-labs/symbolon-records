"""Existing native text-scoring components and fixed fresh familiar-context requests."""
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'latent_binding_novel_whole_native_20260910'
BASE=ROOT.parent/'latent_binding_v1_20260906'
spec=importlib.util.spec_from_file_location('binding_for_fresh_text',BASE/'binding.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
previous=sys.modules.get('binding');sys.modules['binding']=b
try:
    spec=importlib.util.spec_from_file_location('existing_text_scoring_fresh',BASE/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
finally:
    if previous is None:sys.modules.pop('binding',None)
    else:sys.modules['binding']=previous
CAPS={'sender':None,'receiver':None}
PREVIOUS=ROOT.parent/'latent_binding_order1_text_extended_20260908'
HASHES={'sender':'deb5e69d2fcd24d139ce2927f8375c2c445867b196bc179df4688d98990e0ec4',
    'receiver':'6a3e1782ebad58d1e9323d5eeaff191d112848b5f0058394a998d1ac54c492b3'}
def requests(data):
    return [dict(row=i,condition=condition,query=row['query'],answer=row[label],source=data['records'][row['group']][field],example_id=row['id'],group=row['group'],color=row['color'])
        for condition,field,label in [('original','source','answer'),('changed','wrong_source','wrong_answer')] for i,row in enumerate(data['rows'])]
def sender_body(row):return r.SENDER_INSTRUCTION+row['source']+'\n\n'+row['query']
def receiver_body(message):
    assert message in b.IDS
    return r.RECEIVER_INSTRUCTION+'Answer ID: '+message
def message_ids(rows,predictions):
    assert len(rows)==len(predictions)==64 and all(isinstance(i,int) and 0<=i<128 for i in predictions)
    return sorted({row['answer'] for row in rows}|{b.IDS[i] for i in predictions})
