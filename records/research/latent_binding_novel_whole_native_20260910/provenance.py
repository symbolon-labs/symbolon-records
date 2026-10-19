"""Exact historical provenance with a single explicitly registered kernel migration."""
import copy
OLD='Linux-7.2.2-arch1-1-x86_64-with-glibc2.44'
NEW='Linux-7.2.3-arch1-2-x86_64-with-glibc2.44'
def compare(actual,expected,path):
    actual=copy.deepcopy(actual)
    node=actual;reference=expected
    for key in path[:-1]:node=node[key];reference=reference[key]
    key=path[-1]
    assert node[key]==NEW and reference[key]==OLD
    node[key]=reference[key]
    assert actual==expected
def seal(b,root,digest):
    assert b.sha(root/'seal.json')==digest
    files=b.read(root/'seal.json')['files']
    assert set(files)=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'seal.json'}
    assert all(b.sha(root/n)==h for n,h in files.items())
def check(span):
    pair=span.previous;mode=pair.p.parent;b=span.b
    mode.source_check()
    compare(mode.identity(),b.read(mode.ROOT/'FREEZE.json')['identity'],['platform'])
    for name,digest in pair.SEALS.items():seal(b,pair.ROOT.parent/name,digest)
    compare(pair.identity(),b.read(pair.ROOT/'FREEZE.json')['identity'],['parent_identity','platform'])
    seal(b,span.CAST,'36e26febe980cdf4575333f6d5e21f3cf5023eaaffdac39f430f84f640e960f9')
    seal(b,span.s.PREC,'68017b2b26126167461b708e56e938b4a0ce544c5751d3b408da496d4577c265')
    compare(span.identity(),b.read(span.ROOT/'FREEZE.json')['identity'],['parent_identity','parent_identity','platform'])
