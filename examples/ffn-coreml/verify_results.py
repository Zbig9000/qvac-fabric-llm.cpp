"""Validate the committed 2026-10-29 benchmark evidence without loading a model."""

import json,math,statistics
from pathlib import Path
r=Path(__file__).resolve().parent/'results'/'2026-10-29'
assert json.loads((r/'complete.json').read_text())['passed']
qualified=json.loads((r/'qualified-configurations.json').read_text())
cache=json.loads((r/'cache-check.json').read_text())
assert cache['passed'] and cache['runs'][0]['quality']==cache['runs'][1]['quality']
quality=[]
for path in sorted((r/'combined').glob('*/qualified.json')):
 q=json.loads(path.read_text()); runs=json.loads((path.parent/'runs.json').read_text())
 for run in runs:
  assert run['exit_code']==0
  for x in run['quality']:
   assert all(math.isfinite(float(v)) for v in x.values())
   assert int(x['offload_calls'])==len(q['layers'])*int(x['prompt_tokens'])//512
   passed=x['screen_pass']=='1' and float(x['mean_kl'])<=.001 and math.exp(float(x['split_nll'])-float(x['metal_nll']))<=1.01
   if q['passed']: assert passed
 if q['passed']: assert q['cases']==12
 quality.append(dict(name=path.parent.name,**q))
labels=['metal-before']+[x['label'] for x in qualified]+['metal-after']
data={}; samples={}; tokens={}
for label in labels:
 record=json.loads((r/label/'run.json').read_text())
 assert record['exit_code']==0
 reps=3 if label.startswith('metal-') else 5
 n=0 if label.startswith('metal-') else int(label.removeprefix('coreml'))
 assert len(record['results'])==2*(reps+1)
 samples[label]={}; data[label]={}; tokens[label]={}
 for length in [2048,8192]:
  rows=[x for x in record['results'] if int(x['prompt_tokens'])==length]
  assert [int(x['repetition']) for x in rows]==[-1]+list(range(reps))
  for x in rows:
   assert int(x['offload_calls'])==n*length//512 and int(x['decode_offload_calls'])==0
   assert all(math.isfinite(float(v)) for k,v in x.items() if k!='mode')
  measured=rows[1:]; samples[label][length]=measured
  data[label][length]={key:statistics.median(float(x[key]) for x in measured) for key in ['pp_ms','decode_ms','total_ms','pp_tps','decode_tps']}
  tokens[label][length]=[]
  for i in range(reps):
   assert int(measured[i]['generated'])==128
   mode='metal' if n==0 else 'split'
   t=(r/label/'tokens'/f'{mode}-p{length}-r{i}.tokens').read_text().splitlines()
   assert len(t)==128
   tokens[label][length].append(t)
reference={}; drift={}; comparisons=[]; token_checks=[]
for length in [2048,8192]:
 rows=samples['metal-before'][length]+samples['metal-after'][length]
 reference[length]={key:statistics.median(float(x[key]) for x in rows) for key in ['pp_ms','decode_ms','total_ms','pp_tps','decode_tps']}
 drift[length]={key:(data['metal-after'][length][key]/data['metal-before'][length][key]-1)*100 for key in ['pp_ms','total_ms']}
 reference_tokens=tokens['metal-before'][length][0]
 assert all(t==reference_tokens for label in ['metal-before','metal-after'] for t in tokens[label][length]),'Metal continuation changed across runs'
 prompt=(r/'metal-before/tokens'/f'prompt-{length}.tokens').read_text()
 for label in labels:
  assert (r/label/'tokens'/f'prompt-{length}.tokens').read_text()==prompt
 for config in qualified:
  label=config['label']; med=data[label][length]; base=reference[length]; old=data['coreml20'][length]
  checks=[]
  for rep,t in enumerate(tokens[label][length]):
   checks.append(dict(repetition=rep,identical_to_metal=t==reference_tokens,first_difference_1based=next((i+1 for i,(a,b) in enumerate(zip(t,reference_tokens)) if a!=b),None),identical_to_coreml20=t==tokens['coreml20'][length][rep]))
  token_checks.append(dict(label=label,prompt_tokens=length,comparisons=checks))
  comparisons.append(dict(label=label,prompt_tokens=length,median=med,prefill_throughput_gain_vs_metal_percent=(base['pp_ms']/med['pp_ms']-1)*100,total_latency_reduction_vs_metal_percent=(1-med['total_ms']/base['total_ms'])*100,saved_vs_metal_ms=base['total_ms']-med['total_ms'],prefill_throughput_gain_vs20_percent=(old['pp_ms']/med['pp_ms']-1)*100,total_latency_reduction_vs20_percent=(1-med['total_ms']/old['total_ms'])*100,prefill_range_ms=[min(float(x['pp_ms']) for x in samples[label][length]),max(float(x['pp_ms']) for x in samples[label][length])]))
summary=dict(quality=quality,qualified=qualified,reference_medians=reference,reference_drift_percent=drift,comparisons=comparisons,token_checks=token_checks)
expected=json.loads((r/'summary.json').read_text())
assert json.loads(json.dumps(summary))==expected, 'recorded summary differs from raw measurements'
print('Verified 22/24-layer accuracy, all measured timings, reference drift, offload counts and 128-token comparisons.')
for x in comparisons:
    print(x['label'],x['prompt_tokens'],'prefill gain vs Metal %.3f%%'%x['prefill_throughput_gain_vs_metal_percent'],'request latency reduction %.3f%%'%x['total_latency_reduction_vs_metal_percent'])
