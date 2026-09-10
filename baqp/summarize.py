"""Paired, final-round test accuracy; never chooses a winning seed/checkpoint."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import t

def main():
    p=argparse.ArgumentParser(); p.add_argument('root'); args=p.parse_args()
    root=Path(args.root); groups=defaultdict(dict)
    for file in root.rglob('result.json'):
        r=json.loads(file.read_text()); key=(r['dataset'],r['model'],r['alpha'],r['budget_fraction'])
        groups[key].setdefault(r['method'],{})[r['seed']]=r['test']['acc']
    rows=[]; paired=[]
    for key,methods in sorted(groups.items()):
        for method,seeds in sorted(methods.items()):
            v=np.array(list(seeds.values()))*100
            rows.append(dict(dataset=key[0],model=key[1],alpha=key[2],budget_fraction=key[3],method=method,
                             seeds=len(v),mean_test_acc_pct=v.mean(),std_test_acc_pp=v.std(ddof=1) if len(v)>1 else None))
        for method,seeds in sorted(methods.items()):
            if method=='continuous' or 'continuous' not in methods: continue
            common=sorted(set(seeds)&set(methods['continuous']))
            delta=np.array([methods['continuous'][s]-seeds[s] for s in common])*100
            if not len(delta): continue
            width=float(t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None
            paired.append(dict(dataset=key[0],model=key[1],alpha=key[2],budget_fraction=key[3],baseline=method,
                               paired_seeds=common,mean_delta_pp=float(delta.mean()),
                               ci95_low_pp=float(delta.mean()-width) if width is not None else None,
                               ci95_high_pp=float(delta.mean()+width) if width is not None else None))
    for name,items in [('summary',rows),('paired',paired)]:
        (root/(name+'.json')).write_text(json.dumps(items,indent=2))
        if items:
            with (root/(name+'.csv')).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=items[0]); writer.writeheader(); writer.writerows(items)
    print(json.dumps(dict(summary=rows,paired=paired),indent=2))

if __name__=='__main__': main()
