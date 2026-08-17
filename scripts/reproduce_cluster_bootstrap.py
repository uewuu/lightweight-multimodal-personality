from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
TRAITS=list("OCEAN")

def load(pred_path, labels_path, manifest_path):
    p=pd.read_csv(pred_path); y=pd.read_csv(labels_path); m=pd.read_csv(manifest_path)[["sample_id","source_video_id"]]
    z=p.merge(y,on="sample_id",validate="one_to_one").merge(m,on="sample_id",validate="one_to_one")
    if len(z)!=len(p): raise ValueError("Prediction/label/manifest ID join was incomplete")
    P=z[[f"pred_{t}" for t in TRAITS]].to_numpy(float); Y=z[[f"target_{t}" for t in TRAITS]].to_numpy(float)
    return z["sample_id"].to_numpy(), z["source_video_id"].astype(str).to_numpy(), P, Y

def cluster_stats(inv,K,P,Y):
    arrays=[Y,Y*Y,P,P*P,Y*P,np.abs(P-Y)]
    return [np.stack([np.bincount(inv,weights=a[:,t],minlength=K) for t in range(5)],axis=1) for a in arrays]

def calc(n,vals):
    sy,sy2,sp,sp2,syp,ae=vals; n=n[:,None]
    mae=ae/n; racc=1-mae; sst=sy2-sy*sy/n; sse=sy2-2*syp+sp2; r2=1-sse/sst
    cov=syp-sy*sp/n; vy=sy2-sy*sy/n; vp=sp2-sp*sp/n; pcc=cov/np.sqrt(vy*vp)
    return np.stack([mae.mean(1),racc.mean(1),r2.mean(1),pcc.mean(1)],axis=1)

def main():
    ap=argparse.ArgumentParser(description="Paired source-video-cluster bootstrap for two ID-keyed prediction files.")
    ap.add_argument("--baseline",type=Path,required=True); ap.add_argument("--candidate",type=Path,required=True)
    ap.add_argument("--labels",type=Path,required=True); ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--iterations",type=int,default=10000); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--output",type=Path)
    a=ap.parse_args(); ids,cl,B,Y=load(a.baseline,a.labels,a.manifest); ids2,cl2,C,Y2=load(a.candidate,a.labels,a.manifest)
    if not np.array_equal(ids,ids2) or not np.array_equal(cl,cl2) or not np.allclose(Y,Y2): raise ValueError("Baseline and candidate did not align after ID joins")
    uniq,inv=np.unique(cl,return_inverse=True); K=len(uniq); ns=np.bincount(inv,minlength=K).astype(float); bs=cluster_stats(inv,K,B,Y); cs=cluster_stats(inv,K,C,Y)
    ones=np.ones((1,K),float); base=calc(ones@ns,[ones@x for x in bs])[0]; cand=calc(ones@ns,[ones@x for x in cs])[0]
    point=np.array([base[0]-cand[0],cand[1]-base[1],cand[2]-base[2],cand[3]-base[3]])
    rng=np.random.default_rng(a.seed); chunks=[]; step=250
    for start in range(0,a.iterations,step):
        b=min(step,a.iterations-start); sample=rng.choice(K,size=(b,K),replace=True); n=ns[sample].sum(1)
        mb=calc(n,[x[sample].sum(1) for x in bs]); mc=calc(n,[x[sample].sum(1) for x in cs])
        chunks.append(np.column_stack([mb[:,0]-mc[:,0],mc[:,1]-mb[:,1],mc[:,2]-mb[:,2],mc[:,3]-mb[:,3]]))
    boot=np.vstack(chunks); names=["mae_improvement_baseline_minus_candidate","racc_candidate_minus_baseline","r2_candidate_minus_baseline","pcc_candidate_minus_baseline"]
    result={"iterations":a.iterations,"seed":a.seed,"samples":len(ids),"source_video_clusters":K,"ci_method":"percentile 95%","metrics":{n:{"point":float(point[i]),"ci95":[float(x) for x in np.percentile(boot[:,i],[2.5,97.5])]} for i,n in enumerate(names)}}
    text=json.dumps(result,indent=2); print(text)
    if a.output: a.output.write_text(text+"\n",encoding="utf-8")
if __name__=="__main__": main()
