from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TRAITS = list("OCEAN")


def load_joined(pred_path: Path, label_path: Path):
    p = pd.read_csv(pred_path)
    y = pd.read_csv(label_path)
    req_p = ["sample_id"] + [f"pred_{t}" for t in TRAITS]
    req_y = ["sample_id"] + [f"target_{t}" for t in TRAITS]
    missing = [c for c in req_p if c not in p.columns] + [c for c in req_y if c not in y.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    if p["sample_id"].duplicated().any() or y["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique in both files")
    m = p[req_p].merge(y[req_y], on="sample_id", how="inner", validate="one_to_one")
    if len(m) != len(p):
        raise ValueError(f"Only {len(m)}/{len(p)} predictions matched labels")
    pred = m[[f"pred_{t}" for t in TRAITS]].to_numpy(float)
    target = m[[f"target_{t}" for t in TRAITS]].to_numpy(float)
    return m, pred, target


def evaluate(pred, target):
    ae = np.abs(pred-target); mae = ae.mean(0); racc = 1-mae
    yc = target-target.mean(0, keepdims=True); pc = pred-pred.mean(0, keepdims=True)
    sst = np.sum(yc*yc,0); sse = np.sum((target-pred)**2,0); r2 = 1-sse/sst
    denom = np.sqrt(np.sum(yc*yc,0)*np.sum(pc*pc,0)); pcc = np.sum(yc*pc,0)/denom
    yt, pt = target.ravel(), pred.ravel(); pooled_r2 = 1-np.sum((yt-pt)**2)/np.sum((yt-yt.mean())**2)
    return {
        "trait_metrics": {t:{"mae":float(mae[i]),"racc":float(racc[i]),"r2":float(r2[i]),"pcc":float(pcc[i])} for i,t in enumerate(TRAITS)},
        "average": {"mae":float(mae.mean()),"racc":float(racc.mean()),"r2":float(r2.mean()),"pcc":float(pcc.mean()),"pooled_r2":float(pooled_r2)}
    }


def main():
    ap=argparse.ArgumentParser(description="Evaluate ID-keyed OCEAN predictions against locally supplied FI/FIv2 labels.")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    a=ap.parse_args(); _, pred, target = load_joined(a.predictions, a.labels); result=evaluate(pred,target)
    text=json.dumps(result,indent=2); print(text)
    if a.output: a.output.write_text(text+"\n",encoding="utf-8")
if __name__=="__main__": main()
