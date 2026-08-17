from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
TRAITS=list("OCEAN")

def main():
    ap=argparse.ArgumentParser(description="Fit five validation-only trait-wise affine calibration mappings.")
    ap.add_argument("--predictions",type=Path,required=True,help="Validation predictions: sample_id,pred_O..pred_N")
    ap.add_argument("--labels",type=Path,required=True,help="Local validation labels: sample_id,target_O..target_N")
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(); p=pd.read_csv(a.predictions); y=pd.read_csv(a.labels)
    m=p.merge(y,on="sample_id",how="inner",validate="one_to_one")
    if len(m)!=len(p): raise ValueError(f"Only {len(m)}/{len(p)} validation predictions matched labels")
    pars={}
    for t in TRAITS:
        x=m[f"pred_{t}"].to_numpy(float); target=m[f"target_{t}"].to_numpy(float)
        A=np.column_stack([x,np.ones_like(x)]); slope,intercept=np.linalg.lstsq(A,target,rcond=None)[0]
        pars[t]={"slope_a":float(slope),"intercept_b":float(intercept)}
    payload={"fit_split":"validation","fit_samples":len(m),"method":"ordinary least squares, independently per trait","formula":"clip(a_t * raw_prediction_t + b_t, 0, 1)","clip_range":[0.0,1.0],"parameters":pars,"test_labels_used_for_parameter_fitting":False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
