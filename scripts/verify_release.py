from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, re
import pandas as pd, numpy as np, yaml

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); a=ap.parse_args(); root=a.root
    failures=[]
    sums=root/"SHA256SUMS.txt"
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        digest,rel=line.split("  ",1); p=root/rel
        if not p.is_file() or sha(p)!=digest: failures.append(f"hash mismatch: {rel}")
    for p in (root/"artifacts/predictions").glob("*.csv"):
        df=pd.read_csv(p)
        forbidden=[c for c in df.columns if c.lower().startswith("target") or "groundtruth" in c.lower() or c.lower()=="label"]
        if forbidden: failures.append(f"label columns in {p.name}: {forbidden}")
    z=np.load(root/"artifacts/teacher_predictions/full_teacher_train_predictions_seed42.npz",allow_pickle=False)
    if "targets" in z.files: failures.append("teacher NPZ still contains targets")
    ckpts=list(root.rglob("*.ckpt"))
    if ckpts: failures.append(f"checkpoint binaries present: {[str(p.relative_to(root)) for p in ckpts]}")
    try:
        metadata=json.loads((root/"artifacts/provenance/checkpoint_metadata.json").read_text(encoding="utf-8"))
        if any(v.get("distributed_in_public_release") is not False for v in metadata.values()):
            failures.append("checkpoint metadata does not mark all binaries as non-distributed")
    except Exception as exc:
        failures.append(f"checkpoint metadata check failed: {exc}")
    try:
        import sys
        sys.path.insert(0, str(root))
        from model import FinalStudent, count_trainable_parameters
        with (root/"config.yaml").open("r",encoding="utf-8-sig") as f: cfg=yaml.safe_load(f) or {}
        with (root/"configs/generic_joint_seed42.yaml").open("r",encoding="utf-8-sig") as f: gcfg=yaml.safe_load(f) or {}
        f_count=count_trainable_parameters(FinalStudent(cfg)); g_count=count_trainable_parameters(FinalStudent(gcfg))
        if f_count!=500107: failures.append(f"multiplicative parameter count mismatch: {f_count}")
        if g_count!=500875: failures.append(f"Generic-Joint parameter count mismatch: {g_count}")
    except Exception as exc:
        failures.append(f"model smoke check failed: {exc}")
    print(json.dumps({"status":"PASS" if not failures else "FAIL","failures":failures},indent=2))
    raise SystemExit(1 if failures else 0)
if __name__=="__main__": main()
