"""flame_steak (N3DV): two-stage pipeline = stage1 base (60k) + stage2 compression rate sweep (real codec).

Validated rate settings (lmbda_mask = lmbda * 0.8):
    lmbda      lmbda_mask
    0.002      0.0016
    0.004      0.0032
    0.008      0.0064
    0.016      0.0128

Run inside the ted4dgs conda env, from the repo root:  python run_flame_steak.py
Edit DATA_ROOT / LOG_ROOT below for your paths.
"""
import os
import sys

SCENE = "flame_steak"
DATA_ROOT = "./data/N3DV"                   # <-- dataset root
LOG_ROOT  = "./output/release"                 # <-- output/checkpoint root

# (lmbda, lmbda_mask) rate points; lmbda_mask = lmbda * 0.8
RATES = [(0.002, 0.0016), (0.004, 0.0032), (0.008, 0.0064), (0.016, 0.0128)]

DATA = f"{DATA_ROOT}/{SCENE}"
BASE = f"{LOG_ROOT}/{SCENE}/base"
CK   = f"{BASE}/chkpnt60000.pth"
COMMON = (f"--eval --lod 0 --voxel_size 0.0333 --update_init_factor 128 --cam_mask 1 "
          f"-s {DATA} --project_name release_{SCENE} --no_comet")


def run(args):
    print("RUN: python train.py", args)
    if os.system(f"python train.py {args}") != 0:
        sys.exit(f"[{SCENE}] training failed")


# ---- Stage 1: base 0->60000 (skip if checkpoint already exists) ----
if os.path.exists(CK):
    print(f"[{SCENE}] base checkpoint exists ({CK}), skipping stage 1")
else:
    run(f"{COMMON} --iterations 60000 -m {BASE} --exp_name {SCENE}_base "
        f"--lmbda 0.008 --lmbda_mask 0.0064")

# ---- Stage 2: compression rate sweep (load base -> 100000, real codec at 100000) ----
for lmbda, lmbda_mask in RATES:
    out = f"{LOG_ROOT}/{SCENE}/l{lmbda}"
    run(f"{COMMON} --iterations 100000 -m {out} "
        f"--exp_name {SCENE}_l{lmbda} --lmbda {lmbda} --lmbda_mask {lmbda_mask} "
        f"--load_iteration 60000 --start_checkpoint {CK} --ckpt_path {BASE}")

print(f"[{SCENE}] done. Results under {LOG_ROOT}/{SCENE}/ and output/release_{SCENE}/")
