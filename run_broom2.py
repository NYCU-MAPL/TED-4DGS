"""broom2 (hypernerf): two-stage pipeline = stage1 base (60k) + stage2 compression rate sweep (real codec).

Validated rate settings (hypernerf uses lmbda_mask = lmbda):
    lmbda = 0.016, 0.008, 0.004, 0.002, 0.001   (lmbda_mask = lmbda each)
lmbda_mask is applied by patching arguments/broom2.py per rate point.

Run inside the ted4dgs conda env, from the repo root:  python run_broom2.py
Edit DATA_ROOT / LOG_ROOT below for your paths.
"""
import os
import re
import sys

SCENE = "broom2"
DATA_ROOT = "./data/hypernerf"                 # <-- dataset root
LOG_ROOT  = "./output/release"                   # <-- output/checkpoint root

# rate points; hypernerf sets lmbda_mask = lmbda
LAMBDAS = [0.016, 0.008, 0.004, 0.002, 0.001]

DATA = f"{DATA_ROOT}/{SCENE}"
BASE = f"{LOG_ROOT}/{SCENE}/base"
CK   = f"{BASE}/chkpnt60000.pth"
BASE_CFG = f"arguments/{SCENE}.py"
COMMON = (f"--eval --lod 0 --voxel_size 0.01 --update_init_factor 16 "
          f"-s {DATA} --project_name release_{SCENE} --no_comet")


def run(args):
    print("RUN: python train.py", args)
    if os.system(f"python train.py {args}") != 0:
        sys.exit(f"[{SCENE}] training failed")


def make_cfg(lmbda_mask):
    """Write a temp scene config with lmbda_mask = lmbda_mask (hypernerf: lmbda_mask = lmbda)."""
    cfg = f"arguments/{SCENE}_mask{lmbda_mask}.py"
    src = open(BASE_CFG).read()
    src = re.sub(r"lmbda_mask = [0-9.]*,", f"lmbda_mask = {lmbda_mask},", src)
    open(cfg, "w").write(src)
    assert f"lmbda_mask = {lmbda_mask}," in src, "config patch failed"
    return cfg


# ---- Stage 1: base 0->60000 with lmbda 0 (skip if checkpoint already exists) ----
if os.path.exists(CK):
    print(f"[{SCENE}] base checkpoint exists ({CK}), skipping stage 1")
else:
    run(f"{COMMON} --config {BASE_CFG} --iterations 60000 "
        f"-m {BASE} --exp_name {SCENE}_base --lmbda 0.0")

# ---- Stage 2: compression rate sweep (lmbda_mask = lmbda; load base -> 100001, real codec at 100001) ----
for lmbda in LAMBDAS:
    cfg = make_cfg(lmbda)
    out = f"{LOG_ROOT}/{SCENE}/l{lmbda}"
    run(f"{COMMON} --config {cfg} --iterations 100001 -m {out} "
        f"--exp_name {SCENE}_l{lmbda} --lmbda {lmbda} "
        f"--load_iteration 60000 --start_checkpoint {CK} --ckpt_path {BASE}")
    os.remove(cfg)

print(f"[{SCENE}] done. Results under {LOG_ROOT}/{SCENE}/ and output/release_{SCENE}/")
