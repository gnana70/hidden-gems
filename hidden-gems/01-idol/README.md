# Hidden Gems #1: Idol (one image → 3D Gaussian splat)

[Idol](https://huggingface.co/speridlabs/idol) by Speridlabs (Apache 2.0) turns a single picture of an object into a
3D Gaussian splat (`.ply`), including the sides the picture never showed. This folder is the test we ran for the video:
the scripts, the 9 test images and our raw results.

**Test machine:** gaming laptop, RTX 3070 Ti Laptop GPU (8 GB), Intel Core i7-12650H, Windows 11 + WSL2 (Ubuntu 22.04),
Python 3.12, PyTorch 2.15.0 (CUDA 13.2 build), Idol SDK commit `4702a7f` (see `idol-sdk-commit.txt`). 6–7 Nov 2026.

## What we measured
| | Result |
|---|---|
| Fits in 8 GB? | Yes: peak 4.95 GB in PyTorch, 5.38 GB for the whole GPU (NVML, every 20 ms). The model card says 7.2 GB. |
| Time per object (defaults: 50 steps, 262,144 Gaussians) | 98.7 s median of 9 images (95.7–99.5 s). 97% of it is the 50 sampling steps. |
| Fewer steps | 25 steps: 49.4 s, 10 steps: 20.8 s (robot toy, same seed). Hard to tell apart. |
| More Gaussians | 1,048,576 instead of 262,144: +3 s decoding, same memory, a 4× bigger file (71.3 MB). |
| torch.compile | With default settings our WSL session quit as soon as compiling started (`results/20261106-235327`). With `TORCHINDUCTOR_COMPILE_THREADS=1` it worked: 78 s per object (first image 97.6 s while it compiles). |

Wins: retro robot toy, sneakers from a phone-style photo, "HIDDEN GEMS" soda can, bicycle (except the spokes),
even the 4-object desk. Misses: the glass bottle turns milky, the film camera is far too deep from the side, the black
laptop's wallpaper shows on the back of the lid, and the red panda's back depends on the seed (seed 7 good, 42 and 123 not).

## Run it (Windows + WSL2, PowerShell)
1. With your Hugging Face account, accept the terms of the two gated image encoders:
   [DINOv3 ViT-H+](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) (approved by hand by Meta; ours
   took under an hour) and [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) (instant; only its VAE is used).
2. In this folder, copy `.env.example` to `.env` and put a Hugging Face **Read** token in it. `.env` is git-ignored.
3. From this folder:
   ```powershell
   .\Run-Idol-Tests.ps1
   ```
   First run: installs Python 3.12, PyTorch and the Idol SDK into `~/.venvs/idol` inside WSL, downloads ~6.8 GB of weights
   (Idol 3.1 GB, DINOv3 3.4 GB, FLUX.2 VAE 0.3 GB), then runs about 20 generations (30–45 min on our laptop).
   Results land in `results/<date-time>/` and `results/<date-time>-compiled/`.

Close games and other GPU-heavy apps first.

## Run one thing (Ubuntu / WSL tab, from this folder)
```bash
bash run_wsl.sh idol_test.py                                   # the short version shown in the video
bash run_wsl.sh bench.py --no-optimize --only 03-perfume-glass --no-sweep
bash run_wsl.sh render_turntable.py results/<run>/outputs/01-robot-toy.ply --out robot.mp4
```
Plain Python with the SDK:
```python
from idol import IdolPipeline
pipe = IdolPipeline.from_pretrained("speridlabs/idol")
out = pipe("images/01-robot-toy.png", mode="synthetic", seed=42)   # mode="real" for photos
out.gaussian.save_ply("robot.ply")
```

## Files
| File | What |
|---|---|
| `Run-Idol-Tests.ps1` → `run_all.sh` | everything in order: setup (first time), downloads, the full test, the torch.compile comparison |
| `setup_wsl.sh` | one-time environment setup inside WSL (uv, PyTorch with CUDA, the Idol SDK, numba, nvidia-ml-py) |
| `download_weights.py` | checks your token and the gated access, then downloads the weights |
| `bench.py` | mirrors `IdolPipeline.__call__` stage by stage and only adds timing: time per stage, PyTorch peak memory, whole-GPU peak (NVML). Runs the 9 images, a Gaussians/steps sweep and a seed check |
| `idol_test.py` | the minimal loop from the video |
| `render_turntable.py` | renders a `.ply` to a turntable video or a still on the CPU (numba EWA splatting, no CUDA rasterizer needed) |
| `images/` | the 9 test images, made with Google Gemini (see the table below) |
| `results/20261107-000033/` | main run, no torch.compile: `bench.json` (every number), `console.log`, `system.txt`, `outputs/` (what the model saw: `_input.png`, its predicted camera: `_camera.json`), `renders/` (our stills of every result) |
| `results/20261107-000033-compiled/` | torch.compile with one compiler thread (robot and panda) |
| `results/20261106-235327/` | the first attempt, where torch.compile with default settings made WSL quit |

The `.ply` files (18–71 MB each) are not in the repository: run the test to make your own.

## Test images
| # | Image | Why |
|---|---|---|
| 01 | retro robot toy | easy case: one object, plain background |
| 02 | red panda figurine | character, hidden back (tail, hood) |
| 03 | glass perfume bottle | transparency and refraction |
| 04 | road bicycle, side view | thin structures (spokes, frame) |
| 05 | sneakers, phone-style photo | real-photo mode, cluttered background |
| 06 | film camera, front view | the back is never shown |
| 07 | "HIDDEN GEMS" soda can | text on a curved surface |
| 08 | black gaming laptop | dark object on Idol's black background |
| 09 | desk scene (lamp, plant, mug, books) | several objects, which Idol is not built for |

## Licences
Our scripts: MIT. Idol and its SDK: Apache 2.0. FLUX.2 VAE: Apache 2.0. DINOv3: Meta's own licence (check it before
commercial use). The benchmark numbers in the video that are not ours come from Speridlabs' model card (self-reported).
