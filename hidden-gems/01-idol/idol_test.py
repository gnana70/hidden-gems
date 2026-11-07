"""Hidden Gems #1 - Idol: one picture in, a 3D Gaussian splat out.

The short version of our test, as shown in the video. It runs every image in images/ through Idol with the
default settings and prints the time and the GPU memory peak of each one.

  python idol_test.py
"""
import time
from pathlib import Path

import torch
from idol import IdolPipeline

out = Path("outputs"); out.mkdir(exist_ok=True)

# Weights (~3 GB) plus the two gated image encoders download on the first run.
pipe = IdolPipeline.from_pretrained("speridlabs/idol")

for image in sorted(Path("images").glob("*.png")):
    mode = "real" if "photo" in image.stem else "synthetic"   # photos get a looser crop
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    result = pipe(str(image), mode=mode, seed=42, steps=50, num_gaussians=262_144)

    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    result.gaussian.save_ply(str(out / f"{image.stem}.ply"))           # standard 3DGS file
    result.preprocessed_image.save(out / f"{image.stem}_input.png")    # what the model actually saw
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{image.stem:18s} {seconds:6.1f} s   peak {peak:.2f} GB")
