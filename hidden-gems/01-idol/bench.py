"""Hidden Gems #1: the full Idol test on one GPU (the RTX 3070 Ti Laptop, 8 GB, in our case).

What it measures, for every run:
  * wall time per stage: background removal + crop, image encoders (DINOv3 + FLUX.2 VAE), 50-step sampling, decoding
  * GPU memory: PyTorch's peak allocated / reserved, and the whole-device peak from NVML (sampled every 20 ms)
It mirrors IdolPipeline.__call__ step by step (same calls, same order, same seed), so the .ply files are the same
ones `pipeline(image)` writes; it only adds torch.cuda.synchronize() between stages to time them.

  python bench.py                      # everything: 9 images, settings sweep, seed check
  python bench.py --only 01-robot-toy  # one image
  python bench.py --no-optimize --only 01-robot-toy --tag eager   # without torch.compile / batched guidance
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # less fragmentation on an 8 GB GPU
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'idol-sdk'))
from idol import IdolPipeline  # noqa: E402
from idol.sampler import sample_flow  # noqa: E402
from idol.camera import decode_camera  # noqa: E402
from tensordict import TensorDict  # noqa: E402

GB = 2 ** 30


class DeviceMemory:
    """Whole-GPU memory in use (all processes), sampled in a background thread."""
    def __init__(self):
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print(f'(NVML not available: {e}; device-wide memory will not be sampled)')
        self.peak = 0
        self._stop = threading.Event()

    def used(self):
        return self.nv.nvmlDeviceGetMemoryInfo(self.handle).used if self.ok else 0

    def total(self):
        return self.nv.nvmlDeviceGetMemoryInfo(self.handle).total if self.ok else torch.cuda.get_device_properties(0).total_memory

    def __enter__(self):
        self.peak = self.used()
        self._stop.clear()
        if self.ok:
            def loop():
                while not self._stop.is_set():
                    self.peak = max(self.peak, self.used())
                    time.sleep(0.02)
            self._t = threading.Thread(target=loop, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.ok:
            self._t.join()
            self.peak = max(self.peak, self.used())


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def generate(pipe, image, mode, seed, steps, cfg, rescale_t, num_gaussians):
    """IdolPipeline.__call__, stage by stage, with timings."""
    t = {}
    s = time.perf_counter()
    prepared = pipe.preprocess(image, mode=mode)                       # BiRefNet alpha, crop, 1024 x 1024 on black
    pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(pipe.device)
    sync(); t['preprocess'] = time.perf_counter() - s

    s = time.perf_counter()
    torch.manual_seed(seed)
    cond = pipe.image_encoder(pixels)                                   # DINOv3 tokens + FLUX.2 VAE tokens
    sync(); t['encode'] = time.perf_counter() - s

    s = time.perf_counter()
    noise = TensorDict({
        'latent': torch.randn(1, pipe.denoiser.q_token_length, pipe.denoiser.in_channels, device=pipe.device),
        'camera': torch.randn(1, 1, pipe.denoiser.cam_channels, device=pipe.device),
    }, batch_size=1, device=pipe.device)
    with pipe._sampling_settings():
        sample = sample_flow(pipe.denoiser, noise, cond=cond, neg_cond=cond.apply(torch.zeros_like), steps=steps,
                             rescale_t=rescale_t, cfg_strength=cfg, verbose=True, batch_cfg=pipe.optimize)
    sync(); t['sample'] = time.perf_counter() - s

    s = time.perf_counter()
    gaussian = pipe.decode(sample['latent'], num_gaussians)              # octree anchors -> 32 Gaussians each
    sync(); t['decode'] = time.perf_counter() - s
    return gaussian, prepared, decode_camera(sample['camera'][0, 0]), t


def environment():
    env = {'date': datetime.now().isoformat(timespec='seconds'), 'python': platform.python_version(),
           'platform': platform.platform(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           'gpu_total_gb': round(torch.cuda.get_device_properties(0).total_memory / GB, 2) if torch.cuda.is_available() else None}
    try:
        env['nvidia_smi'] = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        pass
    try:
        env['cpu'] = subprocess.run(['bash', '-c', "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
                                    capture_output=True, text=True).stdout.strip()
        env['ram_gb'] = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / GB, 1)
    except Exception:  # noqa: BLE001
        pass
    return env


def plan(args):
    images = sorted((HERE / 'images').glob('*.png'))
    if args.only:
        images = [p for p in images if p.stem in args.only]
    runs = []
    first = True
    for img in images:
        mode = 'real' if 'photo' in img.stem else 'synthetic'
        if first:  # the first generation in a process pays the torch.compile cost: run it twice, keep both
            runs.append(dict(name=img.stem, image=img, mode=mode, label='cold (first run, compiles)' if args.optimize else 'cold (first run)', save=False))
            first = False
        runs.append(dict(name=img.stem, image=img, mode=mode, label='default'))
    if args.sweep and not args.only:
        robot = HERE / 'images' / '01-robot-toy.png'
        for g in (65_536, 524_288, 1_048_576):
            runs.append(dict(name=f'01-robot-toy_g{g}', image=robot, mode='synthetic', gaussians=g, label=f'{g:,} Gaussians'))
        for st in (10, 25):
            runs.append(dict(name=f'01-robot-toy_s{st}', image=robot, mode='synthetic', steps=st, label=f'{st} steps'))
        panda = HERE / 'images' / '02-red-panda.png'
        for sd in (7, 123):
            runs.append(dict(name=f'02-red-panda_seed{sd}', image=panda, mode='synthetic', seed=sd, label=f'seed {sd}'))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*', help='image names (without .png)')
    ap.add_argument('--no-optimize', dest='optimize', action='store_false')
    ap.add_argument('--no-sweep', dest='sweep', action='store_false')
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=None, help='results folder (default results/<date-time>)')
    args = ap.parse_args()

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out = Path(args.out) if args.out else HERE / 'results' / f'{stamp}{"-" + args.tag if args.tag else ""}'
    (out / 'outputs').mkdir(parents=True, exist_ok=True)
    env = environment()
    print(f"GPU: {env['gpu']} ({env['gpu_total_gb']} GB) | torch {env['torch']} CUDA {env['cuda']} | optimize={args.optimize}")
    mem = DeviceMemory()
    report = {'env': env, 'optimize': args.optimize, 'runs': []}

    def save_report():
        (out / 'bench.json').write_text(json.dumps(report, indent=2, default=str))

    idle = mem.used()
    s = time.perf_counter()
    with mem:
        try:
            pipe = IdolPipeline.from_pretrained('speridlabs/idol', optimize=args.optimize)
        except Exception as e:  # noqa: BLE001
            report['load_error'] = repr(e); save_report(); raise
    sync()
    report['load'] = {'seconds': round(time.perf_counter() - s, 2),
                      'device_used_before_gb': round(idle / GB, 2),
                      'device_used_after_load_gb': round(mem.used() / GB, 2),
                      'torch_allocated_after_load_gb': round(torch.cuda.memory_allocated() / GB, 2)}
    print(f"Loaded in {report['load']['seconds']} s; weights on GPU: {report['load']['torch_allocated_after_load_gb']} GB "
          f"(device in use: {report['load']['device_used_after_load_gb']} GB of {round(mem.total() / GB, 2)} GB)")
    save_report()

    for r in plan(args):
        steps, g, seed = r.get('steps', 50), r.get('gaussians', 262_144), r.get('seed', 42)
        print(f"\n== {r['name']}  [{r['label']}]  mode={r['mode']} steps={steps} gaussians={g:,} seed={seed}", flush=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        row = dict(name=r['name'], label=r['label'], mode=r['mode'], steps=steps, gaussians=g, seed=seed)
        try:
            with mem:
                s = time.perf_counter()
                gaussian, prepared, camera, t = generate(pipe, str(r['image']), r['mode'], seed, steps, 3.0, 3.0, g)
                row['total_s'] = round(time.perf_counter() - s, 2)
            row.update({f'{k}_s': round(v, 2) for k, v in t.items()})
            row['torch_peak_allocated_gb'] = round(torch.cuda.max_memory_allocated() / GB, 2)
            row['torch_peak_reserved_gb'] = round(torch.cuda.max_memory_reserved() / GB, 2)
            row['device_peak_gb'] = round(mem.peak / GB, 2)
            row['camera'] = camera
            row['num_gaussians_out'] = int(gaussian.xyz.shape[0])
            if r.get('save', True):
                gaussian.save_ply(str(out / 'outputs' / f"{r['name']}.ply"))
                prepared.save(out / 'outputs' / f"{r['name']}_input.png")
                (out / 'outputs' / f"{r['name']}_camera.json").write_text(json.dumps(camera, indent=2))
            print(f"   {row['total_s']:6.1f} s total = preprocess {t['preprocess']:.1f} + encode {t['encode']:.1f} + "
                  f"sample {t['sample']:.1f} + decode {t['decode']:.1f} | peak {row['torch_peak_allocated_gb']} GB "
                  f"(PyTorch), {row['device_peak_gb']} GB (whole GPU)", flush=True)
            del gaussian
        except torch.cuda.OutOfMemoryError as e:
            row['error'] = 'CUDA out of memory: ' + str(e).splitlines()[0]
            print('   OUT OF MEMORY:', row['error'], flush=True)
        except Exception as e:  # noqa: BLE001
            row['error'] = repr(e)
            traceback.print_exc()
            if r['label'].startswith('cold') and args.optimize:
                report['runs'].append(row); save_report()
                raise SystemExit('The first (compiled) run failed. The --no-optimize results are the main ones.')
        report['runs'].append(row)
        save_report()

    warm = [x for x in report['runs'] if x['label'] == 'default' and 'total_s' in x]
    if warm:
        times = sorted(x['total_s'] for x in warm)
        report['summary'] = {'images': len(warm), 'median_s': times[len(times) // 2], 'min_s': times[0], 'max_s': times[-1],
                             'max_torch_peak_gb': max(x['torch_peak_allocated_gb'] for x in warm),
                             'max_device_peak_gb': max(x['device_peak_gb'] for x in warm)}
        print('\nSummary:', json.dumps(report['summary']))
    save_report()
    print(f'\nResults: {out}')


if __name__ == '__main__':
    main()
