"""Render a 3D Gaussian splat (.ply from Idol) to images or a turntable video, on the CPU.

A small, dependency-light EWA splatting renderer (numba), so the results can be rendered on any machine without a
CUDA rasterizer. It follows the standard 3DGS recipe: 3D covariance from scale + rotation, projected to a 2D
covariance with the camera Jacobian (+0.3 px low-pass), sorted by depth, alpha-composited front to back.

  python render_turntable.py outputs/01-robot-toy.ply --out renders/01-robot-toy.mp4
  python render_turntable.py outputs/01-robot-toy.ply --still renders/robot-front.png --yaw 0
  python render_turntable.py outputs/01-robot-toy.ply --camera outputs/01-robot-toy_camera.json --still input-view.png

Idol's PLY uses the usual 3DGS convention: object inside [-0.5, 0.5]^3, "up" is -y.
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np
from numba import njit, prange
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def load_ply(path):
    v = PlyData.read(path)['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
    color = np.clip(0.5 + SH_C0 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], 1), 0, 1)
    opacity = 1 / (1 + np.exp(-np.asarray(v['opacity'], np.float64)))
    scale = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], 1).astype(np.float64))
    q = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
    M = R * scale[:, None, :]
    cov = M @ M.transpose(0, 2, 1)
    return xyz, color, opacity, cov


def look_at(eye, target=(0, 0, 0), up=(0, -1, 0)):
    eye, target, up = map(lambda a: np.asarray(a, np.float64), (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)                 # camera +z (forward)
    r = np.cross(f, up); r /= np.linalg.norm(r)             # camera +x (right)
    d = np.cross(f, r)                                      # camera +y (down)
    Rw = np.stack([r, d, f])                                # world -> camera rotation
    return Rw, -Rw @ eye


@njit(parallel=True, fastmath=True, cache=True)
def _raster(px, py, depth, a, b, c, op, col, order, W, H, bands, bg):
    img = np.zeros((H, W, 3))
    T = np.ones((H, W))
    rows = (H + bands - 1) // bands
    for band in prange(bands):
        y0 = band * rows
        y1 = min(H, y0 + rows)
        for k in range(order.shape[0]):
            i = order[k]
            # inverse 2D covariance [[a, b], [b, c]]
            det = a[i] * c[i] - b[i] * b[i]
            if det <= 0:
                continue
            ia, ib, ic = c[i] / det, -b[i] / det, a[i] / det
            mid = 0.5 * (a[i] + c[i])
            lam = mid + math.sqrt(max(0.1, mid * mid - det))
            rad = math.ceil(3.0 * math.sqrt(lam))
            ymin = max(y0, int(py[i] - rad)); ymax = min(y1 - 1, int(py[i] + rad))
            if ymin > ymax:
                continue
            xmin = max(0, int(px[i] - rad)); xmax = min(W - 1, int(px[i] + rad))
            for yy in range(ymin, ymax + 1):
                dy = yy + 0.5 - py[i]
                for xx in range(xmin, xmax + 1):
                    t = T[yy, xx]
                    if t < 1e-4:
                        continue
                    dx = xx + 0.5 - px[i]
                    power = -0.5 * (ia * dx * dx + 2 * ib * dx * dy + ic * dy * dy)
                    if power > 0:
                        continue
                    alpha = min(0.99, op[i] * math.exp(power))
                    if alpha < 1.0 / 255.0:
                        continue
                    w = alpha * t
                    img[yy, xx, 0] += w * col[i, 0]
                    img[yy, xx, 1] += w * col[i, 1]
                    img[yy, xx, 2] += w * col[i, 2]
                    T[yy, xx] = t * (1 - alpha)
    for yy in prange(H):
        for xx in range(W):
            for ch in range(3):
                img[yy, xx, ch] += T[yy, xx] * bg[ch]
    return img, 1.0 - T


def render(scene, Rw, tw, fov_deg, W, H, bg=(0, 0, 0), bands=8):
    xyz, color, opacity, cov = scene
    cam = xyz @ Rw.T + tw
    z = cam[:, 2]
    keep = z > 0.05
    cam, col, op, cv = cam[keep], color[keep], opacity[keep], cov[keep]
    z = cam[:, 2]
    f = 0.5 * H / math.tan(math.radians(fov_deg) / 2)
    px = f * cam[:, 0] / z + W / 2
    py = f * cam[:, 1] / z + H / 2
    # Jacobian of the perspective projection, then the 2D covariance J W cov W^T J^T (+0.3 px low-pass)
    J = np.zeros((len(z), 2, 3))
    J[:, 0, 0] = f / z; J[:, 0, 2] = -f * cam[:, 0] / z ** 2
    J[:, 1, 1] = f / z; J[:, 1, 2] = -f * cam[:, 1] / z ** 2
    T = J @ Rw
    c2 = T @ cv @ T.transpose(0, 2, 1)
    a = c2[:, 0, 0] + 0.3; b = c2[:, 0, 1]; c = c2[:, 1, 1] + 0.3
    order = np.argsort(z).astype(np.int64)
    img, alpha = _raster(px, py, z, a, b, c, op, col, order, W, H, bands, np.asarray(bg, np.float64))
    return np.clip(img, 0, 1), alpha


def orbit_eye(yaw, pitch, radius):
    """yaw/pitch in radians around the object's up axis (-y); yaw 0 looks at the object's front (+z side)."""
    return np.array([radius * math.cos(pitch) * math.sin(yaw), -radius * math.sin(pitch),
                     radius * math.cos(pitch) * math.cos(yaw)])


def camera_from_idol(path):
    """Idol's predicted input camera (z-up frame: yaw = atan2(dx, dy), pitch = asin(dz)) in the PLY's frame."""
    cam = json.load(open(path))
    yaw, pitch, radius = cam['yaw'], cam['pitch'], min(cam['radius'], 50.0)
    d = np.array([math.cos(pitch) * math.sin(yaw), math.cos(pitch) * math.cos(yaw), math.sin(pitch)])
    eye = radius * np.array([d[0], -d[2], d[1]])            # (x, y, z) z-up -> (x, -z, y), as in save_ply
    fov = cam['fov'] if cam['radius'] < 1000 else math.degrees(2 * math.atan(0.5 * 1.2 / radius))
    return eye, fov


def hex_rgb(h):
    h = h.lstrip('#'); return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ply')
    p.add_argument('--out', help='turntable video (.mp4)')
    p.add_argument('--still', help='a single image (.png)')
    p.add_argument('--camera', help="Idol's <name>_camera.json: render from the predicted input view")
    p.add_argument('--yaw', type=float, default=0.0, help='degrees, for --still')
    p.add_argument('--pitch', type=float, default=15.0, help='degrees')
    p.add_argument('--radius', type=float, default=2.2)
    p.add_argument('--fov', type=float, default=30.0)
    p.add_argument('--size', type=int, default=720)
    p.add_argument('--seconds', type=float, default=8.0)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--bg', default='#000000')
    p.add_argument('--alpha', action='store_true', help='--still: write RGBA (transparent background)')
    p.add_argument('--yaw0', type=float, default=0.0, help='turntable start angle, degrees')
    p.add_argument('--turns', type=float, default=1.0, help='turntable: number of turns over --seconds')
    p.add_argument('--shrink', type=float, default=1.0, help='scale every Gaussian (e.g. 0.3 shows the individual blobs)')
    p.add_argument('--shrink_to', type=float, default=None, help='turntable: animate the Gaussian scale from --shrink to this')
    p.add_argument('--target', type=float, nargs=3, default=(0, 0, 0), help='look-at point (PLY frame)')
    args = p.parse_args()

    base = load_ply(args.ply)
    scene = base
    if args.shrink != 1.0:
        scene = (base[0], base[1], base[2], base[3] * args.shrink ** 2)
    bg = hex_rgb(args.bg)
    from PIL import Image
    if args.still:
        if args.camera:
            eye, fov = camera_from_idol(args.camera)
        else:
            eye, fov = orbit_eye(math.radians(args.yaw), math.radians(args.pitch), args.radius), args.fov
        Rw, tw = look_at(eye, target=args.target)
        img, alpha = render(scene, Rw, tw, fov, args.size, args.size, bg)
        if args.alpha:
            rgba = np.dstack([img - (1 - alpha)[..., None] * np.asarray(bg), alpha])
            rgba[..., :3] = np.where(alpha[..., None] > 1e-4, rgba[..., :3] / np.maximum(alpha[..., None], 1e-4), 0)
            Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), 'RGBA').save(args.still)
        else:
            Image.fromarray((img * 255).astype(np.uint8)).save(args.still)
        print(args.still)
        return
    out = args.out or os.path.splitext(args.ply)[0] + '_turntable.mp4'
    frames = int(args.seconds * args.fps)
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{args.size}x{args.size}', '-r', str(args.fps), '-i', '-', '-c:v', 'libx264',
                           '-preset', 'veryfast', '-crf', '16', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for n in range(frames):
        yaw = math.radians(args.yaw0) + 2 * math.pi * args.turns * n / frames
        Rw, tw = look_at(orbit_eye(yaw, math.radians(args.pitch), args.radius) + np.asarray(args.target), target=args.target)
        if args.shrink_to is not None:
            u = n / max(1, frames - 1); u = u * u * (3 - 2 * u)
            k = args.shrink + (args.shrink_to - args.shrink) * u
            scene = (base[0], base[1], base[2], base[3] * k * k)
        img, _ = render(scene, Rw, tw, args.fov, args.size, args.size, bg)
        ff.stdin.write((img * 255).astype(np.uint8).tobytes())
    ff.stdin.close(); ff.wait()
    print(out)


if __name__ == '__main__':
    main()
