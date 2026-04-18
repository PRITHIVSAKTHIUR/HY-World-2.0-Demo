"""
HY-World-2.0-Demo - WorldMirror 2.0 World Reconstruction
Custom Ubuntu/Aubergine Theme with Gradio Server + Rerun 3D Viewer

Requires Gradio version: 6.12.0
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import io
import json
import shutil
import sys
import time
import uuid
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

import rerun as rr
try:
    import rerun.blueprint as rrb
except ImportError:
    rrb = None

from gradio import Server
from fastapi import Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Try to import spaces for GPU decorator
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    class _DummyGPU:
        @staticmethod
        def GPU(duration=120):
            def decorator(fn):
                return fn
            return decorator
    spaces = _DummyGPU()

# Global model references
_model  = None
_device = None


def get_model_and_device():
    global _model, _device

    import torch
    from safetensors.torch import load_file as load_safetensors
    from omegaconf import OmegaConf
    from huggingface_hub import snapshot_download
    from hyworldmirror.models.models.worldmirror import WorldMirror

    if _model is None:
        print("[Lazy Load] Loading HY-World-2.0 model...")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            print(f"[Lazy Load] CUDA: {torch.cuda.get_device_name(torch.cuda.current_device())}")

        model_path = "tencent/HY-World-2.0"
        subfolder  = "HY-WorldMirror-2.0"

        repo_root = snapshot_download(
            repo_id=model_path,
            allow_patterns=[f"{subfolder}/**"]
        )
        model_dir = Path(repo_root) / subfolder
        yaml_path = model_dir / "config.yaml"
        json_path = model_dir / "config.json"

        if yaml_path.exists():
            cfg = OmegaConf.load(yaml_path)
            if hasattr(cfg, "wrapper") and hasattr(cfg.wrapper, "model"):
                model_cfg = cfg.wrapper.model
            elif hasattr(cfg, "model"):
                model_cfg = cfg.model
            else:
                model_cfg = cfg
            model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
            model_cfg.pop("_target_", None)
        elif json_path.exists():
            with open(json_path) as f:
                model_cfg = json.load(f)
        else:
            raise FileNotFoundError(f"No config found in {model_dir}")

        _model = WorldMirror(**model_cfg).to(_device)

        safetensors_path = model_dir / "model.safetensors"
        if safetensors_path.exists():
            state   = load_safetensors(str(safetensors_path))
            current = _model.state_dict()
            matched = 0
            for key in current:
                if key in state and current[key].shape == state[key].shape:
                    current[key] = state[key]
                    matched += 1
            _model.load_state_dict(current, strict=True)
            print(f"[Lazy Load] Loaded {matched}/{len(current)} keys")
            del state

        _model.eval()
        for param in _model.parameters():
            param.requires_grad = False

        print("[Lazy Load] Model ready")

    return _model, _device


try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


app = Server()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD", "OPTIONS", "POST"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Type", "Accept-Ranges"],
)

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"

OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Serve example_gradio directory as static files
EXAMPLES_DIR = BASE_DIR / "example_gradio"
EXAMPLES_DIR.mkdir(exist_ok=True)
app.mount("/examples", StaticFiles(directory=str(EXAMPLES_DIR)), name="examples")

DEVICE_LABEL = "gpu-worker"

def _find_rerun_web_assets() -> Optional[Path]:
    """
    rerun-sdk ships a self-contained web viewer under one of several paths
    depending on the installed version.  We probe the known locations.
    """
    import rerun
    pkg_root = Path(rerun.__file__).parent

    candidates = [
        pkg_root / "static",
        pkg_root / "web_viewer",
        pkg_root / "_rerun_bindings" / "static",
        pkg_root / "rerun_bindings" / "static",
    ]
    # Also try installed data directories
    import importlib.resources as _ir
    try:
        candidates.append(Path(str(_ir.files("rerun"))) / "static")
    except Exception:
        pass

    for p in candidates:
        if p.is_dir() and any(p.glob("index.html")):
            return p

    # Broader search: any index.html under the rerun package
    for p in pkg_root.rglob("index.html"):
        return p.parent

    return None


RERUN_WEB_DIR = _find_rerun_web_assets()

if RERUN_WEB_DIR is not None:
    print(f"[Rerun] Serving web viewer from: {RERUN_WEB_DIR}")
    app.mount("/rerun-viewer", StaticFiles(directory=str(RERUN_WEB_DIR), html=True), name="rerun-viewer")
    RERUN_VIEWER_BASE = "/rerun-viewer"
else:
    print("[Rerun] WARNING: local web assets not found — falling back to app.rerun.io")
    RERUN_VIEWER_BASE = None


def process_uploaded_files(files: List[UploadFile], time_interval: float = 1.0):
    target_dir = UPLOAD_DIR / f"input_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []

    for upload in files:
        if not upload.filename:
            continue
        ext       = Path(upload.filename).suffix.lower()
        base_name = Path(upload.filename).stem
        content   = upload.file.read()

        if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}:
            temp_video = target_dir / f"temp{ext}"
            with open(temp_video, "wb") as f:
                f.write(content)
            cap      = cv2.VideoCapture(str(temp_video))
            fps      = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            interval = max(1, int(fps * time_interval))
            idx, saved = 0, 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                if idx % interval == 0:
                    dst = images_dir / f"{base_name}_{saved:06d}.png"
                    cv2.imwrite(str(dst), frame)
                    image_paths.append(str(dst))
                    saved += 1
            cap.release()
            if temp_video.exists():
                os.remove(temp_video)

        elif ext in {".heic", ".heif"}:
            img = Image.open(io.BytesIO(content))
            dst = images_dir / f"{base_name}.jpg"
            img.convert("RGB").save(dst, "JPEG", quality=95)
            image_paths.append(str(dst))
        else:
            dst = images_dir / upload.filename
            with open(dst, "wb") as f:
                f.write(content)
            image_paths.append(str(dst))

    return str(target_dir), sorted(image_paths)


def process_example_file(filepath: str, time_interval: float = 1.0):
    """Process a local example file (image or video) and return target_dir + image_paths."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Example file not found: {filepath}")

    target_dir = UPLOAD_DIR / f"input_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []

    ext       = path.suffix.lower()
    base_name = path.stem

    if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}:
        cap      = cv2.VideoCapture(str(path))
        fps      = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        interval = max(1, int(fps * time_interval))
        idx, saved = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx % interval == 0:
                dst = images_dir / f"{base_name}_{saved:06d}.png"
                cv2.imwrite(str(dst), frame)
                image_paths.append(str(dst))
                saved += 1
        cap.release()
    elif ext in {".heic", ".heif"}:
        img = Image.open(str(path))
        dst = images_dir / f"{base_name}.jpg"
        img.convert("RGB").save(dst, "JPEG", quality=95)
        image_paths.append(str(dst))
    else:
        dst = images_dir / path.name
        shutil.copy2(str(path), str(dst))
        image_paths.append(str(dst))

    return str(target_dir), sorted(image_paths)


def render_depth_colormap(depth_map, mask=None):
    import matplotlib.pyplot as plt
    d     = depth_map.copy()
    valid = (d > 0) & mask if mask is not None else (d > 0)
    if valid.sum() > 0:
        lo, hi = np.percentile(d[valid], 5), np.percentile(d[valid], 95)
        d[valid] = (d[valid] - lo) / (hi - lo + 1e-9)
    rgb = (plt.cm.turbo_r(d)[:, :, :3] * 255).astype(np.uint8)
    if mask is not None:
        rgb[~mask] = [255, 255, 255]
    return rgb


def render_normal_colormap(normal_map, mask=None):
    n = normal_map.copy()
    if mask is not None:
        n[~mask] = 0
    return ((n + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)

def _make_rec(app_id: str) -> "rr.RecordingStream":
    """Create a RecordingStream compatible with rerun >= 0.23."""
    run_id = str(uuid.uuid4())
    return rr.RecordingStream(application_id=app_id, recording_id=run_id)


def build_rerun_reconstruction_recording(
    output_id: str,
    output_subdir: Path,
    glb_path: Path,
    world_points: np.ndarray,
    images_np: np.ndarray,
    camera_poses: np.ndarray,
    camera_intrs: np.ndarray,
    filter_mask: list,
    normals: Optional[np.ndarray],
) -> str:
    rrd_path = str(output_subdir / "reconstruction.rrd")
    rec      = _make_rec("HY-World-2.0-Reconstruction")

    rec.log("world", rr.Clear(recursive=True), static=True)
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Axes
    try:
        rec.log("world/axes/x", rr.Arrows3D(vectors=[[0.5, 0, 0]], colors=[[220, 50, 50]]),  static=True)
        rec.log("world/axes/y", rr.Arrows3D(vectors=[[0, 0.5, 0]], colors=[[50, 200, 50]]),  static=True)
        rec.log("world/axes/z", rr.Arrows3D(vectors=[[0, 0, 0.5]], colors=[[50, 100, 220]]), static=True)
    except Exception:
        pass

    # GLB mesh asset
    if glb_path.exists():
        try:
            rec.log("world/scene_mesh", rr.Asset3D(path=str(glb_path)), static=True)
        except Exception as e:
            print(f"[Rerun] GLB log failed: {e}")

    S = images_np.shape[0] if images_np.ndim == 4 else 1

    for i in range(S):
        img_hwc = np.transpose(images_np[i], (1, 2, 0))          # (H,W,C)
        img_u8  = (np.clip(img_hwc, 0, 1) * 255).astype(np.uint8)
        H, W    = img_u8.shape[:2]
        mask    = filter_mask[i] if (filter_mask and i < len(filter_mask)) else None

        # Point cloud
        if world_points.ndim in (4, 5):
            pts = world_points[i]
        else:
            pts  = world_points
            mask = None

        if pts.ndim == 3:
            pts_f  = pts.reshape(-1, 3)
            col_f  = img_u8.reshape(-1, 3)
            if mask is not None:
                mf    = mask.reshape(-1)
                pts_f = pts_f[mf]
                col_f = col_f[mf]
        else:
            pts_f = pts
            col_f = None

        if pts_f.shape[0] > 0:
            try:
                rec.log(
                    f"world/point_cloud/view_{i:03d}",
                    rr.Points3D(positions=pts_f.astype(np.float32), colors=col_f, radii=0.003),
                    static=True,
                )
            except Exception as e:
                print(f"[Rerun] Points3D view {i} failed: {e}")

        # Camera
        if camera_poses is not None and i < len(camera_poses):
            c2w  = camera_poses[i]
            intr = camera_intrs[i] if camera_intrs is not None else None
            try:
                rec.log(
                    f"world/cameras/cam_{i:03d}",
                    rr.Transform3D(
                        translation=c2w[:3, 3].astype(np.float32),
                        mat3x3=c2w[:3, :3].astype(np.float32),
                    ),
                    static=True,
                )
                if intr is not None:
                    rec.log(
                        f"world/cameras/cam_{i:03d}/pinhole",
                        rr.Pinhole(
                            focal_length=[float(intr[0, 0]), float(intr[1, 1])],
                            principal_point=[float(intr[0, 2]), float(intr[1, 2])],
                            width=W, height=H,
                        ),
                        static=True,
                    )
                    rec.log(
                        f"world/cameras/cam_{i:03d}/pinhole/image",
                        rr.Image(img_u8),
                        static=True,
                    )
            except Exception as e:
                print(f"[Rerun] Camera {i} failed: {e}")

        # Normals
        if normals is not None and i < len(normals):
            try:
                rec.log(
                    f"world/normals/view_{i:03d}",
                    rr.Image(render_normal_colormap(normals[i], mask)),
                    static=True,
                )
            except Exception as e:
                print(f"[Rerun] Normal {i} failed: {e}")

    # Blueprint
    if rrb is not None:
        try:
            bp = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="/world", name="3D Scene"),
                    rrb.Vertical(
                        rrb.Spatial2DView(
                            origin="/world/cameras/cam_000/pinhole/image",
                            name="Camera View",
                        ),
                        rrb.Spatial2DView(
                            origin="/world/normals/view_000",
                            name="Normal Map",
                        ),
                    ),
                    column_shares=[3, 1],
                ),
                collapse_panels=True,
            )
            rec.send_blueprint(bp)
        except Exception as e:
            print(f"[Rerun] Blueprint failed (non-fatal): {e}")

    rec.save(rrd_path)
    print(f"[Rerun] Reconstruction saved -> {rrd_path}")
    return rrd_path


def build_rerun_gaussians_recording(
    output_id: str,
    output_subdir: Path,
    means: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
) -> str:
    rrd_path = str(output_subdir / "gaussians.rrd")
    rec      = _make_rec("HY-World-2.0-Gaussians")

    rec.log("world", rr.Clear(recursive=True), static=True)
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    colors_u8   = (np.clip(colors, 0, 1) * 255).astype(np.uint8) if colors.dtype != np.uint8 else colors
    alpha       = (np.clip(opacities, 0, 1) * 255).astype(np.uint8).reshape(-1, 1)
    colors_rgba = np.concatenate([colors_u8, alpha], axis=1)

    radii = np.clip(np.linalg.norm(scales, axis=1) * 0.5, 0.001, 0.1).astype(np.float32)

    MAX_PTS = 2_000_000
    N       = means.shape[0]
    if N > MAX_PTS:
        idx         = np.random.default_rng(42).choice(N, size=MAX_PTS, replace=False)
        means       = means[idx]
        colors_rgba = colors_rgba[idx]
        radii       = radii[idx]

    try:
        rec.log(
            "world/gaussian_splats",
            rr.Points3D(
                positions=means.astype(np.float32),
                colors=colors_rgba,
                radii=radii,
            ),
            static=True,
        )
    except Exception as e:
        print(f"[Rerun] Gaussians Points3D failed: {e}")

    try:
        rec.log("world/axes/x", rr.Arrows3D(vectors=[[0.5,0,0]], colors=[[220,50,50]]),  static=True)
        rec.log("world/axes/y", rr.Arrows3D(vectors=[[0,0.5,0]], colors=[[50,200,50]]),  static=True)
        rec.log("world/axes/z", rr.Arrows3D(vectors=[[0,0,0.5]], colors=[[50,100,220]]), static=True)
    except Exception:
        pass

    if rrb is not None:
        try:
            bp = rrb.Blueprint(
                rrb.Spatial3DView(origin="/world", name="Gaussian Splats"),
                collapse_panels=True,
            )
            rec.send_blueprint(bp)
        except Exception as e:
            print(f"[Rerun] Gaussians blueprint failed (non-fatal): {e}")

    rec.save(rrd_path)
    print(f"[Rerun] Gaussians saved -> {rrd_path}")
    return rrd_path

def make_rerun_iframe_url(rrd_absolute_url: str) -> str:
    """
    Return the URL to embed in an iframe that shows the Rerun viewer
    pre-loaded with the given .rrd file.

    Priority:
      1. Local web viewer served at /rerun-viewer  (best — no cross-origin issues)
      2. app.rerun.io fallback
    """
    encoded = rrd_absolute_url  # already absolute, will be encoded in JS

    if RERUN_VIEWER_BASE is not None:
        # The local static viewer reads ?url=<rrd_url>
        return f"{RERUN_VIEWER_BASE}/index.html?url={encoded}"
    else:
        return f"https://app.rerun.io/version/latest/index.html?url={encoded}"

@spaces.GPU(duration=120)
def run_full_reconstruction_pipeline(
    target_dir: str,
    frame_selector: str = "All",
    show_camera: bool = True,
    filter_sky_bg: bool = False,
    show_mesh: bool = True,
    filter_ambiguous: bool = True,
):
    import torch
    from hyworldmirror.utils.inference_utils import (
        prepare_images_to_tensor,
        compute_adaptive_target_size,
        compute_sky_mask,
        compute_filter_mask,
        _voxel_prune_gaussians,
        depth_to_world_coords_points,
    )
    from hyworldmirror.utils.visual_util import convert_predictions_to_glb_scene
    from hyworldmirror.utils.save_utils import save_camera_params, save_gs_ply

    image_folder = Path(target_dir) / "images"
    img_paths = sorted(
        glob(str(image_folder / "*.png"))  +
        glob(str(image_folder / "*.jpg"))  +
        glob(str(image_folder / "*.jpeg")) +
        glob(str(image_folder / "*.webp"))
    )
    if not img_paths:
        raise ValueError("No images found in the uploaded directory.")

    model, device = get_model_and_device()
    effective     = compute_adaptive_target_size(img_paths, max_target_size=952)

    with torch.no_grad():
        imgs  = prepare_images_to_tensor(img_paths, target_size=effective, resize_strategy="crop").to(device)
        views = {"img": imgs}
        B, S, C, H, W = imgs.shape

        use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        t0      = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            predictions = model(views=views, cond_flags=[0, 0, 0], is_inference=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0

        sky_mask = compute_sky_mask(
            img_paths, H, W, S, predictions=predictions,
            source="auto", model_threshold=0.45,
            processed_aspect_ratio=W / H,
        )
        filter_mask, gs_filter_mask = compute_filter_mask(
            predictions, imgs, img_paths, H, W, S,
            apply_confidence_mask=False,
            apply_edge_mask=True,
            apply_sky_mask=True,
            confidence_percentile=10.0,
            edge_normal_threshold=1.0,
            edge_depth_threshold=0.03,
            sky_mask=sky_mask,
            use_gs_depth=("gs_depth" in predictions),
        )

        output_id     = uuid.uuid4().hex[:8]
        output_subdir = OUTPUT_DIR / f"recon_{output_id}"
        output_subdir.mkdir(exist_ok=True, parents=True)

        safe_name = frame_selector.replace(".", "_").replace(":", "").replace(" ", "_")
        glb_path  = output_subdir / f"scene_{safe_name}.glb"

        # Build glb_predictions dict
        glb_preds = {}
        if "world_points" not in predictions:
            glb_preds["world_points"] = depth_to_world_coords_points(
                predictions["depth"][0, ..., 0],
                predictions["camera_poses"][0],
                predictions["camera_intrs"][0],
            )[0].cpu().float().numpy()
        else:
            pts = predictions["world_points"]
            if isinstance(pts, torch.Tensor): pts = pts.detach().cpu().numpy()
            if pts.ndim == 5: pts = pts[0]
            glb_preds["world_points"] = pts

        glb_preds["images"]      = imgs[0].detach().cpu().numpy()
        c2w = predictions["camera_poses"]
        if isinstance(c2w, torch.Tensor): c2w = c2w.detach().cpu().numpy()
        if c2w.ndim == 4: c2w = c2w[0]
        glb_preds["camera_poses"] = c2w
        glb_preds["final_mask"]   = np.array(filter_mask)
        glb_preds["sky_mask"]     = np.array(sky_mask)

        normals_np = None
        for key in ("normal", "normals"):
            if key in predictions and predictions[key] is not None:
                n = predictions[key]
                if isinstance(n, torch.Tensor): n = n.detach().cpu().numpy()
                if n.ndim == 5: n = n[0]
                glb_preds["normal"] = n
                normals_np = n
                break

        glb_scene = convert_predictions_to_glb_scene(
            glb_preds,
            filter_by_frames=frame_selector,
            show_camera=show_camera,
            mask_sky_bg=filter_sky_bg,
            as_mesh=show_mesh,
            mask_ambiguous=filter_ambiguous,
        )
        glb_scene.export(file_obj=str(glb_path))

        # Depth & Normal images
        depth_urls, normal_urls = [], []
        for i in range(S):
            depth = predictions["depth"][0, i].cpu().float().numpy().squeeze()
            mask  = filter_mask[i] if i < len(filter_mask) else None

            depth_p = output_subdir / f"depth_{i:03d}.png"
            Image.fromarray(render_depth_colormap(depth, mask)).save(depth_p)
            depth_urls.append(f"/download/{output_id}/depth_{i:03d}.png")

            normal = np.zeros((H, W, 3), dtype=np.float32)
            for k in ("normals", "normal"):
                if k in predictions and predictions[k] is not None:
                    normal = predictions[k][0, i].cpu().float().numpy()
                    break
            normal_p = output_subdir / f"normal_{i:03d}.png"
            Image.fromarray(render_normal_colormap(normal, mask)).save(normal_p)
            normal_urls.append(f"/download/{output_id}/normal_{i:03d}.png")

        save_camera_params(
            predictions["camera_poses"][0].cpu().float().numpy(),
            predictions["camera_intrs"][0].cpu().float().numpy(),
            str(output_subdir),
        )

        # Gaussian splats
        gs_url          = None
        gs_rrd_url      = None
        gs_means_np     = None
        gs_colors_np    = None
        gs_opacities_np = None
        gs_scales_np    = None

        if "splats" in predictions:
            sp        = predictions["splats"]
            means     = sp["means"][0].reshape(-1, 3).cpu()
            scales    = sp["scales"][0].reshape(-1, 3).cpu()
            quats     = sp["quats"][0].reshape(-1, 4).cpu()
            colors    = (sp.get("sh", sp.get("colors"))[0]).reshape(-1, 3).cpu()
            opacities = sp["opacities"][0].reshape(-1).cpu()
            weights   = sp["weights"][0].reshape(-1).cpu() if "weights" in sp else torch.ones_like(opacities)

            if gs_filter_mask is not None:
                keep = torch.from_numpy(gs_filter_mask.reshape(-1)).bool()
                means, scales, quats = means[keep], scales[keep], quats[keep]
                colors, opacities, weights = colors[keep], opacities[keep], weights[keep]

            means, scales, quats, colors, opacities = _voxel_prune_gaussians(
                means, scales, quats, colors, opacities, weights
            )
            if means.shape[0] > 5_000_000:
                idx = torch.from_numpy(
                    np.random.default_rng(42).choice(means.shape[0], 5_000_000, replace=False)
                ).long()
                means, scales, quats, colors, opacities = (
                    means[idx], scales[idx], quats[idx], colors[idx], opacities[idx]
                )

            save_gs_ply(str(output_subdir / "gaussians.ply"), means, scales, quats, colors, opacities)
            gs_url = f"/download/{output_id}/gaussians.ply"

            gs_means_np     = means.float().numpy()
            gs_colors_np    = colors.float().numpy()
            gs_opacities_np = opacities.float().numpy()
            gs_scales_np    = scales.float().numpy()

        # Rerun recordings
        recon_rrd_path = build_rerun_reconstruction_recording(
            output_id=output_id,
            output_subdir=output_subdir,
            glb_path=glb_path,
            world_points=glb_preds["world_points"],
            images_np=glb_preds["images"],
            camera_poses=glb_preds["camera_poses"],
            camera_intrs=predictions["camera_intrs"][0].cpu().float().numpy(),
            filter_mask=list(filter_mask),
            normals=normals_np,
        )
        recon_rrd_url = f"/download/{output_id}/reconstruction.rrd"

        if gs_means_np is not None:
            build_rerun_gaussians_recording(
                output_id=output_id,
                output_subdir=output_subdir,
                means=gs_means_np,
                colors=gs_colors_np,
                opacities=gs_opacities_np,
                scales=gs_scales_np,
            )
            gs_rrd_url = f"/download/{output_id}/gaussians.rrd"

        del predictions, glb_preds, imgs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "success":        True,
            "infer_time":     infer_time,
            "glb_url":        f"/download/{output_id}/{glb_path.name}",
            "depth_urls":     depth_urls,
            "normal_urls":    normal_urls,
            "camera_url":     f"/download/{output_id}/camera_params.json",
            "gs_url":         gs_url,
            "recon_rrd_url":  recon_rrd_url,
            "gs_rrd_url":     gs_rrd_url,
            "num_views":      S,
            "output_id":      output_id,
            # Tell the frontend whether we have a local viewer
            "local_viewer":   RERUN_VIEWER_BASE is not None,
        }


# Serve uploaded/extracted frames as static files for preview
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return HTMLResponse(content=get_html_template())


@app.post("/api/upload")
async def api_upload(
    files: List[UploadFile] = File(...),
    time_interval: float    = Form(1.0),
):
    try:
        target_dir, image_paths = process_uploaded_files(files, time_interval)

        # Build preview URLs for extracted frames
        preview_urls = []
        for p in image_paths:
            rel = Path(p).relative_to(UPLOAD_DIR)
            preview_urls.append(f"/uploads/{rel.as_posix()}")

        return JSONResponse({
            "success":      True,
            "target_dir":   target_dir,
            "image_count":  len(image_paths),
            "image_paths":  image_paths,
            "preview_urls": preview_urls,
        })
    except Exception as e:
        import traceback
        return JSONResponse(
            {"success": False, "error": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/load_example")
async def api_load_example(
    filepath: str       = Form(...),
    time_interval: float = Form(1.0),
):
    """Load a local example file (image or video) and prepare it for reconstruction."""
    try:
        # Resolve relative paths against BASE_DIR for security
        abs_path = (BASE_DIR / filepath).resolve()
        # Ensure it stays within BASE_DIR
        abs_path.relative_to(BASE_DIR.resolve())

        target_dir, image_paths = process_example_file(str(abs_path), time_interval)

        # Build preview URLs for extracted frames
        preview_urls = []
        for p in image_paths:
            # Serve the extracted frames via a special endpoint
            rel = Path(p).relative_to(UPLOAD_DIR)
            preview_urls.append(f"/uploads/{rel.as_posix()}")

        return JSONResponse({
            "success":      True,
            "target_dir":   target_dir,
            "image_count":  len(image_paths),
            "image_paths":  image_paths,
            "preview_urls": preview_urls,
        })
    except Exception as e:
        import traceback
        return JSONResponse(
            {"success": False, "error": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


@app.post("/api/reconstruct")
async def api_reconstruct(
    target_dir: str        = Form(...),
    frame_selector: str    = Form("All"),
    show_camera: bool      = Form(True),
    filter_sky_bg: bool    = Form(False),
    show_mesh: bool        = Form(True),
    filter_ambiguous: bool = Form(True),
):
    try:
        result = run_full_reconstruction_pipeline(
            target_dir=target_dir,
            frame_selector=frame_selector,
            show_camera=show_camera,
            filter_sky_bg=filter_sky_bg,
            show_mesh=show_mesh,
            filter_ambiguous=filter_ambiguous,
        )
        return JSONResponse(result)
    except Exception as e:
        import traceback
        return JSONResponse(
            {"success": False, "error": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


@app.get("/download/{output_id}/{filename}")
async def download_file(output_id: str, filename: str):
    file_path = OUTPUT_DIR / f"recon_{output_id}" / filename
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    if filename.endswith(".rrd"):
        file_size = file_path.stat().st_size

        async def _stream():
            with open(file_path, "rb") as fh:
                while True:
                    chunk = fh.read(512 * 1024)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            _stream(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition":              f"inline; filename={filename}",
                "Content-Length":                   str(file_size),
                "Accept-Ranges":                    "bytes",
                "Access-Control-Allow-Origin":      "*",
                "Access-Control-Allow-Methods":     "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers":     "*",
                "Access-Control-Expose-Headers":    "Content-Length, Content-Type",
                "Cache-Control":                    "no-cache",
            },
        )

    return FileResponse(
        str(file_path),
        filename=filename,
        headers={"Access-Control-Allow-Origin": "*"},
    )

@app.options("/download/{output_id}/{filename}")
async def download_options(output_id: str, filename: str):
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


def get_html_template() -> str:
    # We pass the viewer base path into the template so JS knows where to point
    viewer_base = RERUN_VIEWER_BASE if RERUN_VIEWER_BASE else "__REMOTE__"

    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>HY-World-2.0-Demo - World Reconstruction</title>
<link href="https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root{
  --ub-aubergine:#2C001E;
  --ub-aubergine-dark:#1f0015;
  --ub-orange:#E95420;
  --ub-orange-hover:#c4461a;
  --ub-panel:#3D3D3D;
  --ub-panel-light:#4f4f4f;
  --ub-border:rgba(255,255,255,.1);
  --ub-text:#FFFFFF;
  --ub-muted:#b0b0b0;
  --ub-input:#2b2b2b;
  --r:8px;
}
*{box-sizing:border-box;font-family:'Ubuntu',sans-serif;}
body{margin:0;padding:0;background:var(--ub-aubergine);color:var(--ub-text);min-height:100vh;display:flex;flex-direction:column;}

/* topbar */
.topbar{background:var(--ub-aubergine-dark);padding:16px 24px;border-bottom:1px solid var(--ub-border);text-align:center;font-weight:700;letter-spacing:.5px;color:var(--ub-orange);font-size:1.1rem;}

/* container */
.container{max-width:1500px;margin:0 auto;padding:28px 20px;flex:1;width:100%;}
.header-text{text-align:center;margin-bottom:28px;}
.header-text h1{margin:0 0 8px;font-size:2.1rem;}
.header-text p{color:var(--ub-muted);margin:0;font-size:.95rem;}

/* layout */
.layout{display:grid;grid-template-columns:400px 1fr;gap:22px;align-items:stretch;}

/* panel */
.panel{background:var(--ub-panel);border-radius:var(--r);box-shadow:0 8px 24px rgba(0,0,0,.25);display:flex;flex-direction:column;overflow:hidden;}
.panel-header{padding:14px 20px;background:rgba(0,0,0,.22);border-bottom:1px solid var(--ub-border);font-weight:600;font-size:1rem;flex-shrink:0;}
.panel-body{flex:1;padding:18px;overflow-y:auto;display:flex;flex-direction:column;gap:16px;}

/* upload zone */
.upload-zone{background:var(--ub-input);border:2px dashed var(--ub-muted);border-radius:var(--r);padding:26px 16px;text-align:center;cursor:pointer;transition:border-color .2s,background .2s;}
.upload-zone:hover,.upload-zone.dragover{border-color:var(--ub-orange);background:rgba(233,84,32,.06);}
.upload-zone input[type=file]{display:none;}
.upload-svg{display:flex;justify-content:center;margin-bottom:10px;}
.upload-svg svg{width:42px;height:42px;stroke:var(--ub-orange);fill:none;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round;}
.upload-hint{color:var(--ub-muted);font-size:13px;line-height:1.55;}
.upload-hint strong{color:var(--ub-orange);}

.preview-grid{display:none;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:12px;}
.preview-thumb{aspect-ratio:1;border-radius:4px;overflow:hidden;border:1px solid var(--ub-border);}
.preview-thumb img{width:100%;height:100%;object-fit:cover;}

/* form elements */
.field-label{font-size:13px;font-weight:500;color:var(--ub-muted);margin-bottom:6px;display:block;}
.slider{width:100%;-webkit-appearance:none;height:5px;border-radius:3px;background:var(--ub-input);outline:none;}
.slider::-webkit-slider-thumb{-webkit-appearance:none;width:17px;height:17px;border-radius:50%;background:var(--ub-orange);cursor:pointer;}
.slider-val{text-align:center;margin-top:4px;color:var(--ub-muted);font-size:12px;}
.select{width:100%;background:var(--ub-input);border:1px solid var(--ub-border);color:var(--ub-text);padding:9px 11px;border-radius:4px;outline:none;font-size:13px;}
.select:focus{border-color:var(--ub-orange);}

.checks{display:flex;flex-direction:column;gap:10px;}
.check-row{display:flex;align-items:center;gap:9px;cursor:pointer;}
.check-row input[type=checkbox]{width:16px;height:16px;accent-color:var(--ub-orange);cursor:pointer;flex-shrink:0;}
.check-row span{font-size:13px;}

/* buttons */
.btn{width:100%;padding:12px;border:none;border-radius:4px;font-size:14px;font-weight:700;cursor:pointer;transition:background .2s;flex-shrink:0;}
.btn-primary{background:var(--ub-orange);color:#fff;box-shadow:0 4px 12px rgba(233,84,32,.28);}
.btn-primary:hover{background:var(--ub-orange-hover);}
.btn-primary:disabled{opacity:.55;cursor:not-allowed;}
.btn-sm{padding:9px;font-size:12px;font-weight:600;background:var(--ub-panel-light);color:var(--ub-text);border-radius:4px;text-decoration:none;display:block;text-align:center;transition:background .2s;}
.btn-sm:hover{background:#5a5a5a;}
.dl-group{display:none;flex-direction:column;gap:7px;}
.dl-group.show{display:flex;}

/* log */
.log-wrap{border:1px solid var(--ub-border);border-radius:4px;background:#1a001000;flex-shrink:0;}
.log-head{padding:6px 11px;font-size:11px;font-weight:700;color:var(--ub-muted);background:rgba(0,0,0,.35);border-bottom:1px solid var(--ub-border);text-transform:uppercase;letter-spacing:.5px;}
.log-body{padding:8px;font-family:'Courier New',monospace;font-size:11.5px;color:#ddd;max-height:130px;overflow-y:auto;white-space:pre-wrap;}
.li{color:#5bc0eb;} .ls{color:#9bc53d;} .le{color:#ff5e5b;}

/* viewer panel */
.vpanel{min-height:840px;}
.vtabs{display:flex;background:rgba(0,0,0,.28);border-bottom:1px solid var(--ub-border);flex-shrink:0;border-radius:var(--r) var(--r) 0 0;}
.vtab{padding:12px 20px;cursor:pointer;border-bottom:2px solid transparent;font-size:13px;color:var(--ub-muted);user-select:none;white-space:nowrap;transition:color .15s,background .15s;}
.vtab:hover{color:var(--ub-text);background:rgba(255,255,255,.05);}
.vtab.active{color:var(--ub-orange);border-bottom-color:var(--ub-orange);}

.vcontent{flex:1;position:relative;background:#0e0e0e;border-radius:0 0 var(--r) var(--r);overflow:hidden;min-height:790px;}
.vpane{display:none;width:100%;height:100%;min-height:790px;flex-direction:column;}
.vpane.active{display:flex;}

/* Rerun embed */
.rr-wrap{flex:1;min-height:790px;display:flex;flex-direction:column;position:relative;background:#0d0d0d;}
.rr-iframe{width:100%;flex:1;min-height:790px;border:none;background:#0d0d0d;}

/* placeholder */
.ph{width:100%;flex:1;min-height:790px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#0d0d0d;color:var(--ub-muted);text-align:center;gap:14px;padding:40px;}
.ph svg{width:68px;height:68px;opacity:.3;stroke:var(--ub-muted);}
.ph-title{font-size:1.05rem;color:var(--ub-text);font-weight:500;}
.ph-sub{font-size:13px;max-width:360px;line-height:1.7;}
.rr-badge{display:inline-flex;align-items:center;gap:5px;background:rgba(233,84,32,.1);border:1px solid var(--ub-orange);border-radius:20px;padding:2px 11px;font-size:11px;color:var(--ub-orange);font-weight:700;}

/* loading overlay inside rr-wrap */
.rr-loading{position:absolute;inset:0;background:rgba(13,13,13,.9);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;z-index:10;}
.rr-loading.hide{display:none;}
.rr-spinner{width:46px;height:46px;border:3px solid rgba(255,255,255,.08);border-top-color:var(--ub-orange);border-radius:50%;animation:spin 1s linear infinite;}
.rr-loading-text{font-size:13px;color:var(--ub-muted);}
@keyframes spin{to{transform:rotate(360deg);}}

/* image viewer */
.img-viewer{flex:1;min-height:790px;display:flex;align-items:center;justify-content:center;background:#080808;position:relative;}
.img-viewer img{max-width:100%;max-height:100%;object-fit:contain;}
.nav-bar{position:absolute;bottom:18px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.72);padding:8px 18px;border-radius:20px;display:flex;align-items:center;gap:14px;backdrop-filter:blur(4px);}
.nav-btn{background:none;border:none;color:var(--ub-text);font-size:18px;cursor:pointer;padding:4px 9px;transition:color .15s;}
.nav-btn:hover{color:var(--ub-orange);}
.nav-ind{color:var(--ub-muted);font-size:13px;min-width:72px;text-align:center;}

/* global loader (used for upload/example load only) */
.gloader{position:absolute;inset:0;background:rgba(44,0,30,.93);display:none;flex-direction:column;align-items:center;justify-content:center;z-index:200;}
.gloader.on{display:flex;}
.gspinner{width:52px;height:52px;border:3px solid rgba(255,255,255,.08);border-top-color:var(--ub-orange);border-radius:50%;animation:spin 1s linear infinite;margin-bottom:18px;}
.gloader-txt{font-weight:500;font-size:14px;color:#fff;letter-spacing:.5px;}

/* per-pane inline loader (used during reconstruction) */
.pane-loader{position:absolute;inset:0;background:rgba(13,13,13,.82);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;z-index:15;backdrop-filter:blur(3px);border-radius:0 0 var(--r) var(--r);}
.pane-loader .pane-spinner{width:44px;height:44px;border:3px solid rgba(255,255,255,.08);border-top-color:var(--ub-orange);border-radius:50%;animation:spin 1s linear infinite;}
.pane-loader .pane-loader-label{font-size:13px;color:var(--ub-muted);font-weight:500;letter-spacing:.3px;}
.pane-loader .pane-loader-sub{font-size:11px;color:rgba(255,255,255,.3);}

/* ---- Examples section ---- */
.examples-section{margin-top:28px;}
.examples-title{text-align:center;font-size:.85rem;font-weight:700;color:var(--ub-muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;}
.examples-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;max-width:1500px;margin:0 auto;}
.example-card{background:var(--ub-panel);border:1px solid var(--ub-border);border-radius:var(--r);overflow:hidden;cursor:pointer;transition:border-color .2s,transform .15s,box-shadow .2s;position:relative;}
.example-card:hover{border-color:var(--ub-orange);transform:translateY(-2px);box-shadow:0 8px 24px rgba(233,84,32,.18);}
.example-card:active{transform:translateY(0);}
.example-thumb{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;background:#111;}
.example-thumb-video{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;background:#111;pointer-events:none;}
.example-info{padding:10px 12px;}
.example-name{font-size:12px;font-weight:600;color:var(--ub-text);margin:0 0 2px;}
.example-type{font-size:11px;color:var(--ub-muted);}
.example-badge{position:absolute;top:7px;right:7px;background:rgba(0,0,0,.7);border:1px solid var(--ub-border);border-radius:3px;padding:2px 7px;font-size:10px;font-weight:700;color:var(--ub-muted);text-transform:uppercase;letter-spacing:.5px;}
.example-badge.img{border-color:rgba(91,192,235,.4);color:#5bc0eb;}
.example-badge.vid{border-color:rgba(233,84,32,.4);color:var(--ub-orange);}

@media(max-width:980px){
  .layout{grid-template-columns:1fr;}
  .vcontent,.vpane,.rr-wrap,.rr-iframe,.ph,.img-viewer{min-height:520px;}
  .examples-grid{grid-template-columns:repeat(2,1fr);}
}
</style>
</head>
<body>
<div class="topbar">HY-World-2.0-Demo — WorldMirror 2.0 World Reconstruction</div>

<div class="container">
  <div class="header-text">
    <h1>Universal 3D World Reconstruction</h1>
    <p>Upload images or video · depth maps · normals · point clouds · Gaussian splats · interactive Rerun 3D viewer</p>
  </div>

  <div class="layout">

    <!-- ======== LEFT: Settings ======== -->
    <div class="panel">
      <div class="panel-header">Settings</div>
      <div class="panel-body">

        <!-- Upload -->
        <div>
          <label class="field-label">Upload Images or Video</label>
          <div class="upload-zone" id="dropZone">
            <input type="file" id="fileInput" multiple accept="image/*,video/*"/>
            <div class="upload-svg">
              <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 15V3m0 0L8 7m4-4 4 4"/>
                <path d="M3 17v.5A2.5 2.5 0 0 0 5.5 20h13a2.5 2.5 0 0 0 2.5-2.5V17"/>
              </svg>
            </div>
            <div class="upload-hint">
              <strong>Click to browse</strong> or drag &amp; drop<br>
              PNG · JPG · HEIC · WEBP · MP4 · MOV · AVI
            </div>
          </div>
          <div class="preview-grid" id="previewGrid"></div>
        </div>

        <!-- Interval -->
        <div>
          <label class="field-label">Video Sample Interval</label>
          <input type="range" class="slider" id="timeInterval" min="0.1" max="10" step="0.1" value="1">
          <div class="slider-val"><span id="intervalValue">1.0</span> s</div>
        </div>

        <!-- Image Preview -->
        <div>
          <label class="field-label">Image Preview</label>
          <div id="imagePreviewBox" style="background:var(--ub-input);border:1px solid var(--ub-border);border-radius:var(--r);padding:12px;min-height:60px;">
            <div id="imagePreviewEmpty" style="text-align:center;color:var(--ub-muted);font-size:12px;padding:8px 0;">No images loaded yet. Upload files or select an example.</div>
            <div id="imagePreviewGrid" style="display:none;grid-template-columns:repeat(auto-fill,minmax(80px,1fr));gap:8px;"></div>
            <div id="imagePreviewCount" style="display:none;text-align:center;margin-top:8px;font-size:11px;color:var(--ub-muted);"></div>
          </div>
        </div>

        <!-- Frame selector -->
        <div>
          <label class="field-label">Frame Selector</label>
          <select class="select" id="frameSelector">
            <option value="All">All Frames</option>
          </select>
        </div>

        <!-- Checkboxes -->
        <div class="checks">
          <label class="check-row"><input type="checkbox" id="showCamera" checked><span>Show Camera Frustums</span></label>
          <label class="check-row"><input type="checkbox" id="showMesh"   checked><span>Show as Mesh (vs Point Cloud)</span></label>
          <label class="check-row"><input type="checkbox" id="filterAmb"  checked><span>Filter Low Confidence / Edges</span></label>
          <label class="check-row"><input type="checkbox" id="filterSky"        ><span>Filter Sky Background</span></label>
        </div>

        <!-- Reconstruct -->
        <button class="btn btn-primary" id="reconstructBtn" disabled>Reconstruct 3D Scene</button>

        <!-- Downloads -->
        <div class="dl-group" id="dlGroup">
          <a class="btn-sm" id="dlGLB"       target="_blank">Download GLB Mesh</a>
          <a class="btn-sm" id="dlCamera"    target="_blank">Download Camera JSON</a>
          <a class="btn-sm" id="dlReconRrd"  target="_blank">Download Reconstruction .rrd</a>
          <a class="btn-sm" id="dlGS"   style="display:none" target="_blank">Download Gaussian PLY</a>
          <a class="btn-sm" id="dlGsRrd" style="display:none" target="_blank">Download Gaussians .rrd</a>
        </div>

        <!-- Log -->
        <div class="log-wrap">
          <div class="log-head">Execution Log</div>
          <div class="log-body" id="logBody">[{DEVICE_LABEL}] System ready.</div>
        </div>

      </div>
    </div>

    <!-- ======== RIGHT: Viewer ======== -->
    <div class="panel vpanel">

      <div class="vtabs">
        <div class="vtab active" data-tab="reconstruction">3D Reconstruction</div>
        <div class="vtab"        data-tab="gaussian">Gaussian Splats</div>
        <div class="vtab"        data-tab="depth">Depth Maps</div>
        <div class="vtab"        data-tab="normal">Normal Maps</div>
      </div>

      <div class="vcontent">

        <!-- Global loader (upload/example only) -->
        <div class="gloader" id="gloader">
          <div class="gspinner"></div>
          <div class="gloader-txt" id="gloaderTxt">Loading...</div>
        </div>

        <!-- 3D Reconstruction -->
        <div class="vpane active" id="pane-reconstruction" style="position:relative;">
          <div class="ph" id="ph-reconstruction">
            <svg viewBox="0 0 24 24" fill="none" stroke-width="1.2" xmlns="http://www.w3.org/2000/svg">
              <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/>
              <polyline points="7.5 4.21 12 6.81 16.5 4.21"/>
              <polyline points="7.5 19.79 7.5 14.6 3 12"/>
              <polyline points="21 12 16.5 14.6 16.5 19.79"/>
              <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
              <line x1="12" y1="22.08" x2="12" y2="12"/>
            </svg>
            <div class="ph-title">3D Reconstruction Viewer</div>
            <div class="ph-sub">Upload images and click <strong>Reconstruct 3D Scene</strong>.<br>Results will be displayed here using <span class="rr-badge">Rerun</span> — full point cloud, cameras and normals.</div>
          </div>
        </div>

        <!-- Gaussian Splats -->
        <div class="vpane" id="pane-gaussian" style="position:relative;">
          <div class="ph" id="ph-gaussian">
            <svg viewBox="0 0 24 24" fill="none" stroke-width="1.2" xmlns="http://www.w3.org/2000/svg">
              <circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="7" opacity=".35"/><circle cx="12" cy="12" r="11" opacity=".12"/>
            </svg>
            <div class="ph-title">Gaussian Splats Viewer</div>
            <div class="ph-sub">Splat point cloud rendered via <span class="rr-badge">Rerun</span> after reconstruction.<br>Download the <code>.ply</code> for SuperSplat / PlayCanvas.</div>
          </div>
        </div>

        <!-- Depth -->
        <div class="vpane" id="pane-depth" style="position:relative;">
          <div class="img-viewer">
            <img id="depthImg" src="" alt="Depth" style="display:none">
            <div class="ph" id="ph-depth" style="position:absolute;inset:0;min-height:unset;">
              <svg viewBox="0 0 24 24" fill="none" stroke-width="1.2" xmlns="http://www.w3.org/2000/svg"><rect x="2" y="2" width="20" height="20" rx="2" opacity=".15"/><rect x="5" y="5" width="14" height="14" rx="1.5" opacity=".3"/><rect x="8" y="8" width="8" height="8" rx="1"/><line x1="12" y1="20" x2="12" y2="23" stroke-dasharray="1.5 1"/><line x1="2" y1="12" x2="2" y2="22" opacity=".25"/><line x1="22" y1="12" x2="22" y2="22" opacity=".25"/><path d="M10 12l2-2 2 2" stroke-width="1.4"/><line x1="12" y1="10" x2="12" y2="15" stroke-width="1.4"/></svg>
              <div class="ph-title">Depth Maps</div>
              <div class="ph-sub">Depth visualizations appear here after reconstruction.</div>
            </div>
            <div class="nav-bar" id="depthNav" style="display:none">
              <button class="nav-btn" id="depthPrev">&#9664;</button>
              <span class="nav-ind" id="depthInd">1 / 1</span>
              <button class="nav-btn" id="depthNext">&#9654;</button>
            </div>
          </div>
        </div>

        <!-- Normals -->
        <div class="vpane" id="pane-normal" style="position:relative;">
          <div class="img-viewer">
            <img id="normalImg" src="" alt="Normal" style="display:none">
            <div class="ph" id="ph-normal" style="position:absolute;inset:0;min-height:unset;">
              <svg viewBox="0 0 24 24" fill="none" stroke-width="1.2" xmlns="http://www.w3.org/2000/svg"><path d="M12 3L2 12h3v8h6v-6h2v6h6v-8h3L12 3z"/></svg>
              <div class="ph-title">Normal Maps</div>
              <div class="ph-sub">Surface normal visualizations appear here after reconstruction.</div>
            </div>
            <div class="nav-bar" id="normalNav" style="display:none">
              <button class="nav-btn" id="normalPrev">&#9664;</button>
              <span class="nav-ind" id="normalInd">1 / 1</span>
              <button class="nav-btn" id="normalNext">&#9654;</button>
            </div>
          </div>
        </div>

      </div><!-- /vcontent -->
    </div><!-- /vpanel -->
  </div><!-- /layout -->

  <!-- ======== EXAMPLES SECTION ======== -->
  <div class="examples-section">
    <div class="examples-title">Try an Example</div>
    <div class="examples-grid">

      <!-- Example 1: Image -->
      <div class="example-card" onclick="loadExample('example_gradio/1.png','image')">
        <img
          class="example-thumb"
          src="/examples/1.png"
          alt="Example Image"
          onerror="this.style.background='#1a1a1a';this.alt='Image Preview'"
        >
        <span class="example-badge img">IMG</span>
        <div class="example-info">
          <div class="example-name">Example Scene</div>
          <div class="example-type">Static image · PNG</div>
        </div>
      </div>

      <!-- Example 2: Video -->
      <div class="example-card" onclick="loadExample('example_gradio/1.mp4','video')">
        <video
          class="example-thumb-video"
          src="/examples/1.mp4"
          muted
          loop
          autoplay
          playsinline
          onerror="this.style.background='#1a1a1a'"
        ></video>
        <span class="example-badge vid">VID</span>
        <div class="example-info">
          <div class="example-name">Example Video</div>
          <div class="example-type">Multi-frame · MP4</div>
        </div>
      </div>

    </div>
  </div><!-- /examples-section -->

</div><!-- /container -->

<script>
// ===========================================================
// Config injected from Python
// ===========================================================
var VIEWER_BASE = "{VIEWER_BASE}"; // "/rerun-viewer" or "__REMOTE__"

// ===========================================================
// State
// ===========================================================
var targetDir      = null;
var depthUrls      = [];
var normalUrls     = [];
var depthIdx       = 0;
var normalIdx      = 0;

// ===========================================================
// Log
// ===========================================================
var logBody = document.getElementById('logBody');
function log(msg, cls) {
    cls = cls || '';
    var t = new Date().toLocaleTimeString('en-US',{hour12:false});
    var d = document.createElement('div');
    d.innerHTML = '<span style="color:#444">['+t+']</span> <span class="'+(cls||'')+'">'+msg+'</span>';
    logBody.appendChild(d);
    logBody.scrollTop = logBody.scrollHeight;
}

// ===========================================================
// Tabs
// ===========================================================
document.querySelectorAll('.vtab').forEach(function(tab){
    tab.addEventListener('click', function(){
        document.querySelectorAll('.vtab').forEach(function(t){t.classList.remove('active');});
        document.querySelectorAll('.vpane').forEach(function(p){p.classList.remove('active');});
        tab.classList.add('active');
        document.getElementById('pane-'+tab.dataset.tab).classList.add('active');
    });
});

// ===========================================================
// Upload
// ===========================================================
var dropZone = document.getElementById('dropZone');
var fileInput = document.getElementById('fileInput');
var gloader  = document.getElementById('gloader');
var gloaderTxt = document.getElementById('gloaderTxt');
var reconstructBtn = document.getElementById('reconstructBtn');

dropZone.addEventListener('click', function(){ fileInput.click(); });
fileInput.addEventListener('change', handleFiles);
dropZone.addEventListener('dragover', function(e){ e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', function(){ dropZone.classList.remove('dragover'); });
dropZone.addEventListener('drop', function(e){
    e.preventDefault(); dropZone.classList.remove('dragover');
    handleFiles({target:{files: e.dataTransfer.files}});
});

async function handleFiles(e) {
    var files = Array.from(e.target.files);
    if (!files.length) return;
    showPreviews(files);
    log('Uploading '+files.length+' file(s)...','li');
    gloaderTxt.textContent = 'Uploading...';
    gloader.classList.add('on');

    var fd = new FormData();
    files.forEach(function(f){ fd.append('files', f); });
    fd.append('time_interval', document.getElementById('timeInterval').value);

    try {
        var r = await fetch('/api/upload',{method:'POST',body:fd});
        var d = await r.json();
        if (d.success) {
            targetDir = d.target_dir;
            log('Uploaded '+d.image_count+' image(s). Ready to reconstruct.','ls');
            reconstructBtn.disabled = false;
            var sel = document.getElementById('frameSelector');
            sel.innerHTML = '<option value="All">All Frames</option>';
            for (var i=0; i<d.image_count; i++){
                sel.innerHTML += '<option value="'+i+'">Frame '+(i+1)+'</option>';
            }
            // Show image preview from server-extracted frames
            if (d.preview_urls && d.preview_urls.length) {
                showImagePreview(d.preview_urls);
            }
        } else { log('Upload failed: '+d.error,'le'); }
    } catch(err){ log('Upload error: '+err.message,'le'); }
    finally { gloader.classList.remove('on'); }
}

function showPreviews(files){
    var imgs = files.filter(function(f){ return f.type.startsWith('image/'); }).slice(0,4);
    var pg = document.getElementById('previewGrid');
    pg.innerHTML = '';
    pg.style.display = imgs.length ? 'grid' : 'none';
    imgs.forEach(function(f){
        var dv = document.createElement('div'); dv.className='preview-thumb';
        var im = document.createElement('img'); im.src=URL.createObjectURL(f);
        dv.appendChild(im); pg.appendChild(dv);
    });
}

function showImagePreview(urls) {
    var grid = document.getElementById('imagePreviewGrid');
    var empty = document.getElementById('imagePreviewEmpty');
    var countEl = document.getElementById('imagePreviewCount');
    grid.innerHTML = '';
    empty.style.display = 'none';
    grid.style.display = 'grid';
    countEl.style.display = 'block';
    countEl.textContent = urls.length + ' frame(s) extracted';
    urls.forEach(function(url, idx) {
        var thumb = document.createElement('div');
        thumb.style.cssText = 'aspect-ratio:1;border-radius:4px;overflow:hidden;border:1px solid var(--ub-border);background:#111;position:relative;';
        var img = document.createElement('img');
        img.src = url;
        img.alt = 'Frame ' + (idx + 1);
        img.style.cssText = 'width:100%;height:100%;object-fit:cover;display:block;';
        img.onerror = function(){ this.style.display='none'; };
        var label = document.createElement('div');
        label.style.cssText = 'position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.7);color:#fff;font-size:10px;text-align:center;padding:2px 0;';
        label.textContent = 'F' + (idx + 1);
        thumb.appendChild(img);
        thumb.appendChild(label);
        grid.appendChild(thumb);
    });
}

function clearImagePreview() {
    var grid = document.getElementById('imagePreviewGrid');
    var empty = document.getElementById('imagePreviewEmpty');
    var countEl = document.getElementById('imagePreviewCount');
    grid.innerHTML = '';
    grid.style.display = 'none';
    empty.style.display = 'block';
    countEl.style.display = 'none';
}

document.getElementById('timeInterval').addEventListener('input', function(){
    document.getElementById('intervalValue').textContent = parseFloat(this.value).toFixed(1);
});

// ===========================================================
// Load Example
// ===========================================================
async function loadExample(filepath, type) {
    log('Loading example: ' + filepath, 'li');
    gloaderTxt.textContent = 'Loading example...';
    gloader.classList.add('on');

    // Show a quick preview thumbnail for image examples in the upload area
    if (type === 'image') {
        var pg = document.getElementById('previewGrid');
        pg.innerHTML = '';
        var dv = document.createElement('div'); dv.className='preview-thumb';
        var im = document.createElement('img'); im.src='/examples/' + filepath.split('/').pop();
        dv.appendChild(im); pg.appendChild(dv);
        pg.style.display = 'grid';
    } else {
        document.getElementById('previewGrid').style.display = 'none';
    }

    var fd = new FormData();
    fd.append('filepath', filepath);
    fd.append('time_interval', document.getElementById('timeInterval').value);

    try {
        var r = await fetch('/api/load_example', {method:'POST', body:fd});
        var d = await r.json();
        if (d.success) {
            targetDir = d.target_dir;
            log('Example loaded: ' + d.image_count + ' image(s). Ready to reconstruct.', 'ls');
            reconstructBtn.disabled = false;
            var sel = document.getElementById('frameSelector');
            sel.innerHTML = '<option value="All">All Frames</option>';
            for (var i = 0; i < d.image_count; i++) {
                sel.innerHTML += '<option value="'+i+'">Frame '+(i+1)+'</option>';
            }
            // Show image preview with extracted frames
            if (d.preview_urls && d.preview_urls.length) {
                showImagePreview(d.preview_urls);
            }
            // Scroll settings panel into view
            document.getElementById('reconstructBtn').scrollIntoView({behavior:'smooth', block:'nearest'});
        } else {
            log('Failed to load example: ' + d.error, 'le');
        }
    } catch(err) {
        log('Example load error: ' + err.message, 'le');
    } finally {
        gloader.classList.remove('on');
    }
}

// ===========================================================
// Reconstruct
// ===========================================================
// Per-pane inline loader helpers
var PANE_IDS = ['pane-reconstruction', 'pane-gaussian', 'pane-depth', 'pane-normal'];
var PANE_LABELS = {
    'pane-reconstruction': 'Building 3D reconstruction...',
    'pane-gaussian':       'Generating Gaussian splats...',
    'pane-depth':          'Computing depth maps...',
    'pane-normal':         'Computing normal maps...'
};

function showPaneLoaders() {
    PANE_IDS.forEach(function(id) {
        var pane = document.getElementById(id);
        if (!pane) return;
        // Remove any existing loader first
        var old = pane.querySelector('.pane-loader');
        if (old) old.remove();
        var loader = document.createElement('div');
        loader.className = 'pane-loader';
        loader.innerHTML =
            '<div class="pane-spinner"></div>' +
            '<div class="pane-loader-label">' + (PANE_LABELS[id] || 'Processing...') + '</div>' +
            '<div class="pane-loader-sub">This may take a moment</div>';
        pane.appendChild(loader);
    });
}

function removePaneLoaders() {
    PANE_IDS.forEach(function(id) {
        var pane = document.getElementById(id);
        if (!pane) return;
        var loader = pane.querySelector('.pane-loader');
        if (loader) loader.remove();
    });
}

reconstructBtn.addEventListener('click', async function(){
    if (!targetDir) return;
    log('Starting reconstruction...','li');
    reconstructBtn.disabled = true;

    // Show per-pane inline loaders (containers stay visible)
    showPaneLoaders();

    var fd = new FormData();
    fd.append('target_dir',       targetDir);
    fd.append('frame_selector',   document.getElementById('frameSelector').value);
    fd.append('show_camera',      document.getElementById('showCamera').checked);
    fd.append('filter_sky_bg',    document.getElementById('filterSky').checked);
    fd.append('show_mesh',        document.getElementById('showMesh').checked);
    fd.append('filter_ambiguous', document.getElementById('filterAmb').checked);

    try {
        var r = await fetch('/api/reconstruct',{method:'POST',body:fd});
        var d = await r.json();
        if (d.success){
            log('Reconstruction complete in '+d.infer_time.toFixed(2)+'s ('+d.num_views+' views)','ls');
            removePaneLoaders();
            applyResults(d);
        } else {
            log('Failed: '+d.error,'le');
            console.error(d.trace);
            removePaneLoaders();
        }
    } catch(err){ log('Error: '+err.message,'le'); removePaneLoaders(); }
    finally { reconstructBtn.disabled=false; }
});

// ===========================================================
// Apply results
// ===========================================================
function applyResults(d){
    // Downloads
    document.getElementById('dlGLB').href      = d.glb_url;
    document.getElementById('dlCamera').href   = d.camera_url;
    document.getElementById('dlReconRrd').href = d.recon_rrd_url;
    document.getElementById('dlGroup').classList.add('show');
    if (d.gs_url){
        var g=document.getElementById('dlGS'); g.href=d.gs_url; g.style.display='block';
    }
    if (d.gs_rrd_url){
        var g=document.getElementById('dlGsRrd'); g.href=d.gs_rrd_url; g.style.display='block';
    }

    // Rerun viewers
    mountRerun('pane-reconstruction','ph-reconstruction', d.recon_rrd_url, '3D Reconstruction');
    if (d.gs_rrd_url){
        mountRerun('pane-gaussian','ph-gaussian', d.gs_rrd_url, 'Gaussian Splats');
    } else {
        var ph=document.getElementById('ph-gaussian');
        if(ph){ var s=ph.querySelector('.ph-sub'); if(s) s.textContent='No Gaussian splats were generated for this scene.'; }
    }

    // Image carousels
    depthUrls  = d.depth_urls  || [];
    normalUrls = d.normal_urls || [];
    depthIdx   = 0;
    normalIdx  = 0;
    refreshDepth();
    refreshNormal();

    // Switch tab
    document.querySelector('[data-tab="reconstruction"]').click();
}

// ===========================================================
// Rerun embed — robust two-strategy approach
// ===========================================================
function buildRerunUrl(rrdRelative) {
    /*
     * Build the absolute URL that the Rerun web viewer should fetch.
     * rrdRelative is like  /download/abc123/reconstruction.rrd
     *
     * We always use an absolute URL so that when the iframe's src points
     * to an external host (app.rerun.io) it can fetch back to our server.
     */
    return window.location.origin + rrdRelative;
}

function mountRerun(paneId, phId, rrdRelative, label) {
    var pane = document.getElementById(paneId);
    var ph   = document.getElementById(phId);
    if (!pane || !rrdRelative) return;

    // Remove old iframe if exists
    var old = pane.querySelector('.rr-wrap');
    if (old) old.remove();

    // Remove placeholder
    if (ph) ph.style.display = 'none';

    var absUrl = buildRerunUrl(rrdRelative);

    // Decide viewer URL
    var viewerSrc;
    if (VIEWER_BASE !== '__REMOTE__') {
        // Local viewer bundled with rerun package
        viewerSrc = VIEWER_BASE + '/index.html?url=' + encodeURIComponent(absUrl);
    } else {
        // Fallback: app.rerun.io
        viewerSrc = 'https://app.rerun.io/version/latest/index.html?url=' + encodeURIComponent(absUrl);
    }

    var wrap = document.createElement('div');
    wrap.className = 'rr-wrap';

    // Loading overlay inside the wrap
    var loadingDiv = document.createElement('div');
    loadingDiv.className = 'rr-loading';
    loadingDiv.innerHTML =
        '<div class="rr-spinner"></div>' +
        '<div class="rr-loading-text">Loading Rerun viewer...</div>';
    wrap.appendChild(loadingDiv);

    var iframe = document.createElement('iframe');
    iframe.className       = 'rr-iframe';
    iframe.title           = label;
    iframe.allow           = 'accelerometer; autoplay; clipboard-write; gyroscope; fullscreen; xr-spatial-tracking';
    iframe.allowFullscreen = true;

    iframe.addEventListener('load', function(){
        loadingDiv.classList.add('hide');
        log('Rerun viewer loaded: '+label,'ls');
    });

    iframe.addEventListener('error', function(){
        loadingDiv.classList.add('hide');
        showRerunFallback(wrap, absUrl, label);
        log('Rerun viewer error for: '+label,'le');
    });

    iframe.src = viewerSrc;
    wrap.appendChild(iframe);
    pane.appendChild(wrap);
}

function showRerunFallback(container, absUrl, label) {
    /*
     * If the iframe fails entirely (e.g. local assets not found and
     * app.rerun.io is blocked), show a clear message with a direct link.
     */
    var fb = document.createElement('div');
    fb.className = 'ph';
    fb.style.position = 'absolute';
    fb.style.inset     = '0';
    fb.style.minHeight = 'unset';
    fb.innerHTML =
        '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.2" style="width:64px;height:64px;opacity:.3;stroke:var(--ub-muted)">' +
        '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>' +
        '<div class="ph-title">Viewer Unavailable</div>' +
        '<div class="ph-sub">The interactive viewer could not load.<br>' +
        'Download the <code>.rrd</code> file and open it in the<br>' +
        '<a href="https://rerun.io/viewer" target="_blank" style="color:var(--ub-orange)">Rerun desktop app</a> ' +
        'or <a href="https://app.rerun.io" target="_blank" style="color:var(--ub-orange)">app.rerun.io</a>.' +
        '</div>' +
        '<a href="'+absUrl+'" download style="color:var(--ub-orange);font-size:13px;margin-top:8px;">Download '+label+' .rrd</a>';
    container.innerHTML = '';
    container.appendChild(fb);
}

// ===========================================================
// Depth carousel
// ===========================================================
function refreshDepth(){
    var img = document.getElementById('depthImg');
    var ph  = document.getElementById('ph-depth');
    var nav = document.getElementById('depthNav');
    if (!depthUrls.length) return;
    img.src = depthUrls[depthIdx];
    img.style.display = 'block';
    if (ph) ph.style.display = 'none';
    nav.style.display = 'flex';
    document.getElementById('depthInd').textContent = (depthIdx+1)+' / '+depthUrls.length;
}
document.getElementById('depthPrev').addEventListener('click',function(){
    if(depthIdx>0){depthIdx--;refreshDepth();}
});
document.getElementById('depthNext').addEventListener('click',function(){
    if(depthIdx<depthUrls.length-1){depthIdx++;refreshDepth();}
});

// ===========================================================
// Normal carousel
// ===========================================================
function refreshNormal(){
    var img = document.getElementById('normalImg');
    var ph  = document.getElementById('ph-normal');
    var nav = document.getElementById('normalNav');
    if (!normalUrls.length) return;
    img.src = normalUrls[normalIdx];
    img.style.display = 'block';
    if (ph) ph.style.display = 'none';
    nav.style.display = 'flex';
    document.getElementById('normalInd').textContent = (normalIdx+1)+' / '+normalUrls.length;
}
document.getElementById('normalPrev').addEventListener('click',function(){
    if(normalIdx>0){normalIdx--;refreshNormal();}
});
document.getElementById('normalNext').addEventListener('click',function(){
    if(normalIdx<normalUrls.length-1){normalIdx++;refreshNormal();}
});
</script>
</body>
</html>
"""
    return (html
            .replace("{DEVICE_LABEL}", str(DEVICE_LABEL))
            .replace("{VIEWER_BASE}",  viewer_base))

demo = app

if __name__ == "__main__":
    demo.launch(        
        mcp_server=True,
        ssr_mode=False,
        show_error=True,
        allowed_paths=["example_gradio"],)