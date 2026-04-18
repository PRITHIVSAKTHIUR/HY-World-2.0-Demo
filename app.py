"""
HY-World-2.0-Demo - WorldMirror 2.0 World Reconstruction
OrangeRed Gradio Theme · gr.Tabs · Rerun for 3D + Gaussian Splats · gr.Gallery Preview

Requires Gradio version: 5.49.1
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import shutil
import time
import uuid
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
from PIL import Image

import rerun as rr
try:
    import rerun.blueprint as rrb
except ImportError:
    rrb = None

from gradio_rerun import Rerun

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

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

import gradio as gr
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)


class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )


orange_red_theme = OrangeRedTheme()

css = """
#col-container {
    margin: 0 auto;
    max-width: 1400px;
}
#main-title h1 { font-size: 2.4em !important; }
"""

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

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


def process_uploaded_files(files, time_interval: float = 1.0):
    """Accept a list of file paths from Gradio uploads.
    Returns (target_dir, sorted_image_paths).
    """
    target_dir = UPLOAD_DIR / f"input_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []

    for f in files:
        src = Path(f) if isinstance(f, str) else Path(f.name)
        if not src.exists():
            continue
        ext       = src.suffix.lower()
        base_name = src.stem

        if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}:
            cap      = cv2.VideoCapture(str(src))
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
            img = Image.open(str(src))
            dst = images_dir / f"{base_name}.jpg"
            img.convert("RGB").save(dst, "JPEG", quality=95)
            image_paths.append(str(dst))
        else:
            dst = images_dir / src.name
            shutil.copy2(str(src), str(dst))
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


def on_files_uploaded(files, time_interval):
    """
    Triggered when the user uploads files.
    Extracts frames, returns gallery images + stored target_dir + frame choices.
    """
    if not files:
        return [], None, gr.update(choices=["All"], value="All")

    target_dir, img_paths = process_uploaded_files(files, time_interval)

    # Build gallery list: list of (filepath, caption) tuples
    gallery_items = [(p, Path(p).stem) for p in img_paths]

    # Frame selector choices
    choices = ["All"] + [str(i) for i in range(len(img_paths))]

    return gallery_items, target_dir, gr.update(choices=choices, value="All")


def _make_rec(app_id: str) -> "rr.RecordingStream":
    run_id = str(uuid.uuid4())
    if hasattr(rr, "new_recording"):
        return rr.new_recording(application_id=app_id, recording_id=run_id)
    return rr.RecordingStream(application_id=app_id, recording_id=run_id)


def build_reconstruction_rrd(
    output_subdir: Path,
    glb_path: Path,
    world_points: np.ndarray,
    images_np: np.ndarray,
    camera_poses: np.ndarray,
    camera_intrs: Optional[np.ndarray],
    filter_mask: list,
    normals: Optional[np.ndarray],
) -> str:
    """Build a Rerun recording for the full scene reconstruction."""
    rrd_path = str(output_subdir / "reconstruction.rrd")
    rec      = _make_rec("HY-World-2.0-Reconstruction")

    rec.log("world", rr.Clear(recursive=True), static=True)
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    # Axis helpers
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
        img_hwc = np.transpose(images_np[i], (1, 2, 0))
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
            pts_f = pts.reshape(-1, 3)
            col_f = img_u8.reshape(-1, 3)
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
                    rr.Points3D(
                        positions=pts_f.astype(np.float32),
                        colors=col_f,
                        radii=0.003,
                    ),
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

        # Normal map
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
                            origin=f"/world/cameras/cam_000/pinhole/image",
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
            print(f"[Rerun] Reconstruction blueprint failed (non-fatal): {e}")

    rec.save(rrd_path)
    print(f"[Rerun] Reconstruction saved -> {rrd_path}")
    return rrd_path


def build_gaussians_rrd(
    output_subdir: Path,
    means: np.ndarray,
    colors_np: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
) -> str:
    """
    Build a Rerun recording for Gaussian splats.
    Orientation fix: WorldMirror uses a right-hand Y-up convention but the
    splat means are stored in camera/world space where +Y points DOWN
    (OpenCV convention). We flip Y so the cloud is upright in the viewer.
    """
    rrd_path = str(output_subdir / "gaussians.rrd")
    rec      = _make_rec("HY-World-2.0-Gaussians")

    # Flip Y to convert from OpenCV (Y-down) to viewer (Y-up)
    means_corrected       = means.copy()
    means_corrected[:, 1] = -means_corrected[:, 1]

    rec.log("world", rr.Clear(recursive=True), static=True)
    # Use RIGHT_HAND_Y_UP so +Y is treated as up in the viewer
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    colors_u8   = (np.clip(colors_np, 0, 1) * 255).astype(np.uint8) if colors_np.dtype != np.uint8 else colors_np
    alpha       = (np.clip(opacities, 0, 1) * 255).astype(np.uint8).reshape(-1, 1)
    colors_rgba = np.concatenate([colors_u8, alpha], axis=1)

    radii = np.clip(np.linalg.norm(scales, axis=1) * 0.5, 0.001, 0.1).astype(np.float32)

    MAX_PTS = 2_000_000
    N       = means_corrected.shape[0]
    if N > MAX_PTS:
        sel           = np.random.default_rng(42).choice(N, size=MAX_PTS, replace=False)
        means_corrected = means_corrected[sel]
        colors_rgba   = colors_rgba[sel]
        radii         = radii[sel]

    try:
        rec.log(
            "world/gaussian_splats",
            rr.Points3D(
                positions=means_corrected.astype(np.float32),
                colors=colors_rgba,
                radii=radii,
            ),
            static=True,
        )
    except Exception as e:
        print(f"[Rerun] Gaussians Points3D failed: {e}")

    try:
        rec.log("world/axes/x", rr.Arrows3D(vectors=[[0.5, 0, 0]], colors=[[255, 50, 50]]),  static=True)
        rec.log("world/axes/y", rr.Arrows3D(vectors=[[0, 0.5, 0]], colors=[[50, 200, 50]]),  static=True)
        rec.log("world/axes/z", rr.Arrows3D(vectors=[[0, 0, 0.5]], colors=[[50, 100, 220]]), static=True)
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


@spaces.GPU(duration=120)
def run_reconstruction_pipeline(
    target_dir: str,
    frame_selector: str,
    show_camera: bool,
    filter_sky_bg: bool,
    show_mesh: bool,
    filter_ambiguous: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Runs the WorldMirror pipeline on the already-extracted frames in target_dir.

    Returns:
        recon_rrd_path  str
        depth_imgs      List[str]
        depth_first     str | None
        normal_imgs     List[str]
        normal_first    str | None
        status          str
        gs_rrd_path     str | None
        gs_ply_path     str | None
    """
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

    if not target_dir:
        raise gr.Error("Please upload files first before running reconstruction.")

    image_folder = Path(target_dir) / "images"
    img_paths = sorted(
        glob(str(image_folder / "*.png"))  +
        glob(str(image_folder / "*.jpg"))  +
        glob(str(image_folder / "*.jpeg")) +
        glob(str(image_folder / "*.webp"))
    )
    if not img_paths:
        raise gr.Error("No valid images found. Please re-upload your files.")

    progress(0.10, desc="Loading model...")
    model, device = get_model_and_device()
    effective     = compute_adaptive_target_size(img_paths, max_target_size=952)

    progress(0.20, desc="Running inference...")
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

        progress(0.45, desc="Computing masks...")
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

        # Assemble glb_preds dict
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

        progress(0.60, desc="Building GLB mesh...")
        glb_scene = convert_predictions_to_glb_scene(
            glb_preds,
            filter_by_frames=frame_selector,
            show_camera=show_camera,
            mask_sky_bg=filter_sky_bg,
            as_mesh=show_mesh,
            mask_ambiguous=filter_ambiguous,
        )
        glb_scene.export(file_obj=str(glb_path))

        progress(0.70, desc="Building Rerun reconstruction recording...")
        recon_rrd_path = build_reconstruction_rrd(
            output_subdir=output_subdir,
            glb_path=glb_path,
            world_points=glb_preds["world_points"],
            images_np=glb_preds["images"],
            camera_poses=glb_preds["camera_poses"],
            camera_intrs=predictions["camera_intrs"][0].cpu().float().numpy(),
            filter_mask=list(filter_mask),
            normals=normals_np,
        )

        progress(0.78, desc="Rendering depth and normal maps...")
        depth_imgs, normal_imgs = [], []
        for i in range(S):
            depth = predictions["depth"][0, i].cpu().float().numpy().squeeze()
            mask  = filter_mask[i] if i < len(filter_mask) else None

            depth_p = output_subdir / f"depth_{i:03d}.png"
            Image.fromarray(render_depth_colormap(depth, mask)).save(depth_p)
            depth_imgs.append(str(depth_p))

            normal = np.zeros((H, W, 3), dtype=np.float32)
            for k in ("normals", "normal"):
                if k in predictions and predictions[k] is not None:
                    normal = predictions[k][0, i].cpu().float().numpy()
                    break
            normal_p = output_subdir / f"normal_{i:03d}.png"
            Image.fromarray(render_normal_colormap(normal, mask)).save(normal_p)
            normal_imgs.append(str(normal_p))

        save_camera_params(
            predictions["camera_poses"][0].cpu().float().numpy(),
            predictions["camera_intrs"][0].cpu().float().numpy(),
            str(output_subdir),
        )

        progress(0.88, desc="Processing Gaussian splats...")
        gs_rrd_path = None
        gs_ply_path = None

        if "splats" in predictions:
            sp        = predictions["splats"]
            means     = sp["means"][0].reshape(-1, 3).cpu()
            scales    = sp["scales"][0].reshape(-1, 3).cpu()
            quats     = sp["quats"][0].reshape(-1, 4).cpu()
            colors_t  = (sp.get("sh", sp.get("colors"))[0]).reshape(-1, 3).cpu()
            opacities = sp["opacities"][0].reshape(-1).cpu()
            weights   = sp["weights"][0].reshape(-1).cpu() if "weights" in sp else torch.ones_like(opacities)

            if gs_filter_mask is not None:
                keep = torch.from_numpy(gs_filter_mask.reshape(-1)).bool()
                means, scales, quats = means[keep], scales[keep], quats[keep]
                colors_t, opacities, weights = colors_t[keep], opacities[keep], weights[keep]

            means, scales, quats, colors_t, opacities = _voxel_prune_gaussians(
                means, scales, quats, colors_t, opacities, weights
            )
            if means.shape[0] > 5_000_000:
                idx = torch.from_numpy(
                    np.random.default_rng(42).choice(means.shape[0], 5_000_000, replace=False)
                ).long()
                means, scales, quats, colors_t, opacities = (
                    means[idx], scales[idx], quats[idx], colors_t[idx], opacities[idx]
                )

            ply_out = str(output_subdir / "gaussians.ply")
            save_gs_ply(ply_out, means, scales, quats, colors_t, opacities)
            gs_ply_path = ply_out

            progress(0.94, desc="Building Rerun recording for Gaussian splats...")
            gs_rrd_path = build_gaussians_rrd(
                output_subdir=output_subdir,
                means=means.float().numpy(),
                colors_np=colors_t.float().numpy(),
                opacities=opacities.float().numpy(),
                scales=scales.float().numpy(),
            )

        del predictions, glb_preds, imgs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        status = (
            f"Reconstruction complete — {S} view(s), "
            f"inference time: {infer_time:.2f}s"
        )

        return (
            recon_rrd_path,
            depth_imgs,
            depth_imgs[0]   if depth_imgs  else None,
            normal_imgs,
            normal_imgs[0]  if normal_imgs else None,
            status,
            gs_rrd_path,
            gs_ply_path,
        )


with gr.Blocks(theme=orange_red_theme, css=css, delete_cache=(600, 600)) as demo:

    gr.Markdown(
        "# **HY-World-2.0-Demo — WorldMirror 2.0 World Reconstruction**",
        elem_id="main-title",
    )
    gr.Markdown(
        "Upload images or video to reconstruct a full 3D scene: "
        "point clouds, camera poses, depth maps, surface normals and Gaussian splats — "
        "powered by [HY-World-2.0 / WorldMirror](https://huggingface.co/papers/2604.14268). "
        "After Reconstruction hold tight, the rendering will happen in fraction of seconds."
    )

    # Persistent state
    target_dir_state  = gr.State(None)
    depth_list_state  = gr.State([])
    normal_list_state = gr.State([])
    depth_idx_state   = gr.State(0)
    normal_idx_state  = gr.State(0)

    with gr.Row(elem_id="col-container"):

        with gr.Column(scale=1, min_width=380):

            file_upload = gr.File(
                label="Upload Images or Video",
                file_count="multiple",
                file_types=["image", "video", ".heic", ".heif"],
            )

            # Input frame preview gallery
            input_gallery = gr.Gallery(
                label="Input Frames Preview",
                columns=2,
                rows=2,
                height=220,
                object_fit="cover",
                show_label=True,
                preview=False,
            )

            time_interval = gr.Slider(
                0.1, 10.0, value=1.0, step=0.1,
                label="Video Interval (secs)",
            )

            frame_selector = gr.Dropdown(
                choices=["All"],
                value="All",
                label="Frame Selector",
            )

            with gr.Accordion("Reconstruction Options", open=False):
                show_camera      = gr.Checkbox(value=True,  label="Show Camera Frustums")
                show_mesh        = gr.Checkbox(value=True,  label="Show as Mesh (vs Point Cloud)")
                filter_ambiguous = gr.Checkbox(value=True,  label="Filter Low Confidence / Edges")
                filter_sky_bg    = gr.Checkbox(value=False, label="Filter Sky Background")

            btn_reconstruct = gr.Button("Reconstruct 3D Scene", variant="primary")

            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
                visible=False,
                placeholder="Awaiting reconstruction...",
            )

            with gr.Accordion("Downloads", open=False):
                dl_glb    = gr.DownloadButton(label="Download GLB Mesh",     variant="secondary")
                dl_gs_ply = gr.DownloadButton(label="Download Gaussian PLY", variant="secondary")

        with gr.Column(scale=2):

            with gr.Tabs():

                with gr.Tab("3D Reconstruction"):
                    gr.Markdown("### 3D Scene — Rerun Viewer")
                    rerun_recon = Rerun(
                        label="3D Reconstruction — Rerun Viewer",
                        height=630,
                    )

                with gr.Tab("Gaussian Splats"):
                    gr.Markdown("### Gaussian Splat Point Cloud — Rerun Viewer")
                    rerun_gs = Rerun(
                        label="Gaussian Splats — Rerun Viewer",
                        height=630,
                    )
                    gr.Markdown(
                        "The PLY file (downloadable above) can also be opened in "
                        "[SuperSplat](https://supersplat.xyz) or PlayCanvas for full splat rendering."
                    )

                with gr.Tab("Depth Maps"):
                    gr.Markdown("### Per-View Depth Maps")
                    depth_image = gr.Image(
                        label="Depth Map",
                        type="filepath",
                        height=540,
                        interactive=False,
                    )
                    with gr.Row():
                        btn_depth_prev = gr.Button("Previous", variant="secondary", scale=1)
                        depth_counter  = gr.Textbox(
                            value="–", label="Frame", interactive=False, scale=1
                        )
                        btn_depth_next = gr.Button("Next", variant="secondary", scale=1)

                with gr.Tab("Normal Maps"):
                    gr.Markdown("### Per-View Surface Normal Maps")
                    normal_image = gr.Image(
                        label="Normal Map",
                        type="filepath",
                        height=540,
                        interactive=False,
                    )
                    with gr.Row():
                        btn_normal_prev = gr.Button("Previous", variant="secondary", scale=1)
                        normal_counter  = gr.Textbox(
                            value="–", label="Frame", interactive=False, scale=1
                        )
                        btn_normal_next = gr.Button("Next", variant="secondary", scale=1)
        
    file_upload.upload(
        on_files_uploaded,
        inputs=[file_upload, time_interval],
        outputs=[input_gallery, target_dir_state, frame_selector],
    )

    time_interval.change(
        on_files_uploaded,
        inputs=[file_upload, time_interval],
        outputs=[input_gallery, target_dir_state, frame_selector],
    )

    def reconstruct(
        target_dir,
        frame_selector,
        show_camera,
        filter_sky_bg,
        show_mesh,
        filter_ambiguous,
        progress=gr.Progress(track_tqdm=True),
    ):
        (
            recon_rrd_path,
            depth_list,
            depth_first,
            normal_list,
            normal_first,
            status,
            gs_rrd_path,
            gs_ply_path,
        ) = run_reconstruction_pipeline(
            target_dir,
            frame_selector,
            show_camera,
            filter_sky_bg,
            show_mesh,
            filter_ambiguous,
            progress=progress,
        )

        depth_ctr  = f"1 / {len(depth_list)}"  if depth_list  else "–"
        normal_ctr = f"1 / {len(normal_list)}" if normal_list else "–"

        return (
            recon_rrd_path,  # rerun_recon
            gs_rrd_path,     # rerun_gs
            depth_first,     # depth_image
            depth_list,      # depth_list_state
            0,               # depth_idx_state
            depth_ctr,       # depth_counter
            normal_first,    # normal_image
            normal_list,     # normal_list_state
            0,               # normal_idx_state
            normal_ctr,      # normal_counter
            status,          # status_box
            gs_ply_path,     # dl_glb  (GLB embedded in rrd; offer ply download)
            gs_ply_path,     # dl_gs_ply
        )

    btn_reconstruct.click(
        reconstruct,
        inputs=[
            target_dir_state, frame_selector,
            show_camera, filter_sky_bg, show_mesh, filter_ambiguous,
        ],
        outputs=[
            rerun_recon,
            rerun_gs,
            depth_image,
            depth_list_state,
            depth_idx_state,
            depth_counter,
            normal_image,
            normal_list_state,
            normal_idx_state,
            normal_counter,
            status_box,
            dl_glb,
            dl_gs_ply,
        ],
    )

    # Depth navigation
    def depth_prev(img_list, idx):
        new_idx = max(0, idx - 1)
        ctr     = f"{new_idx + 1} / {len(img_list)}" if img_list else "–"
        return (img_list[new_idx] if img_list else None), new_idx, ctr

    def depth_next(img_list, idx):
        new_idx = min(len(img_list) - 1, idx + 1) if img_list else 0
        ctr     = f"{new_idx + 1} / {len(img_list)}" if img_list else "–"
        return (img_list[new_idx] if img_list else None), new_idx, ctr

    btn_depth_prev.click(
        depth_prev,
        inputs=[depth_list_state, depth_idx_state],
        outputs=[depth_image, depth_idx_state, depth_counter],
    )
    btn_depth_next.click(
        depth_next,
        inputs=[depth_list_state, depth_idx_state],
        outputs=[depth_image, depth_idx_state, depth_counter],
    )

    def normal_prev(img_list, idx):
        new_idx = max(0, idx - 1)
        ctr     = f"{new_idx + 1} / {len(img_list)}" if img_list else "–"
        return (img_list[new_idx] if img_list else None), new_idx, ctr

    def normal_next(img_list, idx):
        new_idx = min(len(img_list) - 1, idx + 1) if img_list else 0
        ctr     = f"{new_idx + 1} / {len(img_list)}" if img_list else "–"
        return (img_list[new_idx] if img_list else None), new_idx, ctr

    btn_normal_prev.click(
        normal_prev,
        inputs=[normal_list_state, normal_idx_state],
        outputs=[normal_image, normal_idx_state, normal_counter],
    )
    btn_normal_next.click(
        normal_next,
        inputs=[normal_list_state, normal_idx_state],
        outputs=[normal_image, normal_idx_state, normal_counter],
    )


if __name__ == "__main__":
    demo.launch(
        mcp_server=True,
        ssr_mode=False,
        show_error=True,
        allowed_paths=[str(OUTPUT_DIR), str(UPLOAD_DIR)],
    )