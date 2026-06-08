# coding=utf-8
"""
基于 base_render.py 改进的缺失异常流水线。
-----------------------------
改动:
1. 修复纹理纯色 → 强制重建 UV
2. 所有部件随机选一个缺失，掩码占比限制 ≤10%
3. 每轮场景搭3个相机(120°间隔)，减少迭代 + 避免遮挡
4. 输出正常图 + 异常图 + 掩码 + 相机参数
3. 每轮场景搭3个相机(120°间隔)，减少迭代 + 避免遮挡
4. 输出正常图 + 异常图 + 掩码 + 相机参数
"""

import os
import sys
import json
import math
import random
import bpy
import numpy as np
from argparse import ArgumentParser
from mathutils import Vector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from utils.base_render import (
    calculate_bbox,
    create_camera,
    create_emission_material,
    full_cleanup,
    get_camera_extrinsics,
    get_camera_intrinsics,
    import_obj_file,
    join_objects,
    recalculate_normal,
    remove_file_if_exists,
    render_still,
    set_camera_pose,
    set_object_materials,
    set_transparent_background,
    setup_render_settings,
    setup_world_background,
    build_texture_library,
    count_mask_pixels,
)

MIN_MASK_RATIO = 0.01
MAX_MASK_RATIO = 0.10
TOP_K_DEFAULT = 3

# ============================================================
# 辅助
# ============================================================

def world_bbox(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_c = Vector((min(v.x for v in corners),
                    min(v.y for v in corners),
                    min(v.z for v in corners)))
    max_c = Vector((max(v.x for v in corners),
                    max(v.y for v in corners),
                    max(v.z for v in corners)))
    return (min_c + max_c) / 2.0, max_c - min_c

def bbox_volume(obj):
    _, size = world_bbox(obj)
    return max(size.x, 1e-6) * max(size.y, 1e-6) * max(size.z, 1e-6)

def choose_texture_materials(texture_root, texture_categories, texture_nums):
    available = [c for c in texture_categories if os.path.isdir(os.path.join(texture_root, c))]
    if not available:
        raise FileNotFoundError(f"No texture categories found under {texture_root}")
    materials = build_texture_library(texture_root, random.choice(available), texture_nums)
    if not materials:
        raise FileNotFoundError(f"No texture images found under {texture_root}")
    return materials

# ============================================================
# UV 强制重建
# ============================================================

def force_regenerate_uvs(obj):
    mesh = obj.data
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    while mesh.uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers[0])
    mesh.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=45.0, island_margin=0.005)
    bpy.ops.object.mode_set(mode='OBJECT')

# ============================================================
# 部件加载（每个 OBJ 单独算体积+伸长率）
# ============================================================

def load_individual_parts(objs_path, texture_root, texture_categories, texture_nums):
    obj_files = sorted([f for f in os.listdir(objs_path) if f.lower().endswith(".obj")])
    if len(obj_files) < 2:
        return []

    materials = choose_texture_materials(texture_root, texture_categories, texture_nums)
    parts = []
    for obj_file in obj_files:
        material = random.choice(materials)
        imported = import_obj_file(os.path.join(objs_path, obj_file), material)
        part = join_objects(imported, os.path.splitext(obj_file)[0])
        if part is not None:
            force_regenerate_uvs(part)
            vol = bbox_volume(part)
            _, size = world_bbox(part)
            min_dim = max(min(size.x, min(size.y, size.z)), 1e-6)
            max_dim = max(size.x, max(size.y, size.z))
            elongation = max_dim / min_dim
            score = elongation / (vol + 1e-8)
            parts.append({
                "object": part,
                "source": os.path.join(objs_path, obj_file),
                "materials": list(part.data.materials),
                "volume": vol,
                "elongation": elongation,
                "score": score,
            })
    return parts

# ============================================================
# 场景构建：选缺失部件
# ============================================================

def build_missing_scene(parts):
    """随机选一个部件缺失，逐个尝试直到找到能用的。"""
    if len(parts) < 2:
        return None

    shuffled = list(parts)
    random.shuffle(shuffled)

    for target in shuffled:
        missing_obj = target["object"]
        missing_source = target["source"]

        visible_list = [p["object"] for p in parts if p["source"] != missing_source]
        if not visible_list:
            continue

        visible_obj = join_objects(list(visible_list), "Missing_visible")
        if visible_obj is None:
            continue

        all_meshes = [visible_obj, missing_obj]

        return {
            "visible_obj": visible_obj,
            "missing_obj": missing_obj,
            "all_meshes": all_meshes,
            "missing_source": missing_source,
            "score": target.get("score", 0),
            "volume": target.get("volume", 0),
            "elongation": target.get("elongation", 0),
        }

    return None

# ============================================================
# 掩码渲染（直接复用原始 render_missing_mug 的 render_mask 模式）
# ============================================================

def render_mask(path, visible_obj, missing_obj, mode):
    """缺失掩码：missing=白 visible=黑  /  完整掩码：都白"""
    white = create_emission_material("Mask_white", (1, 1, 1, 1))
    black = create_emission_material("Mask_black", (0, 0, 0, 1))

    if mode == "missing":
        visible_obj.data.materials.clear()
        visible_obj.data.materials.append(black)
        missing_obj.data.materials.clear()
        missing_obj.data.materials.append(white)
        visible_obj.hide_render = False
        missing_obj.hide_render = False
    elif mode == "complete":
        visible_obj.data.materials.clear()
        visible_obj.data.materials.append(white)
        missing_obj.data.materials.clear()
        missing_obj.data.materials.append(white)
        visible_obj.hide_render = False
        missing_obj.hide_render = False

    set_transparent_background(False)
    setup_world_background(0.0, (0, 0, 0, 1))
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.cycles.samples = 1
    render_still(path)

# ============================================================
# JSON 保存
# ============================================================

def save_camera_json(json_path, camera, missing_source, radius, elevation, azimuth,
                     anomaly_type, score=0, volume=0, elongation=0, mask_ratio=0):
    scene = bpy.context.scene
    rt, rt_ = get_camera_extrinsics(camera)
    data = {
        "K": get_camera_intrinsics(camera, scene),
        "RT": rt,
        "RT_": rt_,
        "anomaly_type": anomaly_type,
        "missing_part": os.path.basename(missing_source),
        "score": round(score, 4),
        "volume": round(volume, 6),
        "elongation": round(elongation, 4),
        "mask_ratio": round(mask_ratio, 6) if mask_ratio else None,
        "radius": radius,
        "elevation_deg": math.degrees(elevation),
        "azimuth_deg": math.degrees(azimuth) % 360,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# ============================================================
# 核心渲染（正常 + 异常 + 掩码）
# ============================================================

def render_missing_small_pair(shape_id, output_dir, scene_data, camera,
                              radius, elevation, azimuth, final_samples):
    visible_obj = scene_data["visible_obj"]
    missing_obj = scene_data["missing_obj"]
    all_meshes = scene_data["all_meshes"]
    missing_source = scene_data["missing_source"]

    # 文件名
    az_deg = int(round(math.degrees(azimuth))) % 360
    el_deg = int(round(math.degrees(elevation)))
    stem = f"render_{shape_id}_{radius}_{az_deg}_{el_deg}_MissingSmall"

    normal_img = os.path.join(output_dir, stem + "_normal.png")
    normal_json = os.path.join(output_dir, stem + "_normal.json")
    anomaly_img = os.path.join(output_dir, stem + "_anomaly.png")
    anomaly_json = os.path.join(output_dir, stem + "_anomaly.json")
    mask_path = os.path.join(output_dir, stem + "_mask.png")
    complete_mask_path = os.path.join(output_dir, stem + "_complete_mask_tmp.png")

    # 保存原始材质
    visible_materials = list(visible_obj.data.materials)
    missing_materials = list(missing_obj.data.materials)

    # ---- Step 1: 渲染掩码检查比例 ----
    set_camera_pose(camera, all_meshes, radius, elevation, azimuth)

    render_mask(mask_path, visible_obj, missing_obj, "missing")
    render_mask(complete_mask_path, visible_obj, missing_obj, "complete")

    missing_pixels = count_mask_pixels(mask_path)
    complete_pixels = count_mask_pixels(complete_mask_path)
    ratio = missing_pixels / complete_pixels if complete_pixels > 0 else 0.0
    remove_file_if_exists(complete_mask_path)

    if ratio < MIN_MASK_RATIO or ratio > MAX_MASK_RATIO:
        remove_file_if_exists(mask_path)
        set_object_materials(visible_obj, visible_materials)
        set_object_materials(missing_obj, missing_materials)
        return False, ratio

    # ---- Step 2: 渲染正常图像（所有部件可见） ----
    set_object_materials(visible_obj, visible_materials)
    set_object_materials(missing_obj, missing_materials)
    visible_obj.hide_render = False
    missing_obj.hide_render = False

    set_transparent_background(True)
    setup_world_background(1.0, (1, 1, 1, 1))
    setup_render_settings(
        bpy.context.scene.render.resolution_x,
        bpy.context.scene.render.resolution_y,
        final_samples,
    )
    render_still(normal_img)
    save_camera_json(normal_json, camera, missing_source,
                     radius, elevation, azimuth, "normal",
                     scene_data["score"], scene_data["volume"],
                     scene_data["elongation"])

    # ---- Step 3: 渲染异常图像（缺失部件隐藏） ----
    missing_obj.hide_render = True
    render_still(anomaly_img)
    save_camera_json(anomaly_json, camera, missing_source,
                     radius, elevation, azimuth, "missing_small",
                     scene_data["score"], scene_data["volume"],
                     scene_data["elongation"], ratio)

    return True, ratio

# ============================================================
# 参数解析 & 主循环
# ============================================================

def parse_args():
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]

    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    parser = ArgumentParser(description="Render missing-small-part anomaly images.")
    parser.add_argument("--root", type=str, default=default_root)
    parser.add_argument("--category", type=str, default="Mug")
    parser.add_argument("--num_views", type=int, default=10)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_shapes", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=100)

    parser.add_argument("--min_mask_ratio", type=float, default=MIN_MASK_RATIO)
    parser.add_argument("--max_mask_ratio", type=float, default=MAX_MASK_RATIO)
    parser.add_argument("--texture_categories", nargs="+",
                        default=["pure_color", "fabric", "ceramic", "leather", "paper"])
    return parser.parse_args(argv)

def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    global MIN_MASK_RATIO, MAX_MASK_RATIO
    MIN_MASK_RATIO = args.min_mask_ratio
    MAX_MASK_RATIO = args.max_mask_ratio

    root = os.path.abspath(args.root)
    category_path = os.path.join(root, "shapes", args.category)
    texture_root = os.path.join(root, "textures")
    image_root = os.path.join(root, "Missing", args.category)

    if not os.path.isdir(category_path):
        raise FileNotFoundError(f"{category_path} not found")
    os.makedirs(image_root, exist_ok=True)

    shapes = sorted([s for s in os.listdir(category_path) if s.startswith("shape_")])
    if args.max_shapes is not None:
        shapes = shapes[:args.max_shapes]
    print(f"[INFO] MissingSmall | category={args.category} | shapes={len(shapes)} | "
          f"views_per_shape={args.num_views}")

    for shape in shapes:
        shape_id = shape.split("_", 1)[1]
        objs_path = os.path.join(category_path, shape, "objs")
        if not os.path.isdir(objs_path):
            print(f"[WARN] missing objs: {os.path.join(category_path, shape)}")
            continue
        if len([f for f in os.listdir(objs_path) if f.lower().endswith(".obj")]) < 2:
            print(f"[SKIP] only one part: {shape}")
            continue

        image_shape_path = os.path.join(image_root, shape)
        os.makedirs(image_shape_path, exist_ok=True)

        # ---- 每 shape 产出 num_views 张，每轮搭3相机 ----
        views_done = 0

        for attempt in range(args.max_attempts):
            if views_done >= args.num_views:
                break

            full_cleanup()
            setup_render_settings(args.width, args.height, args.samples)

            texture_nums = random.randint(1, 5)
            parts = load_individual_parts(
                objs_path, texture_root, args.texture_categories, texture_nums,
            )
            scene_data = build_missing_scene(parts)
            if scene_data is None:
                continue

            radius = round(random.uniform(3, 4.5), 1)
            elevation = random.uniform(math.pi / 9, 2 * math.pi / 9)
            base_az = random.uniform(0, 2 * math.pi)

            for offset in [0, 2 * math.pi / 3, 4 * math.pi / 3]:
                if views_done >= args.num_views:
                    break

                # 重置状态
                scene_data["missing_obj"].hide_render = False
                scene_data["visible_obj"].hide_render = False

                camera = create_camera()
                az = (base_az + offset) % (2 * math.pi)

                ok, ratio = render_missing_small_pair(
                    shape_id, image_shape_path, scene_data, camera,
                    radius, elevation, az, args.samples,
                )
                if ok:
                    print(f"[OK] {shape} view={views_done} "
                          f"missing={os.path.basename(scene_data['missing_source'])} "
                          f"score={scene_data['score']:.2f} "
                          f"az={math.degrees(az):.0f} ratio={ratio:.4f}")
                    views_done += 1

        if views_done < args.num_views:
            print(f"[SKIP] {shape} only got {views_done}/{args.num_views} views")

    full_cleanup()

if __name__ == "__main__":
    main()
