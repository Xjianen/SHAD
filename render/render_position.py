import os
import sys
import json
import math
import random
import bpy
import numpy as np
from argparse import ArgumentParser
from mathutils import Vector, Matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from utils.base_render import (
    calculate_bbox, create_camera, create_emission_material, full_cleanup,
    get_camera_extrinsics, get_camera_intrinsics, import_obj_file, join_objects,
    recalculate_normal, remove_file_if_exists, render_still, set_camera_pose,
    set_object_materials, set_transparent_background, setup_render_settings,
    setup_world_background, build_texture_library, count_mask_pixels,
)

MIN_TRANSLATION_RATIO = 0.05
MAX_TRANSLATION_RATIO = 0.10
MIN_MASK_RATIO = 0.05


def world_bbox(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_c = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_c = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return (min_c + max_c) / 2.0, max_c - min_c


def choose_texture_materials(texture_root, texture_categories, texture_nums):
    available = [c for c in texture_categories if os.path.isdir(os.path.join(texture_root, c))]
    if not available:
        raise FileNotFoundError(f"No texture categories found under {texture_root}")
    materials = build_texture_library(texture_root, random.choice(available), texture_nums)
    if not materials:
        raise FileNotFoundError(f"No texture images found under {texture_root}")
    return materials


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


def load_part_scene(objs_path, texture_root, texture_categories, texture_nums):
    obj_files = sorted([f for f in os.listdir(objs_path) if f.lower().endswith(".obj")])
    if not obj_files:
        return []
    materials = choose_texture_materials(texture_root, texture_categories, texture_nums)
    parts = []
    for obj_file in obj_files:
        material = random.choice(materials)
        imported = import_obj_file(os.path.join(objs_path, obj_file), material)
        part = join_objects(imported, os.path.splitext(obj_file)[0])
        if part is not None:
            force_regenerate_uvs(part)
            center, size = world_bbox(part)
            parts.append({
                "object": part, "source": os.path.join(objs_path, obj_file),
                "materials": list(part.data.materials), "center": center,
                "size": size,
            })
    return parts


def random_unit_vector():
    direction = Vector((random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)))
    while direction.length < 1e-6:
        direction = Vector((random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)))
    direction.normalize()
    return direction


def translate_part(part_obj, delta_world):
    part_obj.matrix_world = Matrix.Translation(delta_world) @ part_obj.matrix_world


# ---- 掩码渲染（带遮挡） ----

def render_mask_occlusion(path, white_objs, black_objs):
    white_mat = create_emission_material("Mw", (1, 1, 1, 1))
    black_mat = create_emission_material("Mb", (0, 0, 0, 1))
    stored = []
    for obj in white_objs:
        stored.append((obj, list(obj.data.materials)))
        obj.data.materials.clear(); obj.data.materials.append(white_mat)
        obj.hide_render = False
    for obj in black_objs:
        if any(obj is w for w in white_objs): continue
        stored.append((obj, list(obj.data.materials)))
        obj.data.materials.clear(); obj.data.materials.append(black_mat)
        obj.hide_render = False
    set_transparent_background(False)
    setup_world_background(0.0, (0, 0, 0, 1))
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.cycles.samples = 1
    render_still(path)
    for obj, mats in stored:
        obj.data.materials.clear()
        for mat in mats: obj.data.materials.append(mat)


def render_mask_all_white(path, all_objects):
    white_mat = create_emission_material("Mall", (1, 1, 1, 1))
    stored = [(obj, list(obj.data.materials)) for obj in all_objects]
    for obj in all_objects:
        obj.data.materials.clear(); obj.data.materials.append(white_mat)
        obj.hide_render = False
    set_transparent_background(False)
    setup_world_background(0.0, (0, 0, 0, 1))
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.cycles.samples = 1
    render_still(path)
    for obj, mats in stored:
        obj.data.materials.clear()
        for mat in mats: obj.data.materials.append(mat)


# ---- JSON ----

def save_json(json_path, camera, part_source, anomaly_type, radius, elevation, azimuth,
              delta=None, normal_mask_ratio=None, anomaly_mask_ratio=None):
    scene = bpy.context.scene
    rt, rt_ = get_camera_extrinsics(camera)
    data = {
        "K": get_camera_intrinsics(camera, scene), "RT": rt, "RT_": rt_,
        "anomaly_type": anomaly_type,
        "translated_part": os.path.basename(part_source),
        "radius": radius, "elevation_deg": math.degrees(elevation),
        "azimuth_deg": math.degrees(azimuth) % 360,
    }
    if delta:
        data["delta"] = [round(delta.x,6), round(delta.y,6), round(delta.z,6)]
        data["delta_length"] = round(math.sqrt(delta.x**2+delta.y**2+delta.z**2), 6)
    if normal_mask_ratio is not None:
        data["normal_mask_ratio"] = round(normal_mask_ratio, 6)
    if anomaly_mask_ratio is not None:
        data["anomaly_mask_ratio"] = round(anomaly_mask_ratio, 6)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ---- 场景构建 ----

def build_position_scene(parts):
    if not parts: return None
    target = random.choice(parts)
    return {"target": target, "all_parts": parts, "target_source": target["source"]}


# ---- 核心渲染 ----

def render_textured_query(image_path, camera, all_parts, radius, elevation, azimuth, samples):
    set_camera_pose(camera, [p["object"] for p in all_parts], radius, elevation, azimuth)
    for part in all_parts:
        part["object"].hide_render = False
        set_object_materials(part["object"], part["materials"])
    set_transparent_background(True)
    setup_world_background(1.0, (1, 1, 1, 1))
    setup_render_settings(bpy.context.scene.render.resolution_x,
                          bpy.context.scene.render.resolution_y, samples)
    render_still(image_path)


def render_position_pair(shape_id, output_dir, scene_data, camera,
                         radius, elevation, azimuth, final_samples):
    target_part = scene_data["target"]
    all_parts = scene_data["all_parts"]
    target_obj = target_part["object"]
    target_materials = list(target_obj.data.materials)
    all_objs = [p["object"] for p in all_parts]
    orig_matrix = target_obj.matrix_world.copy()

    # 平移参数
    center, size = world_bbox(target_obj)
    max_dim = max(size.x, max(size.y, size.z))
    direction = random_unit_vector()
    delta = direction * (max_dim * random.uniform(MIN_TRANSLATION_RATIO, MAX_TRANSLATION_RATIO))

    az_deg = int(round(math.degrees(azimuth))) % 360
    el_deg = int(round(math.degrees(elevation)))
    stem = f"render_{shape_id}_{radius}_{az_deg}_{el_deg}_Position"

    normal_img   = os.path.join(output_dir, stem + "_normal.png")
    normal_json  = os.path.join(output_dir, stem + "_normal.json")
    anomaly_img  = os.path.join(output_dir, stem + "_anomaly.png")
    anomaly_json = os.path.join(output_dir, stem + "_anomaly.json")
    normal_mask_path = os.path.join(output_dir, stem + "_normal_mask.png")
    anomaly_mask_path = os.path.join(output_dir, stem + "_anomaly_mask.png")
    complete_path = os.path.join(output_dir, stem + "_complete_mask_tmp.png")

    # Step 1: 正常纹理图
    render_textured_query(normal_img, camera, all_parts, radius, elevation, azimuth, final_samples)

    # Step 2: 完整物体mask + 平移前目标部件mask（带遮挡）
    other_objs = [p["object"] for p in all_parts if p["object"] != target_obj]
    render_mask_all_white(complete_path, all_objs)
    render_mask_occlusion(normal_mask_path, [target_obj], other_objs)

    complete_px = count_mask_pixels(complete_path)
    normal_px = count_mask_pixels(normal_mask_path)
    normal_ratio = normal_px / complete_px if complete_px > 0 else 0.0
    remove_file_if_exists(complete_path)

    if normal_ratio < MIN_MASK_RATIO:
        for p in [normal_mask_path, normal_img, normal_json]:
            remove_file_if_exists(p)
        set_object_materials(target_obj, target_materials)
        return False, normal_ratio

    # Step 3: 平移target
    translate_part(target_obj, delta)
    target_obj.hide_render = False

    # Step 4: 平移后目标部件mask（带遮挡）
    render_mask_all_white(complete_path, all_objs)
    render_mask_occlusion(anomaly_mask_path, [target_obj], other_objs)

    complete_px = count_mask_pixels(complete_path)
    anomaly_px = count_mask_pixels(anomaly_mask_path)
    anomaly_ratio = anomaly_px / complete_px if complete_px > 0 else 0.0
    remove_file_if_exists(complete_path)

    if anomaly_ratio < MIN_MASK_RATIO:
        for p in [anomaly_mask_path, normal_mask_path, normal_img, normal_json]:
            remove_file_if_exists(p)
        target_obj.matrix_world = orig_matrix
        set_object_materials(target_obj, target_materials)
        return False, anomaly_ratio

    # Step 5: 异常查询图
    set_object_materials(target_obj, target_materials)
    for part in all_parts: part["object"].hide_render = False
    set_transparent_background(True)
    setup_world_background(1.0, (1, 1, 1, 1))
    setup_render_settings(bpy.context.scene.render.resolution_x,
                          bpy.context.scene.render.resolution_y, final_samples)
    render_still(anomaly_img)
    save_json(normal_json, camera, scene_data["target_source"], "normal",
              radius, elevation, azimuth, normal_mask_ratio=normal_ratio)
    save_json(anomaly_json, camera, scene_data["target_source"], "translation",
              radius, elevation, azimuth, delta,
              normal_mask_ratio=normal_ratio, anomaly_mask_ratio=anomaly_ratio)
    return True, anomaly_ratio


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    parser = ArgumentParser(description="Render micro-translation anomaly images.")
    parser.add_argument("--root", type=str, default=default_root)
    parser.add_argument("--category", type=str, default="Mug")
    parser.add_argument("--num_views", type=int, default=10)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_shapes", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=100)
    parser.add_argument("--min_translation_ratio", type=float, default=MIN_TRANSLATION_RATIO)
    parser.add_argument("--max_translation_ratio", type=float, default=MAX_TRANSLATION_RATIO)
    parser.add_argument("--min_mask_ratio", type=float, default=MIN_MASK_RATIO)
    parser.add_argument("--texture_categories", nargs="+",
                        default=["pure_color", "fabric", "ceramic", "leather", "paper"])
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.seed is not None: random.seed(args.seed); np.random.seed(args.seed)
    global MIN_TRANSLATION_RATIO, MAX_TRANSLATION_RATIO, MIN_MASK_RATIO
    MIN_TRANSLATION_RATIO = args.min_translation_ratio; MAX_TRANSLATION_RATIO = args.max_translation_ratio
    MIN_MASK_RATIO = args.min_mask_ratio

    root = os.path.abspath(args.root)
    category_path = os.path.join(root, "shapes", args.category)
    texture_root = os.path.join(root, "textures")
    image_root = os.path.join(root, "Position", args.category)
    if not os.path.isdir(category_path): raise FileNotFoundError(f"{category_path} not found")
    os.makedirs(image_root, exist_ok=True)

    shapes = sorted([s for s in os.listdir(category_path) if s.startswith("shape_")])
    if args.max_shapes: shapes = shapes[:args.max_shapes]
    print(f"[INFO] Position | {args.category} | {len(shapes)} shapes | {args.num_views} views")

    for shape in shapes:
        shape_id = shape.split("_", 1)[1]
        objs_path = os.path.join(category_path, shape, "objs")
        if not os.path.isdir(objs_path): continue
        if not [f for f in os.listdir(objs_path) if f.lower().endswith(".obj")]: continue
        out_dir = os.path.join(image_root, shape); os.makedirs(out_dir, exist_ok=True)

        for view_idx in range(args.num_views):
            ok = False
            for attempt in range(args.max_attempts):
                full_cleanup()
                setup_render_settings(args.width, args.height, args.samples)
                camera = create_camera()
                parts = load_part_scene(objs_path, texture_root, args.texture_categories,
                                        random.randint(1, 5))
                scene_data = build_position_scene(parts)
                if scene_data is None: continue
                radius = round(random.uniform(3, 4.5), 1)
                elevation = random.uniform(math.pi / 9, 2 * math.pi / 9)
                azimuth = random.uniform(0, 2 * math.pi)
                ok, _ = render_position_pair(shape_id, out_dir, scene_data, camera,
                                             radius, elevation, azimuth, args.samples)
                if ok:
                    print(f"[OK] {shape} view={view_idx} part={os.path.basename(scene_data['target_source'])}")
                    break
            if not ok: print(f"[SKIP] {shape} view={view_idx}")

    full_cleanup()


if __name__ == "__main__":
    main()
