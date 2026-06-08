# coding=utf-8
"""
render_broken_end_part.py
-------------------------
基于 BMesh 顶点删除的"端部破损"异常生成流水线。

与原始 render_damage_part.py 的区别：
1. 不用布尔运算 → 改用 BMesh 直接删顶点，零残留
2. 不选最大部件 → 按伸长率选最细长部件（如把手），符合工业微小缺陷实际
3. 不在中间切断 → 只删端点区域，不会产生浮空片段
4. 同时输出正常纹理图和异常纹理图，方便对比
"""

import os
import sys
import json
import math
import random
import bpy
import bmesh
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

# ---- 可调参数 ----
MIN_ELONGATION_RATIO = 1.5       # 部件伸长率阈值（max_dim/min_dim）
MIN_BREAK_RADIUS_RATIO = 0.06   # 删除球半径下限（相对部件max_dim）
MAX_BREAK_RADIUS_RATIO = 0.20   # 删除球半径上限
MIN_MASK_RATIO = 0.003          # 掩码占比下限（端点删除通常很小）
MAX_MASK_RATIO = 0.20           # 掩码占比上限


# ============================================================
# 辅助函数
# ============================================================

def world_bbox(obj):
    """单对象世界空间包围盒，返回 (center, size)。"""
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_c = Vector((min(v.x for v in corners),
                    min(v.y for v in corners),
                    min(v.z for v in corners)))
    max_c = Vector((max(v.x for v in corners),
                    max(v.y for v in corners),
                    max(v.z for v in corners)))
    return (min_c + max_c) / 2.0, max_c - min_c


def choose_texture_materials(texture_root, texture_categories, texture_nums):
    available = [c for c in texture_categories if os.path.isdir(os.path.join(texture_root, c))]
    if not available:
        raise FileNotFoundError(f"No texture categories found under {texture_root}")
    materials = build_texture_library(texture_root, random.choice(available), texture_nums)
    if not materials:
        raise FileNotFoundError(f"No texture images found under {texture_root}")
    return materials


# ============================================================
# 部件加载（含伸长率计算）
# ============================================================


def force_regenerate_uvs(obj):
    """强制删除所有现有UV层，重新用 smart_project 生成。
    
    解决 Blender 4.x OBJ 导入器在无 vt 数据时仍可能创建空 UV 层的问题。
    空 UV 层会导致所有顶点采样纹理的 (0,0) 坐标 → 纯色渲染。
    """
    mesh = obj.data
    
    # 切换到 OBJECT 模式
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # 删除所有现有 UV 层
    while mesh.uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers[0])
    
    # 创建新的 UV 层
    mesh.uv_layers.new(name="UVMap")
    
    # 进入 EDIT 模式，执行 smart_project
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=45.0, island_margin=0.005)
    bpy.ops.object.mode_set(mode='OBJECT')

def load_part_scene(objs_path, texture_root, texture_categories, texture_nums):
    """加载所有 OBJ 部件，计算每部件的 world_bbox 和伸长率。"""
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
            min_dim = max(min(size.x, min(size.y, size.z)), 1e-6)
            max_dim = max(size.x, max(size.y, size.z))
            elongation = max_dim / min_dim
            parts.append({
                "object": part,
                "source": os.path.join(objs_path, obj_file),
                "materials": list(part.data.materials),
                "center": center,
                "size": size,
                "elongation": elongation,
                "max_dim": max_dim,
            })
    return parts


# ============================================================
# 端点区域顶点定位
# ============================================================

def get_endpoint_region_verts(obj, center, size):
    """
    在细长部件的一端（沿最长轴正方向）选取球形邻域内的顶点索引。

    返回:
        vert_indices:  球形邻域内的顶点索引列表
        end_center:    端点球心（世界坐标）
        radius:        删除球半径
    """
    # 确定最长轴
    longest_axis = 0
    if size.y > size.x and size.y >= size.z:
        longest_axis = 1
    if size.z > size.x and size.z > size.y:
        longest_axis = 2

    max_dim = max(size.x, max(size.y, size.z))

    # 端点球心：沿最长轴正方向偏移到bbox端面中心
    end_center = center.copy()
    end_center[longest_axis] = center[longest_axis] + size[longest_axis] * 0.5

    # 删除球半径：相对部件最大尺寸的 6%-20%
    radius = max_dim * random.uniform(MIN_BREAK_RADIUS_RATIO, MAX_BREAK_RADIUS_RATIO)

    # 收集球内的顶点
    mesh = obj.data
    verts_in_region = []
    for i, vert in enumerate(mesh.vertices):
        world_co = obj.matrix_world @ vert.co
        if (world_co - end_center).length <= radius:
            verts_in_region.append(i)

    return verts_in_region, end_center, radius


# ============================================================
# BMesh 顶点操作
# ============================================================

def delete_verts_from_obj(obj, vert_indices):
    """从对象中删除指定索引的顶点（BMesh）。"""
    if not vert_indices:
        return False

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    max_idx = len(bm.verts)
    verts_to_delete = [bm.verts[i] for i in vert_indices if i < max_idx]
    if not verts_to_delete:
        bm.free()
        return False

    bmesh.ops.delete(bm, geom=verts_to_delete, context="VERTS")

    # 清理孤立的边/面（delete VERTS 会自动处理，但再确认一下）
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    # 删除后重新计算法线
    if len(obj.data.polygons) > 0:
        recalculate_normal(obj)
    return len(obj.data.polygons) > 0


def keep_only_verts(obj, vert_indices):
    """只保留指定索引的顶点，删除其余所有顶点。"""
    if not vert_indices:
        return False

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    keep_set = set(vert_indices)
    max_idx = len(bm.verts)
    verts_to_delete = [bm.verts[i] for i in range(max_idx) if i not in keep_set]

    if not verts_to_delete:
        bm.free()
        return True  # 没有要删的，说明全保留了

    bmesh.ops.delete(bm, geom=verts_to_delete, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return len(obj.data.polygons) > 0


def duplicate_object(obj, name):
    """深拷贝一个 mesh 对象。"""
    mesh_copy = obj.data.copy()
    obj_copy = bpy.data.objects.new(name, mesh_copy)
    bpy.context.collection.objects.link(obj_copy)
    obj_copy.matrix_world = obj.matrix_world.copy()
    for mat in obj.data.materials:
        obj_copy.data.materials.append(mat)
    return obj_copy


# ============================================================
# 场景构建
# ============================================================

def build_broken_end_scene(parts, top_k=3):
    """
    从伸长率最高的前 top_k 个细长部件中随机选一个作为破损目标，
    定位其一端端点区域。

    返回场景数据 dict，如果找不到合适的部件则返回 None。
    """
    if len(parts) < 2:
        return None

    # 按伸长率降序排列，取前 top_k 个，然后 shuffle
    candidates = sorted(parts, key=lambda p: p["elongation"], reverse=True)
    candidates = candidates[:min(top_k, len(candidates))]
    random.shuffle(candidates)

    # 逐个尝试 shuffled 候选，确保不同视图能命中不同部件
    for target in candidates:
        target_obj = target["object"]
        vert_indices, end_center, radius = get_endpoint_region_verts(
            target_obj, target["center"], target["size"]
        )

        if len(vert_indices) < 3:
            continue  # 这个部件端点区域太小，试下一个

        return {
            "target": target,
            "all_parts": parts,
            "target_source": target["source"],
            "end_center": end_center,
            "break_radius": radius,
            "vert_indices": vert_indices,
            "elongation": target["elongation"],
        }

    # 所有候选都失败
    return None


# ============================================================
# 渲染辅助
# ============================================================

def render_normal_query(image_path, json_path, camera, all_parts, radius, elevation, azimuth, samples):
    """渲染正常（无破损）的带纹理查询图像。"""
    set_camera_pose(camera, [p["object"] for p in all_parts], radius, elevation, azimuth)

    for part in all_parts:
        part["object"].hide_render = False
        set_object_materials(part["object"], part["materials"])

    set_transparent_background(True)
    setup_world_background(1.0, (1, 1, 1, 1))
    setup_render_settings(
        bpy.context.scene.render.resolution_x,
        bpy.context.scene.render.resolution_y,
        samples,
    )
    render_still(image_path)

    scene = bpy.context.scene
    rt, rt_ = get_camera_extrinsics(camera)
    data = {
        "K": get_camera_intrinsics(camera, scene),
        "RT": rt,
        "RT_": rt_,
        "anomaly_type": "normal",
        "radius": radius,
        "elevation_deg": math.degrees(elevation),
        "azimuth_deg": math.degrees(azimuth) % 360,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def render_occlusion_mask(path, white_obj, black_objs):
    """渲染带遮挡的掩码。
    white_obj  → 白色发光（破损区域）
    black_objs → 黑色发光（其余所有部件，含破损后的目标）
    所有对象均可见，黑色部件会自然遮挡白色区域，确保掩码只含相机可见的破损像素。
    """
    white_mat = create_emission_material("MaskWhite", (1, 1, 1, 1))
    black_mat = create_emission_material("MaskBlack", (0, 0, 0, 1))

    white_orig = list(white_obj.data.materials)
    white_obj.data.materials.clear()
    white_obj.data.materials.append(white_mat)
    white_obj.hide_render = False

    black_stored = []
    for obj in black_objs:
        if obj == white_obj:
            continue
        black_stored.append((obj, list(obj.data.materials)))
        obj.data.materials.clear()
        obj.data.materials.append(black_mat)
        obj.hide_render = False

    set_transparent_background(False)
    setup_world_background(0.0, (0, 0, 0, 1))
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.cycles.samples = 1
    render_still(path)

    white_obj.data.materials.clear()
    for mat in white_orig:
        white_obj.data.materials.append(mat)
    for obj, mats in black_stored:
        obj.data.materials.clear()
        for mat in mats:
            obj.data.materials.append(mat)


def render_complete_mask(path, all_objects):
    """所有对象渲染为白色发光（完整掩码，含破损后的目标）。"""
    white_mat = create_emission_material("MaskComplete", (1, 1, 1, 1))
    stored = []
    for obj in all_objects:
        stored.append((obj, list(obj.data.materials)))
        obj.data.materials.clear()
        obj.data.materials.append(white_mat)
        obj.hide_render = False

    set_transparent_background(False)
    setup_world_background(0.0, (0, 0, 0, 1))
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.cycles.samples = 1
    render_still(path)

    for obj, mats in stored:
        obj.data.materials.clear()
        for mat in mats:
            obj.data.materials.append(mat)
def save_broken_json(json_path, camera, part_source, end_center, break_radius,
                     radius, elevation, azimuth, mask_ratio):
    """保存异常相机参数 + 破损信息到 JSON。"""
    scene = bpy.context.scene
    rt, rt_ = get_camera_extrinsics(camera)
    data = {
        "K": get_camera_intrinsics(camera, scene),
        "RT": rt,
        "RT_": rt_,
        "anomaly_type": "broken_end",
        "broken_part": os.path.basename(part_source),
        "break_center": [round(end_center.x, 6),
                         round(end_center.y, 6),
                         round(end_center.z, 6)],
        "break_radius": round(break_radius, 6),
        "mask_ratio": round(mask_ratio, 6),
        "radius": radius,
        "elevation_deg": math.degrees(elevation),
        "azimuth_deg": math.degrees(azimuth) % 360,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ============================================================
# 核心：渲染一对（正常 + 异常 + 掩码）
# ============================================================

def render_broken_end_pair(shape_id, output_dir, scene_data, camera,
                           radius, elevation, azimuth,
                           min_ratio, max_ratio, final_samples):
    """
    渲染一个视图的完整输出：
      - _normal.png  : 破损前的正常纹理图
      - _normal.json : 正常相机参数
      - _anomaly.png : 破损后的异常纹理图
      - _anomaly.json: 异常相机参数 + 破损信息
      - _mask.png    : 标注掩码（破损区域 = 白色）
    """
    target_part = scene_data["target"]
    all_parts = scene_data["all_parts"]
    target_obj = target_part["object"]
    target_materials = list(target_obj.data.materials)
    vert_indices = scene_data["vert_indices"]
    all_objs = [p["object"] for p in all_parts]

    az_deg = int(round(math.degrees(azimuth))) % 360
    el_deg = int(round(math.degrees(elevation)))
    stem = f"render_{shape_id}_{radius}_{az_deg}_{el_deg}_BrokenEnd"

    normal_img = os.path.join(output_dir, stem + "_normal.png")
    normal_json = os.path.join(output_dir, stem + "_normal.json")
    anomaly_img = os.path.join(output_dir, stem + "_anomaly.png")
    anomaly_json = os.path.join(output_dir, stem + "_anomaly.json")
    mask_path = os.path.join(output_dir, stem + "_mask.png")
    complete_mask_path = os.path.join(output_dir, stem + "_complete_mask_tmp.png")

    # ---- Step 1: 渲染正常图像 ----
    render_normal_query(
        normal_img, normal_json, camera, all_parts,
        radius, elevation, azimuth, final_samples,
    )

    # ---- Step 2: 创建 ghost（只保留端点区域，用于掩码） ----
    ghost_obj = duplicate_object(target_obj, "BrokenEnd_ghost")
    ok_ghost = keep_only_verts(ghost_obj, vert_indices)
    if not ok_ghost:
        # ghost 空了 → 无法生成掩码，清理并返回
        remove_file_if_exists(normal_img)
        remove_file_if_exists(normal_json)
        bpy.data.objects.remove(ghost_obj, do_unlink=True)
        return False, 0.0

    # ---- Step 3: 对原目标删除端点区域顶点 ----
    ok_delete = delete_verts_from_obj(target_obj, vert_indices)
    if not ok_delete:
        # 删除后目标空了 → 删除太多了，清理并返回
        remove_file_if_exists(normal_img)
        remove_file_if_exists(normal_json)
        target_obj.matrix_world = ghost_obj.matrix_world.copy()  # 恢复
        bpy.data.objects.remove(ghost_obj, do_unlink=True)
        return False, 0.0

    # ---- Step 4: 渲染掩码检查比例 ----
    render_complete_mask(complete_mask_path, all_objs)
    render_occlusion_mask(mask_path, ghost_obj, all_objs)

    anomaly_pixels = count_mask_pixels(mask_path)
    complete_pixels = count_mask_pixels(complete_mask_path)
    ratio = anomaly_pixels / complete_pixels if complete_pixels > 0 else 0.0
    remove_file_if_exists(complete_mask_path)

    if ratio < min_ratio or ratio > max_ratio:
        # 比例不合法，清理所有输出
        remove_file_if_exists(mask_path)
        remove_file_if_exists(normal_img)
        remove_file_if_exists(normal_json)
        bpy.data.objects.remove(ghost_obj, do_unlink=True)
        # 注意：target_obj 已经损坏了，无法恢复。
        # 因为这是 attempt 循环内的，外层会 full_cleanup 重建场景。
        return False, ratio

    # ---- Step 5: 渲染异常查询图像 ----
    bpy.data.objects.remove(ghost_obj, do_unlink=True)
    set_object_materials(target_obj, target_materials)
    for part in all_parts:
        part["object"].hide_render = False

    set_transparent_background(True)
    setup_world_background(1.0, (1, 1, 1, 1))
    setup_render_settings(
        bpy.context.scene.render.resolution_x,
        bpy.context.scene.render.resolution_y,
        final_samples,
    )
    render_still(anomaly_img)

    save_broken_json(
        anomaly_json, camera, scene_data["target_source"],
        scene_data["end_center"], scene_data["break_radius"],
        radius, elevation, azimuth, ratio,
    )

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
    parser = ArgumentParser(description="Render broken-end (端点破损) anomaly images and masks.")
    parser.add_argument("--root", type=str, default=default_root)
    parser.add_argument("--category", type=str, default="Mug")
    parser.add_argument("--num_views", type=int, default=10)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_shapes", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_attempts", type=int, default=50)
    parser.add_argument("--min_elongation", type=float, default=MIN_ELONGATION_RATIO)
    parser.add_argument("--min_break_radius_ratio", type=float, default=MIN_BREAK_RADIUS_RATIO)
    parser.add_argument("--max_break_radius_ratio", type=float, default=MAX_BREAK_RADIUS_RATIO)
    parser.add_argument("--min_mask_ratio", type=float, default=MIN_MASK_RATIO)
    parser.add_argument("--max_mask_ratio", type=float, default=MAX_MASK_RATIO)
    parser.add_argument("--texture_categories", nargs="+",
                        default=["pure_color", "fabric", "wood", "paper", "leather"])
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # 应用命令行参数覆盖全局默认值
    global MIN_ELONGATION_RATIO, MIN_BREAK_RADIUS_RATIO, MAX_BREAK_RADIUS_RATIO
    global MIN_MASK_RATIO, MAX_MASK_RATIO
    MIN_ELONGATION_RATIO = args.min_elongation
    MIN_BREAK_RADIUS_RATIO = args.min_break_radius_ratio
    MAX_BREAK_RADIUS_RATIO = args.max_break_radius_ratio
    MIN_MASK_RATIO = args.min_mask_ratio
    MAX_MASK_RATIO = args.max_mask_ratio

    root = os.path.abspath(args.root)
    category_path = os.path.join(root, "shapes", args.category)
    texture_root = os.path.join(root, "textures")
    image_root = os.path.join(root, "Broken", args.category)

    if not os.path.isdir(category_path):
        raise FileNotFoundError(f"{category_path} not found")
    os.makedirs(image_root, exist_ok=True)

    shapes = sorted([s for s in os.listdir(category_path) if s.startswith("shape_")])
    if args.max_shapes is not None:
        shapes = shapes[:args.max_shapes]
    print(f"[INFO] BrokenEnd | category={args.category} | shapes={len(shapes)} | "
          f"views_per_shape={args.num_views}")

    for shape in shapes:
        shape_id = shape.split("_", 1)[1]
        objs_path = os.path.join(category_path, shape, "objs")
        if not os.path.isdir(objs_path):
            print(f"[WARN] missing objs: {os.path.join(category_path, shape)}")
            continue

        obj_files = [f for f in os.listdir(objs_path) if f.lower().endswith(".obj")]
        if len(obj_files) < 2:
            print(f"[SKIP] only one part: {shape}")
            continue

        image_shape_path = os.path.join(image_root, shape)
        os.makedirs(image_shape_path, exist_ok=True)

        for view_idx in range(args.num_views):
            ok = False
            last_ratio = 0.0

            for attempt in range(args.max_attempts):
                full_cleanup()
                setup_render_settings(args.width, args.height, args.samples)
                camera = create_camera()

                texture_nums = random.randint(1, 5)
                parts = load_part_scene(
                    objs_path, texture_root, args.texture_categories, texture_nums,
                )
                scene_data = build_broken_end_scene(parts, args.top_k)
                if scene_data is None:
                    last_ratio = 0.0
                    continue

                radius = round(random.uniform(3, 4.5), 1)
                elevation = random.uniform(math.pi / 9, 2 * math.pi / 9)
                azimuth = random.uniform(0, 2 * math.pi)

                ok, last_ratio = render_broken_end_pair(
                    shape_id, image_shape_path, scene_data, camera,
                    radius, elevation, azimuth,
                    args.min_mask_ratio, args.max_mask_ratio,
                    args.samples,
                )
                if ok:
                    print(
                        f"[OK] {shape} view={view_idx} "
                        f"part={os.path.basename(scene_data['target_source'])} "
                        f"elong={scene_data['elongation']:.1f} "
                        f"radius={scene_data['break_radius']:.3f} "
                        f"ratio={last_ratio:.4f}"
                    )
                    break

            if not ok:
                print(f"[SKIP] no valid broken-end: {shape} view={view_idx} "
                      f"last_ratio={last_ratio:.4f}")

    full_cleanup()


if __name__ == "__main__":
    main()
