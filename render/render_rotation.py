# coding=utf-8
import os, sys, json, math, random
import bpy, numpy as np
from argparse import ArgumentParser
from mathutils import Vector, Matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path: sys.path.append(SCRIPT_DIR)

from utils.base_render import (
    calculate_bbox, create_camera, create_emission_material, full_cleanup,
    get_camera_extrinsics, get_camera_intrinsics, import_obj_file, join_objects,
    recalculate_normal, remove_file_if_exists, render_still, set_camera_pose,
    set_object_materials, set_transparent_background, setup_render_settings,
    setup_world_background, build_texture_library, count_mask_pixels,
)

MIN_ROTATION_DEG = 5.0
MAX_ROTATION_DEG = 20.0
MIN_MASK_RATIO = 0.05


def world_bbox(obj):
    cs = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(v.x for v in cs), min(v.y for v in cs), min(v.z for v in cs)))
    hi = Vector((max(v.x for v in cs), max(v.y for v in cs), max(v.z for v in cs)))
    return (lo + hi) / 2.0, hi - lo


def choose_texture_materials(tr, tc, tn):
    av = [c for c in tc if os.path.isdir(os.path.join(tr, c))]
    if not av: raise FileNotFoundError(f"No texture categories under {tr}")
    ms = build_texture_library(tr, random.choice(av), tn)
    if not ms: raise FileNotFoundError(f"No textures under {tr}")
    return ms


def force_regenerate_uvs(obj):
    m = obj.data
    if bpy.context.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True); bpy.context.view_layer.objects.active = obj
    while m.uv_layers: m.uv_layers.remove(m.uv_layers[0])
    m.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=45.0, island_margin=0.005)
    bpy.ops.object.mode_set(mode='OBJECT')


def load_part_scene(objs_path, tr, tc, tn):
    fs = sorted([f for f in os.listdir(objs_path) if f.lower().endswith(".obj")])
    if not fs: return []
    ms = choose_texture_materials(tr, tc, tn)
    parts = []
    for f in fs:
        mat = random.choice(ms)
        im = import_obj_file(os.path.join(objs_path, f), mat)
        p = join_objects(im, os.path.splitext(f)[0])
        if p is not None:
            force_regenerate_uvs(p)
            c, s = world_bbox(p)
            parts.append({"object": p, "source": os.path.join(objs_path, f),
                          "materials": list(p.data.materials), "center": c,
                          "size": s})
    return parts


def choose_bbox_corner_pivot(obj):
    return obj.matrix_world @ Vector(random.choice(obj.bound_box))


def random_unit_vector():
    axis = Vector((random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)))
    while axis.length < 1e-6:
        axis = Vector((random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)))
    axis.normalize()
    return axis


def rotate_part_around_pivot(obj, pivot, angle_deg, axis):
    T = Matrix.Translation
    obj.matrix_world = T(pivot) @ Matrix.Rotation(math.radians(angle_deg), 4, axis) @ T(-pivot) @ obj.matrix_world


# ============================================================
# 掩码渲染（参照 render_missing_mug.py：物体保持可见，黑/白材质实现遮挡）
# ============================================================

def render_occlusion_mask(path, white_objs, black_objs):
    """white_objs=白色发光, black_objs=黑色发光, 全部可见 → 黑色自然遮挡白色"""
    wm = create_emission_material("_mw", (1, 1, 1, 1))
    bm = create_emission_material("_mb", (0, 0, 0, 1))

    white_set = set(id(o) for o in white_objs)
    stored = []
    for o in white_objs:
        stored.append((o, list(o.data.materials)))
        o.data.materials.clear(); o.data.materials.append(wm); o.hide_render = False
    for o in black_objs:
        if id(o) in white_set: continue
        stored.append((o, list(o.data.materials)))
        o.data.materials.clear(); o.data.materials.append(bm); o.hide_render = False

    set_transparent_background(False); setup_world_background(0.0, (0, 0, 0, 1))
    bpy.context.scene.cycles.use_denoising = False; bpy.context.scene.cycles.samples = 1
    render_still(path)

    for o, ms in stored:
        o.data.materials.clear()
        for m in ms: o.data.materials.append(m)


def render_complete_mask(path, all_objs):
    """全部=白色发光, 都可见（完整掩码）"""
    render_occlusion_mask(path, all_objs, [])


# ---- JSON ----

def save_json(jp, cam, psrc, atype, r, el, az, pivot=None, ang=None, axis=None,
              normal_mask_ratio=None, anomaly_mask_ratio=None):
    s = bpy.context.scene; rt, rt_ = get_camera_extrinsics(cam)
    d = {"K": get_camera_intrinsics(cam, s), "RT": rt, "RT_": rt_,
         "anomaly_type": atype, "rotated_part": os.path.basename(psrc),
         "radius": r, "elevation_deg": math.degrees(el), "azimuth_deg": math.degrees(az) % 360}
    if pivot: d["pivot_world"] = [round(pivot.x, 6), round(pivot.y, 6), round(pivot.z, 6)]
    if ang is not None: d["angle_deg"] = round(ang, 3)
    if axis: d["axis"] = [round(axis.x, 6), round(axis.y, 6), round(axis.z, 6)]
    if normal_mask_ratio is not None: d["normal_mask_ratio"] = round(normal_mask_ratio, 6)
    if anomaly_mask_ratio is not None: d["anomaly_mask_ratio"] = round(anomaly_mask_ratio, 6)
    with open(jp, "w", encoding="utf-8") as f: json.dump(d, f, indent=4)


# ---- 场景构建 ----

def build_rotation_scene(parts):
    if not parts: return None
    t = random.choice(parts)
    return {"target": t, "all_parts": parts, "target_source": t["source"]}


# ---- 核心渲染 ----

def render_textured_query(path, cam, all_parts, r, el, az, samples):
    set_camera_pose(cam, [p["object"] for p in all_parts], r, el, az)
    for p in all_parts: p["object"].hide_render = False; set_object_materials(p["object"], p["materials"])
    set_transparent_background(True); setup_world_background(1.0, (1, 1, 1, 1))
    setup_render_settings(bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y, samples)
    render_still(path)


def render_rotation_pair(shape_id, out_dir, scene_data, cam, r, el, az, samples):
    target = scene_data["target"]
    all_parts = scene_data["all_parts"]
    tobj = target["object"]
    tmats = list(tobj.data.materials)
    orig_matrix = tobj.matrix_world.copy()
    all_objs = [p["object"] for p in all_parts]

    # 旋转参数
    pivot = choose_bbox_corner_pivot(tobj)
    axis = random_unit_vector()
    angle = random.uniform(MIN_ROTATION_DEG, MAX_ROTATION_DEG)
    if random.random() < 0.5: angle = -angle

    az_d = int(round(math.degrees(az))) % 360; el_d = int(round(math.degrees(el)))
    stem = f"render_{shape_id}_{r}_{az_d}_{el_d}_Rotation"

    n_img = os.path.join(out_dir, stem + "_normal.png")
    n_json = os.path.join(out_dir, stem + "_normal.json")
    a_img = os.path.join(out_dir, stem + "_anomaly.png")
    a_json = os.path.join(out_dir, stem + "_anomaly.json")
    n_mask = os.path.join(out_dir, stem + "_normal_mask.png")
    a_mask = os.path.join(out_dir, stem + "_anomaly_mask.png")
    comp_p = os.path.join(out_dir, stem + "_complete_tmp.png")

    # Step 1: 正常纹理图（旋转前）
    render_textured_query(n_img, cam, all_parts, r, el, az, samples)

    # Step 2: 完整物体mask + 旋转前部件mask（带遮挡）
    render_complete_mask(comp_p, all_objs)
    other_objs = [p["object"] for p in all_parts if p["object"] != tobj]
    render_occlusion_mask(n_mask, [tobj], other_objs)

    complete_px = count_mask_pixels(comp_p)
    normal_px = count_mask_pixels(n_mask)
    normal_ratio = normal_px / complete_px if complete_px > 0 else 0.0
    remove_file_if_exists(comp_p)

    if normal_ratio < MIN_MASK_RATIO:
        remove_file_if_exists(n_img); remove_file_if_exists(n_json); remove_file_if_exists(n_mask)
        return False, 0.0

    # Step 3: 旋转target
    rotate_part_around_pivot(tobj, pivot, angle, axis)

    # Step 4: 旋转后部件mask（带遮挡）
    render_complete_mask(comp_p, all_objs)
    render_occlusion_mask(a_mask, [tobj], other_objs)

    complete_px = count_mask_pixels(comp_p)
    anomaly_px = count_mask_pixels(a_mask)
    anomaly_ratio = anomaly_px / complete_px if complete_px > 0 else 0.0
    remove_file_if_exists(comp_p)

    if anomaly_ratio < MIN_MASK_RATIO:
        remove_file_if_exists(a_mask); remove_file_if_exists(n_mask)
        remove_file_if_exists(n_img); remove_file_if_exists(n_json)
        tobj.matrix_world = orig_matrix
        set_object_materials(tobj, tmats)
        return False, anomaly_ratio

    # Step 5: 异常查询图（target保持旋转位置）
    set_object_materials(tobj, tmats)
    for p in all_parts: p["object"].hide_render = False
    set_transparent_background(True); setup_world_background(1.0, (1, 1, 1, 1))
    setup_render_settings(bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y, samples)
    render_still(a_img)
    save_json(n_json, cam, scene_data["target_source"], "normal", r, el, az,
              normal_mask_ratio=normal_ratio)
    save_json(a_json, cam, scene_data["target_source"], "rotation", r, el, az,
              pivot, angle, axis,
              normal_mask_ratio=normal_ratio, anomaly_mask_ratio=anomaly_ratio)
    return True, anomaly_ratio


def parse_args():
    av = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    dr = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    p = ArgumentParser(description="Render micro-rotation anomaly.")
    p.add_argument("--root", type=str, default=dr); 
    p.add_argument("--category", type=str, default="Mug")
    p.add_argument("--num_views", type=int, default=10)
    p.add_argument("--width", type=int, default=512); 
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--samples", type=int, default=64); 
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_shapes", type=int, default=None); 
    p.add_argument("--max_attempts", type=int, default=100)
    p.add_argument("--min_rotation_deg", type=float, default=MIN_ROTATION_DEG)
    p.add_argument("--max_rotation_deg", type=float, default=MAX_ROTATION_DEG)
    p.add_argument("--min_mask_ratio", type=float, default=MIN_MASK_RATIO)
    p.add_argument("--texture_categories", nargs="+",
                   default=["pure_color", "fabric", "wood", "paper"])
    return p.parse_args(av)


def main():
    args = parse_args()
    if args.seed is not None: random.seed(args.seed); np.random.seed(args.seed)
    global MIN_ROTATION_DEG, MAX_ROTATION_DEG, MIN_MASK_RATIO
    MIN_ROTATION_DEG = args.min_rotation_deg; MAX_ROTATION_DEG = args.max_rotation_deg
    MIN_MASK_RATIO = args.min_mask_ratio

    root = os.path.abspath(args.root)
    cp = os.path.join(root, "shapes", args.category)
    tr = os.path.join(root, "textures")
    ir = os.path.join(root, "Rotation", args.category)
    if not os.path.isdir(cp): raise FileNotFoundError(f"{cp} not found")
    os.makedirs(ir, exist_ok=True)

    shapes = sorted([s for s in os.listdir(cp) if s.startswith("shape_")])
    if args.max_shapes: shapes = shapes[:args.max_shapes]
    print(f"[INFO] Rotation | {args.category} | {len(shapes)} shapes | {args.num_views} views")

    for shape in shapes:
        sid = shape.split("_", 1)[1]
        op = os.path.join(cp, shape, "objs")
        if not os.path.isdir(op): continue
        if not [f for f in os.listdir(op) if f.lower().endswith(".obj")]: continue
        od = os.path.join(ir, shape); os.makedirs(od, exist_ok=True)

        vd = 0
        for attempt in range(args.max_attempts):
            if vd >= args.num_views: break
            full_cleanup()
            setup_render_settings(args.width, args.height, args.samples)
            parts = load_part_scene(op, tr, args.texture_categories, random.randint(1, 5))

            rr = round(random.uniform(3, 4.5), 1)
            el = random.uniform(math.pi / 9, 2 * math.pi / 9)
            base_az = random.uniform(0, 2 * math.pi)

            for off in [0, 2 * math.pi / 3, 4 * math.pi / 3]:
                if vd >= args.num_views: break
                az = (base_az + off) % (2 * math.pi)
                cam = create_camera()

                sd = build_rotation_scene(parts)
                if sd is None: continue

                orig_mat = sd["target"]["object"].matrix_world.copy()

                ok, ratio = render_rotation_pair(sid, od, sd, cam, rr, el, az, args.samples)

                sd["target"]["object"].matrix_world = orig_mat

                if ok:
                    print(f"[OK] {shape} view={vd} part={os.path.basename(sd['target_source'])} "
                          f"az={math.degrees(az):.0f} mask_ratio={ratio:.3f}")
                    vd += 1

        if vd < args.num_views:
            print(f"[SKIP] {shape} only {vd}/{args.num_views} views")

    full_cleanup()


if __name__ == "__main__":
    main()
