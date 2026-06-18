# coding=utf-8
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

from utils.base_render import (  # noqa: E402
    calculate_bbox,
    create_camera,
    create_emission_material,
    full_cleanup,
    get_camera_extrinsics,
    get_camera_intrinsics,
    import_obj_file,
    join_objects,
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


MIN_DENT_RATIO = 0.005
MAX_DENT_RATIO = 0.25
DENT_REGION_SUBDIV_CUTS = 2
DEFAULT_DENT_MIN_HEIGHT_RATIO = 0.10
DEFAULT_DENT_MAX_HEIGHT_RATIO = 0.14
DEFAULT_DENT_MASK_SCALE = 1.0
DEFAULT_DENT_SHORT_AXIS_MIN_RATIO = 0.055
DEFAULT_DENT_SHORT_AXIS_MAX_RATIO = 0.095
DEFAULT_DENT_LONG_AXIS_MIN_RATIO = DEFAULT_DENT_SHORT_AXIS_MAX_RATIO
DEFAULT_DENT_LONG_AXIS_MAX_RATIO = 0.3
DEFAULT_DENT_LONG_AXIS_PLATEAU_RATIO = 0.55
DEFAULT_DENT_EDGE_TAPER_START = 0.88
DEFAULT_WHOLE_MESH_SUBDIV_CUTS = 2
GENERATED_UV_NAME = "DentTextureUV"


def bbox_volume(obj):
    _, size = calculate_bbox(obj)
    return max(size.x, 1e-6) * max(size.y, 1e-6) * max(size.z, 1e-6)


def choose_texture_materials(texture_root, texture_categories, texture_nums):
    available = [c for c in texture_categories if os.path.isdir(os.path.join(texture_root, c))]
    if not available:
        raise FileNotFoundError(f"No texture categories found under {texture_root}")

    categories = list(available)
    random.shuffle(categories)

    for category in categories:
        materials = build_texture_library(texture_root, category, texture_nums)
        if materials:
            return materials

    raise FileNotFoundError(f"No texture images found under {texture_root}")


def has_image_texture_material(obj):
    for material in obj.data.materials:
        if material is None or not material.use_nodes or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.bl_idname == "ShaderNodeTexImage" and node.image is not None:
                return True
    return False


def uv_layer_has_area(mesh, uv_layer):
    if uv_layer is None or not uv_layer.data:
        return False

    u_min = v_min = float("inf")
    u_max = v_max = float("-inf")
    for item in uv_layer.data:
        u_min = min(u_min, item.uv.x)
        u_max = max(u_max, item.uv.x)
        v_min = min(v_min, item.uv.y)
        v_max = max(v_max, item.uv.y)
    return (u_max - u_min) > 1e-5 and (v_max - v_min) > 1e-5


def first_usable_uv_layer(mesh):
    for uv_layer in mesh.uv_layers:
        if uv_layer_has_area(mesh, uv_layer):
            return uv_layer
    return None


def set_active_uv_layer(mesh, uv_layer):
    for index, layer in enumerate(mesh.uv_layers):
        if layer == uv_layer:
            mesh.uv_layers.active_index = index
            try:
                layer.active_render = True
            except AttributeError:
                pass
            return


def assign_box_projected_uvs(obj):
    mesh = obj.data
    if not mesh.vertices or not mesh.polygons:
        return None

    uv_layer = mesh.uv_layers.get(GENERATED_UV_NAME)
    if uv_layer is None:
        uv_layer = mesh.uv_layers.new(name=GENERATED_UV_NAME)

    coords = [vert.co for vert in mesh.vertices]
    min_x = min(co.x for co in coords)
    min_y = min(co.y for co in coords)
    min_z = min(co.z for co in coords)
    max_x = max(co.x for co in coords)
    max_y = max(co.y for co in coords)
    max_z = max(co.z for co in coords)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    span_z = max(max_z - min_z, 1e-6)

    for poly in mesh.polygons:
        normal = poly.normal
        abs_x, abs_y, abs_z = abs(normal.x), abs(normal.y), abs(normal.z)
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            co = mesh.vertices[vertex_index].co
            if abs_z >= abs_x and abs_z >= abs_y:
                uv = ((co.x - min_x) / span_x, (co.y - min_y) / span_y)
            elif abs_y >= abs_x:
                uv = ((co.x - min_x) / span_x, (co.z - min_z) / span_z)
            else:
                uv = ((co.y - min_y) / span_y, (co.z - min_z) / span_z)
            uv_layer.data[loop_index].uv = uv

    set_active_uv_layer(mesh, uv_layer)
    mesh.update()
    obj["_dent_generated_uvs"] = True
    return uv_layer


def bind_image_textures_to_uv(obj, uv_layer):
    if uv_layer is None:
        return

    uv_name = uv_layer.name
    uv_names = {layer.name for layer in obj.data.uv_layers}
    for material in obj.data.materials:
        if material is None or not material.use_nodes or material.node_tree is None:
            continue

        tree = material.node_tree
        for node in tree.nodes:
            if node.bl_idname == "ShaderNodeUVMap":
                node.uv_map = uv_name

        for node in tree.nodes:
            if node.bl_idname != "ShaderNodeTexImage" or node.image is None:
                continue
            vector_input = node.inputs.get("Vector")
            if vector_input is None or vector_input.is_linked:
                continue
            uv_node = tree.nodes.new(type="ShaderNodeUVMap")
            uv_node.uv_map = uv_name
            uv_node.location = (node.location.x - 220, node.location.y)
            tree.links.new(uv_node.outputs["UV"], vector_input)


def prepare_texture_mapping(obj):
    if obj is None or obj.type != "MESH" or not has_image_texture_material(obj):
        return

    uv_layer = first_usable_uv_layer(obj.data)
    if uv_layer is None:
        uv_layer = assign_box_projected_uvs(obj)
    else:
        set_active_uv_layer(obj.data, uv_layer)
        obj["_dent_generated_uvs"] = False

    bind_image_textures_to_uv(obj, uv_layer)


def load_part_scene(objs_path, texture_root, texture_categories, texture_nums, subdiv_cuts=0):
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
            prepare_texture_mapping(part)
            subdivide_whole_mesh(part, subdiv_cuts)
            parts.append({
                "object": part,
                "source": os.path.join(objs_path, obj_file),
                "materials": list(part.data.materials),
            })
    return parts


def make_orthonormal_basis(normal):
    normal = normal.normalized()
    ref = Vector((0.0, 0.0, 1.0))
    if abs(normal.dot(ref)) > 0.92:
        ref = Vector((1.0, 0.0, 0.0))
    tangent_u = ref.cross(normal).normalized()
    tangent_v = normal.cross(tangent_u).normalized()
    return tangent_u, tangent_v


def rotate_tangent_basis(tangent_u, tangent_v, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated_u = (tangent_u * cos_a + tangent_v * sin_a).normalized()
    rotated_v = (-tangent_u * sin_a + tangent_v * cos_a).normalized()
    return rotated_u, rotated_v


def normalized_ratio_range(min_ratio, max_ratio, fallback_min, fallback_max):
    min_ratio = fallback_min if min_ratio is None else float(min_ratio)
    max_ratio = fallback_max if max_ratio is None else float(max_ratio)
    min_ratio = max(min_ratio, 1e-5)
    max_ratio = max(max_ratio, 1e-5)
    if min_ratio > max_ratio:
        min_ratio, max_ratio = max_ratio, min_ratio
    return min_ratio, max_ratio


def clamp(value, low, high):
    return min(max(value, low), high)


def quadratic_plateau_falloff(abs_coord, plateau_ratio):
    plateau_ratio = clamp(plateau_ratio, 0.0, 0.95)
    abs_coord = clamp(abs_coord, 0.0, 1.0)
    if abs_coord <= plateau_ratio:
        return 1.0

    t = (abs_coord - plateau_ratio) / max(1.0 - plateau_ratio, 1e-6)
    return max(0.0, 1.0 - t * t)


def quadratic_edge_taper(r, start=DEFAULT_DENT_EDGE_TAPER_START):
    start = clamp(start, 0.0, 0.98)
    r = clamp(r, 0.0, 1.0)
    if r <= start:
        return 1.0

    t = (r - start) / max(1.0 - start, 1e-6)
    return max(0.0, 1.0 - t * t)


def smooth_target_normals(obj):
    for poly in obj.data.polygons:
        poly.use_smooth = True
    obj.data.update()


def exterior_vertex_indices(obj, center, min_dot=0.15):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()

    indices = []
    for vert in mesh.vertices:
        world_co = obj.matrix_world @ vert.co
        radial = world_co - center
        if radial.length < 1e-6:
            continue
        normal = (normal_matrix @ vert.normal).normalized()
        if normal.dot(radial.normalized()) >= min_dot:
            indices.append(vert.index)

    obj_eval.to_mesh_clear()
    return indices


def boundary_vertex_indices(obj):
    mesh = obj.data
    edge_face_counts = {tuple(sorted(edge.vertices)): 0 for edge in mesh.edges}
    for poly in mesh.polygons:
        verts = list(poly.vertices)
        for i, v1 in enumerate(verts):
            v2 = verts[(i + 1) % len(verts)]
            key = tuple(sorted((v1, v2)))
            if key in edge_face_counts:
                edge_face_counts[key] += 1

    boundary = set()
    for (v1, v2), count in edge_face_counts.items():
        if count <= 1:
            boundary.add(v1)
            boundary.add(v2)
    return boundary


def choose_dent_seed(target_obj, camera=None,
                      dent_min_height_ratio=DEFAULT_DENT_MIN_HEIGHT_RATIO,
                      dent_max_height_ratio=DEFAULT_DENT_MAX_HEIGHT_RATIO,
                      dent_mask_scale=DEFAULT_DENT_MASK_SCALE,
                      short_axis_min_ratio=DEFAULT_DENT_SHORT_AXIS_MIN_RATIO,
                      short_axis_max_ratio=DEFAULT_DENT_SHORT_AXIS_MAX_RATIO,
                      long_axis_min_ratio=DEFAULT_DENT_LONG_AXIS_MIN_RATIO,
                      long_axis_max_ratio=DEFAULT_DENT_LONG_AXIS_MAX_RATIO,
                      long_axis_plateau_ratio=DEFAULT_DENT_LONG_AXIS_PLATEAU_RATIO):
    center, size = calculate_bbox(target_obj)
    exterior = exterior_vertex_indices(target_obj, center)
    boundary = boundary_vertex_indices(target_obj)
    vertices = target_obj.data.vertices
    z_min = center.z - size.z * 0.5
    z_max = center.z + size.z * 0.5
    z_margin = size.z * 0.12
    interior_exterior = []
    for idx in exterior:
        world_co = target_obj.matrix_world @ vertices[idx].co
        if idx in boundary:
            continue
        if world_co.z < z_min + z_margin or world_co.z > z_max - z_margin:
            continue
        interior_exterior.append(idx)
    if interior_exterior:
        exterior = interior_exterior
    if not exterior:
        exterior = list(range(len(vertices)))
    if not exterior:
        return None

    normal_matrix = target_obj.matrix_world.to_3x3().inverted().transposed()
    visible = []
    if camera is not None:
        for idx in exterior:
            world_co = target_obj.matrix_world @ vertices[idx].co
            normal = normal_matrix @ vertices[idx].normal
            view_dir = camera.location - world_co
            if normal.length < 1e-6 or view_dir.length < 1e-6:
                continue
            if normal.normalized().dot(view_dir.normalized()) >= 0.65:
                visible.append(idx)

    seed_index = random.choice(visible if visible else exterior)
    seed_world = target_obj.matrix_world @ vertices[seed_index].co
    seed_normal = normal_matrix @ vertices[seed_index].normal
    if seed_normal.length < 1e-6:
        seed_normal = seed_world - center
    if seed_normal.length < 1e-6:
        seed_normal = Vector((1.0, 0.0, 0.0))
    seed_normal.normalize()

    tangent_u, tangent_v = make_orthonormal_basis(seed_normal)
    tangent_u, tangent_v = rotate_tangent_basis(tangent_u, tangent_v, random.uniform(0.0, 2.0 * math.pi))
    max_dim = max(size.x, size.y, size.z, 1e-3)
    mask_scale = max(dent_mask_scale, 0.05)
    min_height_ratio, max_height_ratio = normalized_ratio_range(
        dent_min_height_ratio,
        dent_max_height_ratio,
        DEFAULT_DENT_MIN_HEIGHT_RATIO,
        DEFAULT_DENT_MAX_HEIGHT_RATIO,
    )

    short_min, short_max = normalized_ratio_range(
        short_axis_min_ratio,
        short_axis_max_ratio,
        DEFAULT_DENT_SHORT_AXIS_MIN_RATIO,
        DEFAULT_DENT_SHORT_AXIS_MAX_RATIO,
    )
    long_min, long_max = normalized_ratio_range(
        long_axis_min_ratio,
        long_axis_max_ratio,
        DEFAULT_DENT_LONG_AXIS_MIN_RATIO,
        DEFAULT_DENT_LONG_AXIS_MAX_RATIO,
    )
    long_min = max(long_min, short_max)
    long_max = max(long_max, long_min)

    short_axis = max_dim * random.uniform(short_min, short_max) * mask_scale
    long_axis = max_dim * random.uniform(long_min, long_max) * mask_scale
    if long_axis < short_axis:
        short_axis, long_axis = long_axis, short_axis
    axis_a = long_axis
    axis_b = short_axis

    height = max_dim * random.uniform(min_height_ratio, max_height_ratio)
    normal_band = max(max_dim * 0.05, short_axis * 0.45)

    return {
        "center": seed_world,
        "normal": seed_normal,
        "tangent_u": tangent_u,
        "tangent_v": tangent_v,
        "axis_a": axis_a,
        "axis_b": axis_b,
        "long_axis": long_axis,
        "short_axis": short_axis,
        "long_axis_ratio": long_axis / max_dim,
        "short_axis_ratio": short_axis / max_dim,
        "long_axis_plateau_ratio": clamp(long_axis_plateau_ratio, 0.0, 0.95),
        "height": height,
        "height_ratio": height / max_dim,
        "min_height_ratio": min_height_ratio,
        "max_height_ratio": max_height_ratio,
        "mask_scale": mask_scale,
        "normal_band": normal_band,
        "max_dim": max_dim,
    }


def dent_weight(world_co, params):
    delta = world_co - params["center"]
    normal_offset = abs(delta.dot(params["normal"])) / params["normal_band"]
    if normal_offset > 1.0:
        return 0.0

    u = delta.dot(params["tangent_u"]) / params["axis_a"]
    v = delta.dot(params["tangent_v"]) / params["axis_b"]
    r2 = u * u + v * v
    if r2 > 1.0:
        return 0.0

    r = math.sqrt(r2)
    long_weight = quadratic_plateau_falloff(abs(u), params.get("long_axis_plateau_ratio", 0.0))
    short_weight = max(0.0, 1.0 - v * v)
    edge_weight = quadratic_edge_taper(r)
    normal_weight = max(0.0, 1.0 - normal_offset * normal_offset)
    planar_weight = long_weight * short_weight * edge_weight
    return planar_weight * normal_weight


def dent_planar_weight(world_co, params):
    delta = world_co - params["center"]
    u = delta.dot(params["tangent_u"]) / params["axis_a"]
    v = delta.dot(params["tangent_v"]) / params["axis_b"]
    r2 = u * u + v * v
    if r2 > 1.0:
        return 0.0

    r = math.sqrt(r2)
    long_weight = quadratic_plateau_falloff(abs(u), params.get("long_axis_plateau_ratio", 0.0))
    short_weight = max(0.0, 1.0 - v * v)
    edge_weight = quadratic_edge_taper(r)
    return long_weight * short_weight * edge_weight


def candidate_dent_faces(obj, params):
    mesh = obj.data
    face_indices = []
    for poly in mesh.polygons:
        coords = [obj.matrix_world @ mesh.vertices[idx].co for idx in poly.vertices]
        center = sum(coords, Vector((0.0, 0.0, 0.0))) / len(coords)
        if dent_weight(center, params) > 0.0:
            face_indices.append(poly.index)
            continue
        if any(dent_weight(co, params) > 0.0 for co in coords):
            face_indices.append(poly.index)
    return face_indices


def subdivide_dent_region(obj, face_indices, cuts=DENT_REGION_SUBDIV_CUTS):
    if not face_indices or cuts <= 0:
        return

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    bm.edges.ensure_lookup_table()

    face_index_set = set(face_indices)
    edges = set()
    for face in bm.faces:
        if face.index in face_index_set:
            edges.update(face.edges)

    if edges:
        bmesh.ops.subdivide_edges(
            bm,
            edges=list(edges),
            cuts=cuts,
            use_grid_fill=True,
            smooth=0.0,
        )
        bm.normal_update()
        bm.to_mesh(mesh)
        mesh.update()

    bm.free()
    if obj.get("_dent_generated_uvs"):
        assign_box_projected_uvs(obj)
    smooth_target_normals(obj)


def subdivide_whole_mesh(obj, cuts=DEFAULT_WHOLE_MESH_SUBDIV_CUTS):
    if obj is None or obj.type != "MESH" or cuts <= 0:
        return

    mesh = obj.data
    if not mesh.edges:
        return

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()
    edges = list(bm.edges)

    if edges:
        bmesh.ops.subdivide_edges(
            bm,
            edges=edges,
            cuts=cuts,
            use_grid_fill=True,
            smooth=0.0,
        )
        bm.normal_update()
        bm.to_mesh(mesh)
        mesh.update()

    bm.free()
    if obj.get("_dent_generated_uvs"):
        assign_box_projected_uvs(obj)
    smooth_target_normals(obj)


def create_constant_displace_texture():
    image = bpy.data.images.new("Dent_displace_white_image", width=1, height=1, alpha=True)
    image.pixels[0:4] = (1.0, 1.0, 1.0, 1.0)

    texture = bpy.data.textures.new("Dent_displace_white_texture", type="IMAGE")
    texture.image = image
    return texture


def create_dent_vertex_group(target_obj, params):
    mesh = target_obj.data
    vertex_group = target_obj.vertex_groups.new(name="Dent_Displace_Weight")
    selected_vertices = set()
    vertex_weights = {}

    for vert in mesh.vertices:
        world_co = target_obj.matrix_world @ vert.co
        weight = dent_weight(world_co, params)
        if weight <= 0.0:
            continue
        vertex_group.add([vert.index], weight, "REPLACE")
        selected_vertices.add(vert.index)
        vertex_weights[vert.index] = weight

    return vertex_group, selected_vertices, vertex_weights


def collect_dent_faces(target_obj, params, vertex_weights):
    mesh = target_obj.data
    face_weights = {}

    for poly in mesh.polygons:
        if not poly.vertices:
            continue

        coords = [target_obj.matrix_world @ mesh.vertices[idx].co for idx in poly.vertices]
        center = sum(coords, Vector((0.0, 0.0, 0.0))) / len(coords)
        weight = dent_weight(center, params)
        if weight <= 0.0:
            weight = max((vertex_weights.get(idx, 0.0) for idx in poly.vertices), default=0.0)
        if weight > 0.0:
            face_weights[poly.index] = weight

    return face_weights


def apply_dent_displace_modifier(target_obj, vertex_group, params):
    texture = create_constant_displace_texture()
    modifier = target_obj.modifiers.new("Dent_Displace", "DISPLACE")
    modifier.strength = -params["height"]
    modifier.mid_level = 0.0
    modifier.direction = "NORMAL"
    modifier.vertex_group = vertex_group.name
    modifier.texture = texture

    bpy.ops.object.select_all(action="DESELECT")
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)

    smooth_target_normals(target_obj)


def deform_target_to_dent(target_obj, params):
    vertex_group, selected_vertices, vertex_weights = create_dent_vertex_group(target_obj, params)
    if not selected_vertices:
        return selected_vertices, [], vertex_weights, {}

    face_weights = collect_dent_faces(target_obj, params, vertex_weights)
    apply_dent_displace_modifier(target_obj, vertex_group, params)

    selected_faces = sorted(face_weights.keys())
    return selected_vertices, selected_faces, vertex_weights, face_weights


def create_dent_mask_object(target_obj, selected_faces, offset, push_normal):
    mesh = target_obj.data
    if not selected_faces:
        return None

    used_indices = []
    used_set = set()
    for face_index in selected_faces:
        for vid in mesh.polygons[face_index].vertices:
            if vid not in used_set:
                used_set.add(vid)
                used_indices.append(vid)

    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_indices)}
    verts = []
    for old_idx in used_indices:
        vert = mesh.vertices[old_idx]
        world_co = target_obj.matrix_world @ vert.co
        verts.append(tuple(world_co + push_normal * offset))

    faces = []
    for face_index in selected_faces:
        poly = mesh.polygons[face_index]
        faces.append([index_map[idx] for idx in poly.vertices])

    mask_mesh = bpy.data.meshes.new("Dent_mask_mesh")
    mask_mesh.from_pydata(verts, [], faces)
    mask_mesh.update()
    mask_obj = bpy.data.objects.new("Dent_mask_region", mask_mesh)
    bpy.context.collection.objects.link(mask_obj)
    smooth_target_normals(mask_obj)
    return mask_obj


def export_obj_scene(path, objects):
    objects = [obj for obj in objects if obj is not None and obj.type == "MESH"]
    if not objects:
        raise ValueError("No mesh objects available for OBJ export")

    path = os.path.abspath(path)
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]

    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=path,
            export_selected_objects=True,
            export_materials=True,
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=path,
            use_selection=True,
            use_materials=True,
        )
    return path


def normalize_shape_name(shape):
    if shape is None:
        return None
    return shape if shape.startswith("shape_") else f"shape_{shape}"


def build_dent_scene(parts, largest_k, camera=None,
                      dent_min_height_ratio=DEFAULT_DENT_MIN_HEIGHT_RATIO,
                      dent_max_height_ratio=DEFAULT_DENT_MAX_HEIGHT_RATIO,
                      dent_mask_scale=DEFAULT_DENT_MASK_SCALE,
                      short_axis_min_ratio=DEFAULT_DENT_SHORT_AXIS_MIN_RATIO,
                      short_axis_max_ratio=DEFAULT_DENT_SHORT_AXIS_MAX_RATIO,
                      long_axis_min_ratio=DEFAULT_DENT_LONG_AXIS_MIN_RATIO,
                      long_axis_max_ratio=DEFAULT_DENT_LONG_AXIS_MAX_RATIO,
                      long_axis_plateau_ratio=DEFAULT_DENT_LONG_AXIS_PLATEAU_RATIO,
                      whole_mesh_subdiv_cuts=DEFAULT_WHOLE_MESH_SUBDIV_CUTS,
                      dent_region_subdiv_cuts=DENT_REGION_SUBDIV_CUTS):
    if not parts:
        return None

    candidates = sorted(parts, key=lambda item: bbox_volume(item["object"]), reverse=True)
    target_info = random.choice(candidates[:max(1, min(largest_k, len(candidates)))])
    target_obj = target_info["object"]

    params = choose_dent_seed(
        target_obj,
        camera,
        dent_min_height_ratio=dent_min_height_ratio,
        dent_max_height_ratio=dent_max_height_ratio,
        dent_mask_scale=dent_mask_scale,
        short_axis_min_ratio=short_axis_min_ratio,
        short_axis_max_ratio=short_axis_max_ratio,
        long_axis_min_ratio=long_axis_min_ratio,
        long_axis_max_ratio=long_axis_max_ratio,
        long_axis_plateau_ratio=long_axis_plateau_ratio,
    )
    if params is None:
        return None

    face_indices = candidate_dent_faces(target_obj, params)
    if not face_indices:
        return None
    subdivide_dent_region(target_obj, face_indices, dent_region_subdiv_cuts)

    selected_vertices, selected_faces, _, _ = deform_target_to_dent(target_obj, params)
    if not selected_vertices or not selected_faces:
        return None

    mask_obj = create_dent_mask_object(
        target_obj,
        selected_faces,
        max(params["max_dim"] * 0.004, 1e-4),
        params["normal"],
    )
    if mask_obj is None:
        return None
    mask_obj.hide_render = True

    dented_parts = [item["object"] for item in parts]
    return {
        "dented_parts": dented_parts,
        "mask_obj": mask_obj,
        "target_source": target_info["source"],
        "all_for_camera": dented_parts,
        "dent_axis_a": params["axis_a"],
        "dent_axis_b": params["axis_b"],
        "dent_long_axis": params["long_axis"],
        "dent_short_axis": params["short_axis"],
        "dent_long_axis_ratio": params["long_axis_ratio"],
        "dent_short_axis_ratio": params["short_axis_ratio"],
        "dent_long_axis_plateau_ratio": params["long_axis_plateau_ratio"],
        "dent_depth": params["height"],
        "dent_depth_ratio": params["height_ratio"],
        "dent_min_depth_ratio": params["min_height_ratio"],
        "dent_max_depth_ratio": params["max_height_ratio"],
        "dent_height": params["height"],
        "dent_height_ratio": params["height_ratio"],
        "dent_min_height_ratio": params["min_height_ratio"],
        "dent_max_height_ratio": params["max_height_ratio"],
        "dent_mask_scale": params["mask_scale"],
        "dent_deformation_method": "displace_modifier_vertex_group_negative_strength",
        "whole_mesh_subdiv_cuts": whole_mesh_subdiv_cuts,
        "dent_region_subdiv_cuts": dent_region_subdiv_cuts,
        "dent_center": list(params["center"]),
        "dent_normal": list(params["normal"]),
    }


def render_dent_mask(path, dented_parts, mask_obj, mode):
    white = create_emission_material("Dent_mask_white", (1, 1, 1, 1))
    black = create_emission_material("Dent_mask_black", (0, 0, 0, 1))

    if mode == "dent":
        for obj in dented_parts:
            obj.data.materials.clear()
            obj.data.materials.append(black)
            obj.hide_render = False
        mask_obj.data.materials.clear()
        mask_obj.data.materials.append(white)
        mask_obj.hide_render = False
    elif mode == "complete":
        for obj in dented_parts:
            obj.data.materials.clear()
            obj.data.materials.append(white)
            obj.hide_render = False
        mask_obj.hide_render = True
    else:
        raise ValueError(f"Unknown mask mode: {mode}")

    set_transparent_background(False)
    setup_world_background(0.0, (0, 0, 0, 1))
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.cycles.samples = 1
    render_still(path)


def save_dent_json(image_path, camera, scene_data, radius, elevation, azimuth, ratio):
    scene = bpy.context.scene
    rt, rt_ = get_camera_extrinsics(camera)
    data = {
        "K": get_camera_intrinsics(camera, scene),
        "RT": rt,
        "RT_": rt_,
        "anomaly_type": "Dent",
        "dent_part": os.path.basename(scene_data["target_source"]),
        "dent_mask_ratio": ratio,
        "dent_axis_a": scene_data["dent_axis_a"],
        "dent_axis_b": scene_data["dent_axis_b"],
        "dent_long_axis": scene_data["dent_long_axis"],
        "dent_short_axis": scene_data["dent_short_axis"],
        "dent_long_axis_ratio": scene_data["dent_long_axis_ratio"],
        "dent_short_axis_ratio": scene_data["dent_short_axis_ratio"],
        "dent_long_axis_plateau_ratio": scene_data["dent_long_axis_plateau_ratio"],
        "dent_depth": scene_data["dent_depth"],
        "dent_depth_ratio": scene_data["dent_depth_ratio"],
        "dent_min_depth_ratio": scene_data["dent_min_depth_ratio"],
        "dent_max_depth_ratio": scene_data["dent_max_depth_ratio"],
        "dent_height": scene_data["dent_height"],
        "dent_height_ratio": scene_data["dent_height_ratio"],
        "dent_min_height_ratio": scene_data["dent_min_height_ratio"],
        "dent_max_height_ratio": scene_data["dent_max_height_ratio"],
        "dent_mask_scale": scene_data["dent_mask_scale"],
        "dent_deformation_method": scene_data["dent_deformation_method"],
        "whole_mesh_subdiv_cuts": scene_data["whole_mesh_subdiv_cuts"],
        "dent_region_subdiv_cuts": scene_data["dent_region_subdiv_cuts"],
        "dent_center_world": scene_data["dent_center"],
        "dent_normal_world": scene_data["dent_normal"],
        "radius": radius,
        "elevation_deg": math.degrees(elevation),
        "azimuth_deg": math.degrees(azimuth) % 360,
    }
    with open(image_path.replace(".png", ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def restore_materials(objects, material_map):
    for obj in objects:
        if obj.name in material_map:
            set_object_materials(obj, material_map[obj.name])


def render_dent_pair(shape_id, output_dir, scene_data, camera, radius, elevation, azimuth,
                      min_dent_ratio, max_dent_ratio, final_samples):
    dented_parts = scene_data["dented_parts"]
    mask_obj = scene_data["mask_obj"]
    all_for_camera = scene_data["all_for_camera"]

    set_camera_pose(camera, all_for_camera, radius, elevation, azimuth)
    az_deg = int(round(math.degrees(azimuth))) % 360
    el_deg = int(round(math.degrees(elevation)))
    stem = f"render_{shape_id}_{radius}_{az_deg}_{el_deg}_Dent_anomaly"

    image_path = os.path.join(output_dir, stem + ".png")
    mask_path = os.path.join(output_dir, stem + "_mask.png")
    complete_mask_path = os.path.join(output_dir, stem + "_complete_mask_tmp.png")

    material_map = {obj.name: list(obj.data.materials) for obj in dented_parts + [mask_obj]}

    render_dent_mask(mask_path, dented_parts, mask_obj, "dent")
    render_dent_mask(complete_mask_path, dented_parts, mask_obj, "complete")

    dent_pixels = count_mask_pixels(mask_path)
    complete_pixels = count_mask_pixels(complete_mask_path)
    remove_file_if_exists(complete_mask_path)

    ratio = dent_pixels / complete_pixels if complete_pixels > 0 else 0.0
    if ratio < min_dent_ratio or ratio > max_dent_ratio:
        remove_file_if_exists(mask_path)
        restore_materials(dented_parts + [mask_obj], material_map)
        return False, ratio

    restore_materials(dented_parts + [mask_obj], material_map)
    for obj in dented_parts:
        obj.hide_render = False
    mask_obj.hide_render = True

    set_transparent_background(True)
    setup_world_background(1.0, (1, 1, 1, 1))
    setup_render_settings(
        bpy.context.scene.render.resolution_x,
        bpy.context.scene.render.resolution_y,
        final_samples,
    )
    render_still(image_path)
    save_dent_json(image_path, camera, scene_data, radius, elevation, azimuth, ratio)
    return True, ratio


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]

    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    parser = ArgumentParser(description="Render local dented-part anomaly images and masks.")
    parser.add_argument("--root", type=str, default=default_root)
    parser.add_argument("--category", type=str, default="Mug")
    parser.add_argument("--num_views", type=int, default=10)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_shapes", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=30)
    parser.add_argument("--largest_k", type=int, default=1)
    parser.add_argument("--min_dent_ratio", type=float, default=MIN_DENT_RATIO)
    parser.add_argument("--max_dent_ratio", type=float, default=MAX_DENT_RATIO)
    parser.add_argument("--dent_min_depth_ratio", "--dent_min_height_ratio", type=float,
                        default=DEFAULT_DENT_MIN_HEIGHT_RATIO,
                        dest="dent_min_height_ratio",
                        help="Minimum dent depth as a ratio of the target part max dimension.")
    parser.add_argument("--dent_max_depth_ratio", "--dent_max_height_ratio", type=float,
                        default=DEFAULT_DENT_MAX_HEIGHT_RATIO,
                        dest="dent_max_height_ratio",
                        help="Maximum dent depth as a ratio of the target part max dimension.")
    parser.add_argument("--dent_mask_scale", type=float, default=DEFAULT_DENT_MASK_SCALE,
                        help="Scale factor for the elliptical dent influence region and mask size.")
    parser.add_argument("--dent_short_axis_min_ratio", type=float,
                        default=DEFAULT_DENT_SHORT_AXIS_MIN_RATIO,
                        help="Minimum short-axis length as a ratio of the target part max dimension.")
    parser.add_argument("--dent_short_axis_max_ratio", type=float,
                        default=DEFAULT_DENT_SHORT_AXIS_MAX_RATIO,
                        help="Maximum short-axis length as a ratio of the target part max dimension.")
    parser.add_argument("--dent_long_axis_min_ratio", type=float,
                        default=DEFAULT_DENT_LONG_AXIS_MIN_RATIO,
                        help="Minimum long-axis length as a ratio of the target part max dimension.")
    parser.add_argument("--dent_long_axis_max_ratio", type=float,
                        default=DEFAULT_DENT_LONG_AXIS_MAX_RATIO,
                        help="Maximum long-axis length as a ratio of the target part max dimension.")
    parser.add_argument("--dent_long_axis_plateau_ratio", type=float,
                        default=DEFAULT_DENT_LONG_AXIS_PLATEAU_RATIO,
                        help="Flat high-dent fraction along the normalized long-axis half-length.")
    parser.add_argument("--whole_mesh_subdiv_cuts", type=int, default=DEFAULT_WHOLE_MESH_SUBDIV_CUTS,
                        help="Subdivision cuts applied to every imported mesh before scene construction.")
    parser.add_argument("--dent_region_subdiv_cuts", type=int, default=DENT_REGION_SUBDIV_CUTS,
                        help="Optional extra subdivision cuts applied only to the dent candidate region.")
    parser.add_argument("--export_subdivided_obj", type=str, default=None,
                        help="If set, import one shape, subdivide all parts, export this OBJ, then exit.")
    parser.add_argument("--export_shape", type=str, default=None,
                        help="Shape folder or id to export when --export_subdivided_obj is set.")
    parser.add_argument("--texture_categories", nargs="+",
                        default=["pure_color", "fabric", "ceramic", "paper"])
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    root = os.path.abspath(args.root)
    category_path = os.path.join(root, "shapes", args.category)
    texture_root = os.path.join(root, "textures")
    image_root = os.path.join(root, "Dent", args.category)

    if not os.path.isdir(category_path):
        raise FileNotFoundError(f"{category_path} not found")
    os.makedirs(image_root, exist_ok=True)

    shapes = sorted([s for s in os.listdir(category_path) if s.startswith("shape_")])
    if args.max_shapes is not None:
        shapes = shapes[:args.max_shapes]
    print(f"[INFO] category={args.category}, shapes={len(shapes)}, views_per_shape={args.num_views}")

    if args.export_subdivided_obj:
        shape = normalize_shape_name(args.export_shape) if args.export_shape else (shapes[0] if shapes else None)
        if shape is None:
            raise FileNotFoundError(f"No shape_* folders found under {category_path}")
        if shape not in shapes:
            raise FileNotFoundError(f"{shape} not found under {category_path}")

        objs_path = os.path.join(category_path, shape, "objs")
        if not os.path.isdir(objs_path):
            raise FileNotFoundError(f"missing objs: {objs_path}")

        full_cleanup()
        texture_nums = random.randint(3, 7)
        parts = load_part_scene(
            objs_path,
            texture_root,
            args.texture_categories,
            texture_nums,
            subdiv_cuts=args.whole_mesh_subdiv_cuts,
        )
        export_path = export_obj_scene(args.export_subdivided_obj, [item["object"] for item in parts])
        print(
            f"[OK] exported subdivided OBJ: {export_path} "
            f"shape={shape} parts={len(parts)} cuts={args.whole_mesh_subdiv_cuts}"
        )
        full_cleanup()
        return

    for shape in shapes:
        shape_id = shape.split("_", 1)[1]
        objs_path = os.path.join(category_path, shape, "objs")
        if not os.path.isdir(objs_path):
            print(f"[WARN] missing objs: {os.path.join(category_path, shape)}")
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

                texture_nums = random.randint(3, 7)
                parts = load_part_scene(
                    objs_path,
                    texture_root,
                    args.texture_categories,
                    texture_nums,
                    subdiv_cuts=args.whole_mesh_subdiv_cuts,
                )
                radius = round(random.uniform(3, 4.5), 1)
                elevation = random.uniform(math.pi / 9, 2 * math.pi / 9)
                azimuth = random.uniform(0, 2 * math.pi)
                set_camera_pose(camera, [item["object"] for item in parts], radius, elevation, azimuth)

                scene_data = build_dent_scene(
                    parts,
                    args.largest_k,
                    camera,
                    dent_min_height_ratio=args.dent_min_height_ratio,
                    dent_max_height_ratio=args.dent_max_height_ratio,
                    dent_mask_scale=args.dent_mask_scale,
                    short_axis_min_ratio=args.dent_short_axis_min_ratio,
                    short_axis_max_ratio=args.dent_short_axis_max_ratio,
                    long_axis_min_ratio=args.dent_long_axis_min_ratio,
                    long_axis_max_ratio=args.dent_long_axis_max_ratio,
                    long_axis_plateau_ratio=args.dent_long_axis_plateau_ratio,
                    whole_mesh_subdiv_cuts=args.whole_mesh_subdiv_cuts,
                    dent_region_subdiv_cuts=args.dent_region_subdiv_cuts,
                )
                if scene_data is None:
                    last_ratio = 0.0
                    continue

                ok, last_ratio = render_dent_pair(
                    shape_id,
                    image_shape_path,
                    scene_data,
                    camera,
                    radius,
                    elevation,
                    azimuth,
                    args.min_dent_ratio,
                    args.max_dent_ratio,
                    args.samples,
                )
                if ok:
                    print(
                        f"[OK] {shape} view={view_idx} "
                        f"part={os.path.basename(scene_data['target_source'])} "
                        f"ratio={last_ratio:.3f} "
                        f"short_axis={scene_data['dent_short_axis']:.4f} "
                        f"long_axis={scene_data['dent_long_axis']:.4f} "
                        f"long_plateau={scene_data['dent_long_axis_plateau_ratio']:.2f} "
                        f"depth={scene_data['dent_depth']:.4f}"
                    )
                    break

            if not ok:
                print(f"[SKIP] no valid dent: {shape} view={view_idx} last_ratio={last_ratio:.3f}")

    full_cleanup()


if __name__ == "__main__":
    main()
