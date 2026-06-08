# coding=utf-8
import os
import gc
import bpy
import math
import random
import numpy as np
from mathutils import Vector


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
MIN_MISSING_RATIO = 0.05
MAX_MISSING_RATIO = 0.50


def calc_loop_normals_from_pos(obj):
    mesh = obj.data
    vert_co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", vert_co)
    vert_co = vert_co.reshape(-1, 3)

    loop_vidx = np.empty(len(mesh.loops), dtype=np.uint32)
    mesh.loops.foreach_get("vertex_index", loop_vidx)

    poly_start = np.empty(len(mesh.polygons), dtype=np.uint32)
    poly_tot = np.empty(len(mesh.polygons), dtype=np.uint32)
    mesh.polygons.foreach_get("loop_start", poly_start)
    mesh.polygons.foreach_get("loop_total", poly_tot)

    def make_poly_normal(i):
        st, n = poly_start[i], poly_tot[i]
        vi = loop_vidx[st:st + n]
        v0, v1, v2 = vert_co[vi[0]], vert_co[vi[1]], vert_co[vi[2]]
        return np.cross(v1 - v0, v2 - v0)

    poly_n = np.array([make_poly_normal(i) for i in range(len(mesh.polygons))], dtype=np.float32)
    norm = np.linalg.norm(poly_n, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    poly_n /= norm

    loop_n = np.empty((len(mesh.loops), 3), dtype=np.float32)
    for i, (st, n) in enumerate(zip(poly_start, poly_tot)):
        loop_n[st:st + n] = poly_n[i]
    return loop_n


def set_loop_normals(obj, loop_normals):
    mesh = obj.data
    if not mesh.has_custom_normals:
        mesh.create_normals_split()
    mesh.normals_split_custom_set(loop_normals)
    mesh.polygons.foreach_set("use_smooth", np.zeros(len(mesh.polygons), dtype=bool))
    mesh.update()


def recalculate_normal(obj, only_blender_buildin=False):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.set_normals_from_faces()
    bpy.ops.object.editmode_toggle()
    if not only_blender_buildin:
        set_loop_normals(obj, calc_loop_normals_from_pos(obj))


def calculate_bbox(objects):
    if not isinstance(objects, (list, tuple)):
        objects = [objects]

    corners = []
    for obj in objects:
        if obj.type == "MESH":
            corners.extend([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])

    if not corners:
        return Vector((0, 0, 0)), Vector((1, 1, 1))

    min_corner = Vector((
        min(v.x for v in corners),
        min(v.y for v in corners),
        min(v.z for v in corners),
    ))
    max_corner = Vector((
        max(v.x for v in corners),
        max(v.y for v in corners),
        max(v.z for v in corners),
    ))
    return (min_corner + max_corner) / 2, max_corner - min_corner


def create_camera():
    cam_data = bpy.data.cameras.new("RenderCam")
    cam = bpy.data.objects.new("RenderCam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def camera_look_at(camera_obj, target_pose):
    direction = target_pose - camera_obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera_obj.rotation_euler = rot_quat.to_euler()


def set_transparent_background(enabled=True):
    bpy.context.scene.render.film_transparent = enabled


def setup_world_background(strength=1.0, color=(1, 1, 1, 1)):
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs["Color"].default_value = color
    bg.inputs["Strength"].default_value = strength
    output = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], output.inputs["Surface"])


def setup_render_settings(res_x=512, res_y=512, samples=64):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1


def get_camera_intrinsics(camera, scene):
    render = scene.render
    f_mm = camera.data.lens
    sensor_width = camera.data.sensor_width
    res_x = render.resolution_x
    res_y = render.resolution_y
    scale = render.resolution_percentage / 100.0
    fx = f_mm * res_x * scale / sensor_width
    fy = fx
    cx = res_x * scale / 2.0
    cy = res_y * scale / 2.0
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def get_camera_extrinsics(camera):
    c2w = np.array(camera.matrix_world)
    w2c = np.linalg.inv(c2w)
    return w2c.tolist(), c2w.tolist()


def create_texture_material(texture_path):
    mat = bpy.data.materials.new(name="Mat_" + os.path.basename(texture_path))
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    tex_node = nodes.new(type="ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(texture_path)
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def create_emission_material(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def build_texture_library(texture_root, texture_category, texture_nums):
    texture_dir = os.path.join(texture_root, texture_category)
    if not os.path.isdir(texture_dir):
        return []
    texture_files = [f for f in os.listdir(texture_dir) if f.lower().endswith(IMAGE_EXTS)]
    selected = random.sample(texture_files, min(texture_nums, len(texture_files)))
    return [create_texture_material(os.path.join(texture_dir, f)) for f in selected]


def ensure_uv(obj):
    if not obj.data.uv_layers:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.uv.smart_project()
        bpy.ops.object.mode_set(mode="OBJECT")


def import_obj_file(obj_path, material=None):
    before = set(bpy.context.scene.objects)
    bpy.ops.wm.obj_import(filepath=obj_path)
    after = set(bpy.context.scene.objects)
    new_objs = [obj for obj in (after - before) if obj.type == "MESH"]

    for obj in new_objs:
        ensure_uv(obj)
        if material is not None:
            obj.data.materials.clear()
            obj.data.materials.append(material)
        recalculate_normal(obj)
    return new_objs


def join_objects(objects, name):
    objects = [obj for obj in objects if obj and obj.type == "MESH"]
    if not objects:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    merged = bpy.context.view_layer.objects.active
    merged.name = name
    return merged



def set_camera_pose(camera, objects, radius, elevation, azimuth):
    center, _ = calculate_bbox(objects)
    x = radius * math.cos(elevation) * math.cos(azimuth)
    y = radius * math.cos(elevation) * math.sin(azimuth)
    z = radius * math.sin(elevation)
    camera.location = center + Vector((x, y, z))
    camera_look_at(camera, center)
    bpy.context.view_layer.update()


def render_still(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)



def set_object_materials(obj, materials):
    obj.data.materials.clear()
    for mat in materials:
        obj.data.materials.append(mat)


def remove_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def full_cleanup():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.data.orphans_purge(do_recursive=True)
    for collection in (bpy.data.cameras, bpy.data.lights, bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(collection):
            collection.remove(item)
    gc.collect()

