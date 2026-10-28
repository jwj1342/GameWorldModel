"""blender -b --python-exit-code 1 --python render_blender.py -- PACKET NEW_OUT

Imports frozen production meshes and camera states, then renders with CPU Cycles.
Y-up world coordinates are retained; no physics, animation interpolation or gameplay.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys


def matrix_rows(values):
    """Three.js column-major array -> Blender matrix row sequences."""
    if len(values) != 16 or not all(math.isfinite(x) for x in values):
        raise ValueError('invalid matrix')
    return [[values[c * 4 + r] for c in range(4)] for r in range(4)]


def validate_packet(packet):
    if packet.get('format') != 'gwm-frozen-scene' or packet.get('schema_version') != 1:
        raise ValueError('unsupported packet format')
    if not packet.get('frames'):
        raise ValueError('no frames')
    for dim in ('width', 'height'):
        if type(packet[dim]) is not int or packet[dim] < 1:
            raise ValueError('invalid dimensions')
    expected = None
    for frame in packet['frames']:
        if not math.isfinite(frame['t']) or frame['t'] < 0 or frame['t'] != frame['requested_t']:
            raise ValueError('time mismatch')
        camera = frame['camera']
        matrix_rows(camera['matrix_world'])
        if not 0 < camera['fov'] < 180 or abs(camera['aspect'] - packet['width'] / packet['height']) > 1e-9:
            raise ValueError('invalid camera')
        keys = [mesh['key'] for mesh in frame['meshes']]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError('empty or duplicate mesh IDs')
        if expected is None:
            expected = keys
        if keys != expected:
            raise ValueError('mesh identities changed')
        for mesh in frame['meshes']:
            matrix_rows(mesh['matrix_world'])
            if not mesh['vertices'] or any(len(v) != 3 or not all(math.isfinite(x) for x in v) for v in mesh['vertices']):
                raise ValueError('invalid vertices')
            if not mesh['triangles'] or any(len(t) != 3 or any(type(i) is not int or not 0 <= i < len(mesh['vertices']) for i in t) for t in mesh['triangles']):
                raise ValueError('invalid triangles')


def run(packet_path, out):
    import bpy
    from mathutils import Matrix, Vector
    from bpy_extras.object_utils import world_to_camera_view

    raw = packet_path.read_bytes()
    packet = json.loads(raw)
    validate_packet(packet)
    if hashlib.sha256((packet_path.parent / 'program.json').read_bytes()).hexdigest() != packet['program_sha256']:
        raise ValueError('source Program hash mismatch')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()  # Never overwrite an earlier run.
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 8
    scene.render.resolution_x, scene.render.resolution_y = packet['width'], packet['height']
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.4, 0.4, 0.4, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.8
    sun = bpy.data.objects.new('pilot_light', bpy.data.lights.new('pilot_light', 'SUN'))
    scene.collection.objects.link(sun)
    sun.data.energy = 2
    sun.rotation_euler = Vector((-4, -8, -6)).to_track_quat('-Z', 'Y').to_euler()
    cam = bpy.data.objects.new('shared_camera', bpy.data.cameras.new('shared_camera'))
    scene.collection.objects.link(cam)
    scene.camera = cam
    reports = []
    for frame_no, frame in enumerate(packet['frames']):
        for obj in list(scene.objects):
            if obj.type == 'MESH':
                bpy.data.objects.remove(obj, do_unlink=True)
        spec = frame['camera']
        cam.matrix_world = Matrix(matrix_rows(spec['matrix_world']))
        cam.data.type = 'PERSP'
        cam.data.sensor_fit = 'VERTICAL'
        cam.data.sensor_height = 36
        cam.data.lens = 18 / math.tan(math.radians(spec['fov']) / 2)
        cam.data.clip_start, cam.data.clip_end = spec['near'], spec['far']
        imported = []
        max_matrix_error = max(abs(cam.matrix_world[r][c] - matrix_rows(spec['matrix_world'])[r][c]) for r in range(4) for c in range(4))
        max_projection_error = 0.0
        max_vertex_error = 0.0
        for mesh in frame['meshes']:
            data = bpy.data.meshes.new(mesh['key'])
            data.from_pydata(mesh['vertices'], [], mesh['triangles'])
            data.update()
            max_vertex_error = max(max_vertex_error, *(abs(actual.co[j] - expected[j])
                for actual, expected in zip(data.vertices, mesh['vertices']) for j in range(3)))
            obj = bpy.data.objects.new(mesh['key'], data)
            scene.collection.objects.link(obj)
            obj.matrix_world = Matrix(matrix_rows(mesh['matrix_world']))
            obj.hide_render = not mesh['visible']
            material = bpy.data.materials.new(mesh['key'])
            material.use_nodes = True
            bsdf = material.node_tree.nodes['Principled BSDF']
            bsdf.inputs['Base Color'].default_value = (*mesh['material']['color'], 1)
            bsdf.inputs['Roughness'].default_value = mesh['material']['roughness']
            bsdf.inputs['Metallic'].default_value = mesh['material']['metalness']
            data.materials.append(material)
            rows = matrix_rows(mesh['matrix_world'])
            max_matrix_error = max(max_matrix_error, max(abs(obj.matrix_world[r][c] - rows[r][c]) for r in range(4) for c in range(4)))
            imported.append({'key': mesh['key'], 'object_id': mesh['object_id'],
                             'matrix_world': [obj.matrix_world[r][c] for c in range(4) for r in range(4)]})
        bpy.context.view_layer.update()
        for mesh in frame['meshes']:
            for probe in mesh['probes']:
                projected = world_to_camera_view(scene, cam, Vector(probe['world']))
                observed = (projected.x * 2 - 1, projected.y * 2 - 1)
                max_projection_error = max(max_projection_error, *(abs(a - b) for a, b in zip(observed, probe['ndc'])))
        report = {'t': frame['t'], 'requested_t': frame['requested_t'], 'meshes': imported,
                  'max_matrix_error': max_matrix_error, 'max_projection_ndc_error': max_projection_error,
                  'max_vertex_error': max_vertex_error,
                  'camera_matrix_world': [cam.matrix_world[r][c] for c in range(4) for r in range(4)],
                  'rgb': f'blender_{frame_no:04d}.png'}
        reports.append(report)
        if max_matrix_error > 1e-5 or max_vertex_error > 1e-5 or max_projection_error > 1e-4:
            (out / 'failed_parity.json').write_text(json.dumps(report, indent=2))
            raise ValueError('state/camera parity failed; rendering stopped')
        scene.render.filepath = str(out / report['rgb'])
        bpy.ops.render.render(write_still=True)
    result = {'ok': True, 'blender_version': bpy.app.version_string, 'device': 'CPU',
              'packet_sha256': hashlib.sha256(raw).hexdigest(), 'frames': reports,
              'limitations': ['shared states, not independent dynamics', 'lighting and shading are not matched',
                              'RGB only; no cross-engine depth/ID metrics']}
    (out / 'report.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    args = sys.argv[sys.argv.index('--') + 1:]
    if len(args) != 2:
        raise SystemExit('expected PACKET NEW_OUT after --')
    run(Path(args[0]).resolve(), Path(args[1]).resolve())
