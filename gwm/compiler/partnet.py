"""Private single-hinge URDF -> existing Program and GLB bridge.

No dataset is redistributed by this module. Texture/material appearance and
physical joint constraints are outside this first import subset.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import posixpath
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


def _numbers(text, default, length=3):
    v = np.asarray([float(x) for x in text.split()] if text else default, dtype=float)
    if v.shape != (length,) or not np.isfinite(v).all():
        raise ValueError('invalid finite vector')
    return v


def _origin(element):
    xyz = _numbers(element.get('xyz') if element is not None else None, [0, 0, 0])
    x, y, z = _numbers(element.get('rpy') if element is not None else None, [0, 0, 0])
    cx, cy, cz = np.cos([x, y, z]); sx, sy, sz = np.sin([x, y, z])
    rx = np.array([[1,0,0], [0,cx,-sx], [0,sx,cx]])
    ry = np.array([[cy,0,sy], [0,1,0], [-sy,0,cy]])
    rz = np.array([[cz,-sz,0], [sz,cz,0], [0,0,1]])
    t = np.eye(4); t[:3,:3] = rz @ ry @ rx; t[:3,3] = xyz
    return t


def _turn(axis, angle):
    x, y, z = axis
    k = np.array([[0,-z,y], [z,0,-x], [-y,x,0]])
    r = np.eye(3) + math.sin(angle)*k + (1-math.cos(angle))*(k@k)
    t = np.eye(4); t[:3,:3] = r
    return t


def _obj(raw):
    vertices, faces = [], []
    for line in raw.decode('utf-8').splitlines():
        fields = line.split()
        if not fields or fields[0].startswith('#'):
            continue
        if fields[0] == 'v':
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == 'f':
            ids = [int(x.split('/')[0]) for x in fields[1:]]
            ids = [i-1 if i > 0 else len(vertices)+i for i in ids]
            if len(ids) < 3 or any(i < 0 or i >= len(vertices) for i in ids):
                raise ValueError('invalid OBJ face')
            faces.extend([[ids[0], ids[i], ids[i+1]] for i in range(1, len(ids)-1)])
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all() or not faces:
        raise ValueError('OBJ requires finite vertices and triangle faces')
    return v, np.asarray(faces, dtype=np.uint32)


def _glb(vertices, faces, color):
    """Encode an opaque, untextured diagnostic mesh; preserve all triangles."""
    v = np.asarray(vertices, dtype='<f4'); f = np.asarray(faces, dtype='<u4')
    normals = np.zeros_like(v)
    n = np.cross(v[f[:,1]]-v[f[:,0]], v[f[:,2]]-v[f[:,0]])
    for i in range(3):
        np.add.at(normals, f[:,i], n)
    norms = np.linalg.norm(normals, axis=1)
    normals[norms > 1e-12] /= norms[norms > 1e-12, None]
    blocks = [v.tobytes(), normals.tobytes(), f.tobytes()]
    offsets = [0, len(blocks[0]), len(blocks[0])+len(blocks[1])]
    data = b''.join(blocks)
    document = {'asset': {'version':'2.0', 'generator':'gwm-private-partnet-bridge'},
        'scene':0, 'scenes':[{'nodes':[0]}], 'nodes':[{'mesh':0}],
        'meshes':[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1}, 'indices':2, 'material':0}]}],
        'materials':[{'pbrMetallicRoughness':{'baseColorFactor':[*color,1], 'metallicFactor':0,'roughnessFactor':0.8},'doubleSided':True}],
        'buffers':[{'byteLength':len(data)}],
        'bufferViews':[{'buffer':0,'byteOffset':o,'byteLength':len(b)} for o,b in zip(offsets,blocks)],
        'accessors':[{'bufferView':0,'componentType':5126,'count':len(v),'type':'VEC3','min':v.min(axis=0).tolist(),'max':v.max(axis=0).tolist()},
                     {'bufferView':1,'componentType':5126,'count':len(v),'type':'VEC3'},
                     {'bufferView':2,'componentType':5125,'count':f.size,'type':'SCALAR'}]}
    encoded = json.dumps(document, separators=(',',':'), allow_nan=False).encode()
    encoded += b' ' * (-len(encoded)%4); data += b'\0' * (-len(data)%4)
    return (struct.pack('<4sII',b'glTF',2,12+8+len(encoded)+8+len(data))
            + struct.pack('<I4s',len(encoded),b'JSON') + encoded
            + struct.pack('<I4s',len(data),b'BIN\0') + data)


def import_single_hinge(archive_path, out_dir, *, expected_sha256, revision):
    """Return a validated existing-DSL Program plus explicit non-CC0 assets.

    Subset: one finite revolute leaf; all ancestors/other links fixed; OBJ visuals.
    The fixed root chain is baked into mesh vertices. No geometric normalization.
    """
    source = Path(archive_path)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha != expected_sha256:
        raise ValueError('source archive SHA256 mismatch')
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError('refusing to overwrite import output')
    with zipfile.ZipFile(source) as z:
        if sum(x.file_size for x in z.infolist()) > 1024**3:
            raise ValueError('archive exceeds 1 GiB uncompressed limit')
        names = z.namelist()
        urdfs = [n for n in names if n.endswith('/mobility.urdf')]
        if len(urdfs) != 1:
            raise ValueError('expected one mobility.urdf')
        urdf = urdfs[0]; prefix = posixpath.dirname(urdf)
        root = ET.fromstring(z.read(urdf))
        links = {x.get('name'): x for x in root.findall('link')}
        joints = root.findall('joint')
        if len(links) != len(root.findall('link')) or any(j.find('mimic') is not None for j in joints):
            raise ValueError('duplicate links or mimic unsupported')
        moving = [j for j in joints if j.get('type') != 'fixed']
        if len(moving) != 1 or moving[0].get('type') != 'revolute':
            raise ValueError('expected one finite revolute joint')
        hinge = moving[0]; moving_link = hinge.find('child').get('link')
        if any(j.find('parent').get('link') == moving_link for j in joints):
            raise ValueError('revolute child must be a leaf')
        by_child = {j.find('child').get('link'):j for j in joints}
        if len(by_child) != len(joints) or not set(by_child).issubset(links):
            raise ValueError('invalid tree')
        roots = set(links)-set(by_child)
        if len(roots) != 1:
            raise ValueError('expected one root')
        def transform(name, trail=()):
            if name not in links or name in trail:
                raise ValueError('invalid parent or cycle')
            if name not in by_child:
                return np.eye(4)
            j = by_child[name]
            return transform(j.find('parent').get('link'), (*trail,name)) @ _origin(j.find('origin'))
        transforms = {name:transform(name) for name in links}
        axis = _numbers(hinge.find('axis').get('xyz') if hinge.find('axis') is not None else None,[1,0,0])
        if np.linalg.norm(axis) < 1e-12:
            raise ValueError('zero axis')
        axis /= np.linalg.norm(axis)
        limit = hinge.find('limit')
        lower,upper = (float(limit.get(k)) for k in ('lower','upper')) if limit is not None else (math.nan,math.nan)
        if not math.isfinite(lower+upper) or not lower < upper:
            raise ValueError('invalid joint limits')
        joint = transforms[moving_link]
        world_axis = joint[:3,:3] @ axis; pivot = joint[:3,3]
        groups = {'body':([],[]), 'lid':([],[])}
        source_mapping = []
        for name,link in links.items():
            category = 'lid' if name == moving_link else 'body'
            vertices,faces = groups[category]
            for visual in link.findall('visual'):
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    raise ValueError('only OBJ mesh visuals supported')
                path = mesh.get('filename','')
                member = posixpath.normpath(posixpath.join(prefix,path))
                if path.startswith('/') or not member.startswith(prefix+'/') or not member.lower().endswith('.obj'):
                    raise ValueError('unsupported mesh path')
                v,f = _obj(z.read(member))
                scale = _numbers(mesh.get('scale'), [1,1,1])
                if np.any(scale <= 0):
                    raise ValueError('non-positive mesh scale unsupported')
                t = transforms[name] @ _origin(visual.find('origin'))
                v = (v*scale) @ t[:3,:3].T + t[:3,3]
                f = f + sum(len(a) for a in vertices)
                vertices.append(v); faces.append(f)
                source_mapping.append({'link':name,'program_id':category,'mesh':member})
        combined = {}
        for name,(v,f) in groups.items():
            if not v:
                raise ValueError('body and lid must contain geometry')
            combined[name] = (np.concatenate(v),np.concatenate(f))
    out.mkdir(parents=True)
    objects,requests = [],[]
    mapping = {'source_archive_sha256':sha,'repository':'sapien-sim/PartNetMobility',
               'revision':revision,'source_urdf':urdf,'source_mapping':source_mapping,
               'source_joint':{'name':hinge.get('name'),'child':moving_link,'axis':axis.tolist(),
                               'zero_matrix_world':joint.tolist(),'limits_rad':[lower,upper]},
               'appearance':'diagnostic solid colors; source textures/materials not imported',
               'units':'URDF coordinates retained; no independent physical scale measurement',
               'mesh_counts':{},'states':[]}
    for name,(v,f) in combined.items():
        mapping['mesh_counts'][name] = {'vertices':len(v),'triangles':len(f)}
        center = (v.min(axis=0)+v.max(axis=0))/2
        extent = v.max(axis=0)-v.min(axis=0)
        if np.any(extent < 1e-8):
            raise ValueError('degenerate visual bounds')
        file = out/f'{name}.glb'
        file.write_bytes(_glb(v-center,f,[0.22,0.34,0.55] if name=='body' else [0.7,0.23,0.08]))
        node = {'id':name,'class':f'partnet_{name}','geom':{'kind':'asset','query':f'PartNet {name}',
                'ref':f'partnet_{name}','extent':extent.tolist()},'pose':{'pos':center.tolist(),'quat':[0,0,0,1]},
                'interaction':{'grab':'none'}}
        if name=='lid':
            node['motion'] = {'type':'revolute','axis':world_axis.tolist(),'pivot':pivot.tolist(),
                              'from_deg':math.degrees(lower),'range_deg':[math.degrees(lower),math.degrees(upper)],
                              'schedule':[{'t':0,'to_deg':math.degrees(upper),'duration':1}]}
            lid_center = center
        objects.append(node)
        requests.append({'id':name,'resolved':'library_ref','asset_id':f'partnet_{name}',
                         'license':'PartNet-Mobility research terms; upstream ShapeNet conditions',
                         'source':str(file.resolve()),'file':f'assets/{name}.glb'})
    all_v = np.concatenate([v for v,_ in combined.values()])
    focus = (all_v.min(axis=0)+all_v.max(axis=0))/2
    radius = float(np.linalg.norm(all_v.max(axis=0)-all_v.min(axis=0)))
    program = {'meta':{'clip':'partnet_single_hinge','fps':30,'duration':1,'units':'m','up':'y',
                      'notes':'Known source asset; fixed ancestor chain baked into rest meshes; untextured diagnostic.'},
               'camera':{'intrinsics':{'fov_deg':45,'aspect':4/3,'far':max(50,radius*10)},
                         'keyframes':[{'t':0,'pos':(focus+radius*np.array([1.2,0.7,1.4])).tolist(),
                                       'look_at':focus.tolist()}]},'static':[],'objects':objects}
    for t in [0,0.5,1]:
        q = lower+(upper-lower)*t
        turn = _turn(world_axis,q)
        pos = turn[:3,:3] @ (lid_center-pivot)+pivot
        mapping['states'].append({'t':t,'q_rad':q,'lid_pos':pos.tolist(),'lid_rotation':turn[:3,:3].tolist()})
    from .validate import validate
    validation = validate(program)
    if not validation['ok']:
        raise ValueError('generated Program failed validation: '+json.dumps(validation))
    manifest = {'strategy':['explicit_source'], 'library':None, 'requests':requests}
    for name,data in [('program.json',program),('asset_manifest.json',manifest),('source_mapping.json',mapping),('validation.json',validation)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return program,manifest,mapping
