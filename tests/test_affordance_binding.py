"""Geometry and explicit-slot regressions; no browser or model dependency."""
import copy
import pytest
from gwm.binding.affordance import _extent, classify
from gwm.binding.platformer import bind
from gwm.config import load_config

def node(geom=None, oid="target"):
    return {"id": oid, "class": "generic", "geom": geom or {"kind":"primitive","shape":"box","extent":[.5,.06,.5]},
            "pose":{"pos":[0,2,0],"quat":[0,0,0,1]}}

def scene(objects):
    return {"static":[{"id":"floor","class":"floor","geom":{"kind":"primitive","shape":"box","extent":[24,.2,24]},
                       "pose":{"pos":[0,-.1,0]}}], "objects": objects}

@pytest.mark.parametrize("axis, expected", [("x",[4,.3,.3]),("y",[.3,4,.3]),("z",[.3,.3,4])])
def test_cylinder_height_follows_axis(axis, expected):
    o=node({"kind":"primitive","shape":"cylinder","radius":.15,"height":4,"axis":axis})
    assert _extent(o)==pytest.approx(expected)
    assert classify(scene([o]))["target"]["role"]!="collectible"

@pytest.mark.parametrize("geom, expected", [
    ({"kind":"primitive","shape":"cylinder","radius_top":0,"radius_bottom":.6,"height":4},[1.2,4,1.2]),
    ({"kind":"primitive","shape":"cone","radius":.15,"height":4},[.3,4,.3]),
    ({"kind":"primitive","shape":"sphere","radius":.25},[.5,.5,.5]),
    ({"kind":"primitive","shape":"plane","size":[2,3]},[2,.1,3]),
    ({"kind":"heightfield","scale":[4,1,5]},[4,1,5]),
])
def test_declared_shape_dimensions(geom, expected):
    assert _extent(node(geom))==pytest.approx(expected)

@pytest.mark.parametrize("geom", [{"kind":"primitive","shape":"box"},
    {"kind":"primitive","shape":"cylinder","radius":.2},
    {"kind":"primitive","shape":"box","extent":[float("nan"),1,1]},
    {"kind":"primitive","shape":"box","extent":[0,1,1]}])
def test_unknown_dimensions_do_not_invent_small_collectible(geom):
    o=node(geom); assert _extent(o) is None
    assert classify(scene([o]))["target"]["role"]=="decoration"

def test_tall_cylinder_does_not_gain_collection_event():
    o=node({"kind":"primitive","shape":"cylinder","radius":.15,"height":4})
    result=bind(scene([o]),None,load_config())
    assert result["binding"]["slots"]["collectibles"]==[]
    assert result["objects"][0].get("events",[])==[]

def test_primitive_parameters_take_precedence_over_unused_extent():
    o=node({"kind":"primitive","shape":"cylinder","radius":.15,"height":4,"extent":[.3,.3,.3]})
    assert _extent(o)==pytest.approx([.3,4,.3])

def test_empty_override_prevents_new_events():
    p=scene([node()]);p["binding"]={"template":"platformer_3p","slots":{"collectibles":[]}}
    result=bind(p,None,load_config())
    assert result["binding"]["slots"]["collectibles"]==[]
    assert result["objects"][0].get("events",[])==[]

def test_explicit_subset_controls_new_events():
    p=scene([node(oid="first"),node(oid="second")])
    p["binding"]={"template":"platformer_3p","slots":{"collectibles":["first"]}}
    result=bind(p,None,load_config())
    assert result["objects"][0]["events"]==[{"type":"despawn_on_contact","with":"player"}]
    assert result["objects"][1].get("events",[])==[]

def test_existing_authored_event_is_preserved():
    o=node();o["events"]=[{"type":"despawn_on_contact","with":"player"}]
    p=scene([o]);p["binding"]={"template":"platformer_3p","slots":{"collectibles":[]}}
    result=bind(copy.deepcopy(p),None,load_config())
    assert result["objects"][0]["events"]==o["events"]

def test_missing_slot_infers_event_once():
    result=bind(scene([node()]),None,load_config())
    result=bind(result,None,load_config())
    assert result["binding"]["slots"]["collectibles"]==["target"]
    assert result["objects"][0]["events"]==[{"type":"despawn_on_contact","with":"player"}]
