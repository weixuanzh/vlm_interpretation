from typing import Iterable, Iterator, Tuple, Dict, Set
import ezdxf
from ezdxf.addons import importer

# a dxf will be exploded into a list of those entities
PRIMITIVE_TYPES = {
    "LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE",
    "SOLID", "POINT"
}
# complex entities are exploded into primitive ones
COMPLEX_TYPES = {"INSERT", "DIMENSION", "HATCH", "LEADER", "MLEADER", "LWPOLYLINE", "POLYLINE"}


def _split_prims_complex(children):
    prims, complexes = [], []
    for c in children:
        entity_type = c.dxftype()
        if entity_type in PRIMITIVE_TYPES:
            prims.append(c)
        else:
            complexes.append(c)
    return prims, complexes

# for LWPOLYLINE/POLYLINE
def expand_polyline(pl):
    if not hasattr(pl, "virtual_entities"):
        return [], []

    try:
        segs = list(pl.virtual_entities())
    except Exception:
        return [], []
    return _split_prims_complex(segs)

# for INSERT
def expand_insert(ins):
    if not hasattr(ins, "virtual_entities"):
        return [], []
    return _split_prims_complex(ins.virtual_entities())

# For DIMENSION
def expand_dimension(msp, dim):
    block_name = getattr(dim.dxf, "geometry", None)
    if block_name is not None:
        # search for inserts that match the dimension
        inserts = [ins for ins in msp.query("INSERT") if getattr(ins.dxf, "name", None) == block_name]
        # found corresponding inserts, expand them
        if inserts:
            children = []
            for ins in inserts:
                if hasattr(ins, "virtual_entities"):
                    children.extend(list(ins.virtual_entities()))
            return _split_prims_complex(children)
        else:
            # no corresponding INSERT is found, generate entities
            if hasattr(dim, "virtual_entities"):
                return _split_prims_complex(dim.virtual_entities())
            
    return [], []

# for HATCH
def expand_hatch(hatch):
    if hasattr(hatch, "virtual_entities"):
        return _split_prims_complex(hatch.virtual_entities())
    return [], []

# for LEADER or MLEADER
def expand_leader_like(entity):
    if hasattr(entity, "virtual_entities"):
        return _split_prims_complex(entity.virtual_entities())
    return [], []

# recursively expand, until there are no more complex entities
def expand_entity(msp, entity, out, debug=False):
    entity_type = entity.dxftype()
    if entity_type in PRIMITIVE_TYPES:
        primitives, complexes = [entity], []
    elif entity_type == "DIMENSION":
        primitives, complexes = expand_dimension(msp, entity)
    elif entity_type == "HATCH":
        primitives, complexes = expand_hatch(entity)
    elif entity_type == "INSERT":
        primitives, complexes = expand_insert(entity)
    elif entity_type in ["LEADER", "MLEADER"]:
        primitives, complexes = expand_leader_like(entity)
    elif entity_type in ["POLYLINE", "LWPOLYLINE"]:
        primitives, complexes = expand_polyline(entity)
    else:
        primitives, complexes = [], []
        if debug:
            print("Unknown dxftype encountered:", entity_type)

    out.extend(primitives)
    for e in complexes:
        expand_entity(msp, e, out)
    
def convert_to_primitives(doc, debug=False):
    msp = doc.modelspace()
    out = []
    for entity in msp:
        expand_entity(msp, entity, out, debug=debug)
    return out

# function to verify if the extracted primitives actually make sense
def write_dxf_from_primitives(primitives, out_path, dxfversion="R2018"):
    # Make a clean target doc with standard resources preinstalled
    target_doc = ezdxf.new(dxfversion, setup=True)
    target_msp = target_doc.modelspace()

    # Importer automatically brings over layers, linetypes, text styles, etc.
    imp = importer.Importer(primitives[0].doc, target_doc)

    # Put everything into the target modelspace
    imp.import_entities(primitives, target_msp)
    imp.finalize()  # important: copies all remaining required table entries

    # Optional: run an audit to catch broken refs before saving
    auditor = target_doc.audit()
    if auditor.has_errors:
        print("Audit errors:", len(auditor.errors))
    if auditor.has_fixes:
        print("Audit fixes:", len(auditor.fixes))

    target_doc.saveas(out_path)
    return out_path

# dict[str: int], where str is the text description of what that argument is (like "x_coord")
# int is its id, indicative of different embeddings in tokenizenzation process
ARGUMENT_MAP = {"X_COORD": 1, "Y_COORD": 2, "RADIUS": 3, "ANGLE": 4, "MAJOR_AXIS": 5, "RATIO": 6}
# same here, text description of primitive type -> id
PRIMITIVE_MAP = {"LINE": 1, "ARC": 2, "ELLIPSE": 3}

# safe helper function to get fields from a dxf entity
def _get_dxf_fields(entity, fields):
    dxf = getattr(entity, "dxf", None)
    if dxf is None:
        return False, {}
    out = {}
    for f in fields:
        val = getattr(dxf, f, None)
        if val is None:
            return False, {}
        out[f] = val
    return True, out

# get attributes from primitive entities
# gives status, argument, argument_type_id, primitive_type_id tuple
# if unexpected happen, status is 1
def line_to_attr(line):
    # get fields with check
    status, fields = _get_dxf_fields(line, ["start", "end"])
    if not status:
        return 1, [], [], -100
    
    # build arg and arg_types
    args = [*(fields["start"][:2]), *(fields["end"][:2])]
    arg_types = [ARGUMENT_MAP.get("X_COORD"), ARGUMENT_MAP.get("Y_COORD"), 
                 ARGUMENT_MAP.get("X_COORD"), ARGUMENT_MAP.get("Y_COORD")]
    return 0, args, arg_types, PRIMITIVE_MAP.get("LINE")

def arc_to_attr(arc):    
    # get fields with check
    status, fields = _get_dxf_fields(arc, ["center", "radius", "start_angle", "end_angle"])
    if not status:
        return 1, [], [], -100
    
    # build outputs
    args = [*(fields["center"][:2]), fields["radius"], fields["start_angle"], fields["end_angle"]]
    arg_types = [ARGUMENT_MAP.get("X_COORD"), ARGUMENT_MAP.get("Y_COORD"),
                 ARGUMENT_MAP.get("RADIUS"),
                 ARGUMENT_MAP.get("ANGLE"),
                 ARGUMENT_MAP.get("ANGLE")]  
    return 0, args, arg_types, PRIMITIVE_MAP.get("ARC")

def ellipse_to_attr(ellipse):
    # get fields with check
    status, fields = _get_dxf_fields(ellipse, ["center", "major_axis", "ratio", "start_param", "end_param"])
    if not status:
        return 1, [], [], -100
    
    # build outputs
    args = [*(fields["center"][:2]), fields["major_axis"], fields["ratio"], fields["start_param"], fields["end_param"]]
    arg_types = [ARGUMENT_MAP.get("X_COORD"), ARGUMENT_MAP.get("Y_COORD"),
                 ARGUMENT_MAP.get("MAJOR_AXIS"),
                 ARGUMENT_MAP.get("RATIO"),
                 ARGUMENT_MAP.get("ANGLE"),
                 ARGUMENT_MAP.get("ANGLE")]  
    return 0, args, arg_types, PRIMITIVE_MAP.get("ELLIPSE")

def spline_to_attr(spline):
    return 1, [], [], -100
    # get fields with check
    status, fields = _get_dxf_fields(spline, ["center", "major_axis", "ratio", "start_param", "end_param"])
    if not status:
        return 1, [], [], -100
    
    # build outputs
    args = []
    arg_types = []
    return 0, args, arg_types, PRIMITIVE_MAP.get("LINE")

def polyline_to_attr(polyline):
    return 1, [], [], -100
    args = []
    arg_types = []
    return 0, args, arg_types, PRIMITIVE_MAP.get("LINE")

def solid_to_attr(solid):
    args = []
    arg_types = []
    return 0, args, arg_types, PRIMITIVE_MAP.get("LINE")

def point_to_attr(point):
    args = []
    arg_types = []
    return 0, args, arg_types, PRIMITIVE_MAP.get("LINE")

def text_to_attr(text):
    args = []
    arg_types = []
    return 0, args, arg_types, PRIMITIVE_MAP.get("LINE")

# convert the attribute returned by previous functions to a tensor
# 3 tensors: (arg_vec_length, 1) for normalized arguments
#            (arg_vec_length, num_arg_types) for argument types (one hot vector)
#            (1, num_prim_types) for primitive type (one hot vector)
def attr_to_tensor(attr, arg_vec_length, num_arg_types, num_prim_types):
    return 1

def primitives_to_tensor(primitives):

    for primitive in primitives:
        print(1)
    return 1

    