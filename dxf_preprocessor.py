from typing import Iterable, Iterator, Tuple, Dict, Set
import ezdxf
from ezdxf.addons import importer
import torch

# a dxf will be exploded into a list of those entities
PRIMITIVE_TYPES = {
    "LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE",
    "SOLID", "POINT", "TEXT", "MTEXT"
}
# complex entities are exploded into primitive ones
COMPLEX_TYPES = {"INSERT", "DIMENSION", "HATCH", "LEADER", "MLEADER", "LWPOLYLINE", "POLYLINE"}

# dict[str: int], where str is the text description of what that argument is (like "x_coord")
# int is its id, indicative of different embeddings in tokenizenzation process
ARGUMENT_MAP = {"EMPTY": 0, "X_COORD": 1, "Y_COORD": 2, "RADIUS": 3, "ANGLE": 4, "MAJOR_AXIS": 5, "MINOR_AXIS": 6}
# same here, text description of primitive type -> id
PRIMITIVE_MAP = {"ERROR": 0, "LINE": 1, "ARC": 2, "ELLIPSE": 3}

def _split_prims_complex(children):
    prims, prims_text, complexes = [], [], []
    for c in children:
        entity_type = c.dxftype()
        if entity_type in PRIMITIVE_TYPES:
            if entity_type in ["TEXT", "MTEXT"]:
                prims_text.append(c)
            else:
                prims.append(c)
        else:
            complexes.append(c)
    return prims, prims_text, complexes

# for LWPOLYLINE/POLYLINE
def expand_polyline(pl):
    if not hasattr(pl, "virtual_entities"):
        return [], [], []

    try:
        segs = list(pl.virtual_entities())
    except Exception:
        return [], [], []
    return _split_prims_complex(segs)

# for INSERT
def expand_insert(ins):
    if not hasattr(ins, "virtual_entities"):
        return [], [], []
    return _split_prims_complex(ins.virtual_entities())

# For DIMENSION
# there are two cases: either a raw dimension, or a dimension that already has an INSERT corresponding to it
# prefer expanding the INSERT if possible, otherwise generate entities from the dimension itself
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
            
    return [], [], []

# for HATCH
def expand_hatch(hatch):
    if hasattr(hatch, "virtual_entities"):
        return _split_prims_complex(hatch.virtual_entities())
    return [], [], []

# for LEADER or MLEADER
def expand_leader_like(entity):
    if hasattr(entity, "virtual_entities"):
        return _split_prims_complex(entity.virtual_entities())
    return [], [], []

# recursively expand, until there are no more complex entities
def expand_entity(msp, entity, out, out_text, debug=False):
    entity_type = entity.dxftype()
    if entity_type in PRIMITIVE_TYPES:
        if entity_type in ["TEXT", "MTEXT"]:
            primitives, primtives_text, complexes = [], [entity], []
        else:
            primitives, primtives_text, complexes = [entity], [], []
    elif entity_type == "DIMENSION":
        primitives, primtives_text, complexes = expand_dimension(msp, entity)
    elif entity_type == "HATCH":
        primitives, primtives_text, complexes = expand_hatch(entity)
    elif entity_type == "INSERT":
        primitives, primtives_text, complexes = expand_insert(entity)
    elif entity_type in ["LEADER", "MLEADER"]:
        primitives, primtives_text, complexes = expand_leader_like(entity)
    elif entity_type in ["POLYLINE", "LWPOLYLINE"]:
        primitives, primtives_text, complexes = expand_polyline(entity)
    else:
        primitives, primtives_text, complexes = [], [], []
        if debug:
            print("Unknown dxftype encountered:", entity_type)

    out.extend(primitives)
    out_text.extend(primtives_text)
    for e in complexes:
        expand_entity(msp, e, out, out_text)
    
def convert_to_primitives(doc, debug=False):
    msp = doc.modelspace()
    # store text and other primitives (line, circle, etc.) separately
    # they go through different encoding processes
    out = []
    out_text = []
    for entity in msp:
        expand_entity(msp, entity, out, out_text, debug=debug)
    return out, out_text

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
        return 1, [], [], PRIMITIVE_MAP.get("ERROR")

    # build arg and arg_types
    args = [fields["start"][0], fields["start"][1],
            fields["end"][0], fields["end"][1]]
    arg_types = [ARGUMENT_MAP.get("X_COORD"), ARGUMENT_MAP.get("Y_COORD"), 
                 ARGUMENT_MAP.get("X_COORD"), ARGUMENT_MAP.get("Y_COORD")]
    return 0, args, arg_types, PRIMITIVE_MAP.get("LINE")

def arc_to_attr(arc):    
    # get fields with check
    status, fields = _get_dxf_fields(arc, ["center", "radius", "start_angle", "end_angle"])
    if not status:
        return 1, [], [], PRIMITIVE_MAP.get("ERROR")
    
    # build outputs
    args = [fields["center"][0], fields["center"][1],
            fields["radius"],
            fields["start_angle"],
            fields["end_angle"]]
    arg_types = [ARGUMENT_MAP.get("X_COORD"), ARGUMENT_MAP.get("Y_COORD"),
                 ARGUMENT_MAP.get("RADIUS"),
                 ARGUMENT_MAP.get("ANGLE"),
                 ARGUMENT_MAP.get("ANGLE")]  
    return 0, args, arg_types, PRIMITIVE_MAP.get("ARC")

def ellipse_to_attr(ellipse):
    # get fields with check
    status, fields = _get_dxf_fields(ellipse, ["center", "major_axis", "ratio", "start_param", "end_param"])
    if not status:
        return 1, [], [], PRIMITIVE_MAP.get("ERROR")
    
    # build outputs
    args = [fields["center"][0], fields["center"][1],
            abs(fields["major_axis"][0]),
            abs(fields["major_axis"][0]) * fields["ratio"],
            fields["start_param"],
            fields["end_param"]]
    arg_types = [ARGUMENT_MAP.get("X_COORD"), ARGUMENT_MAP.get("Y_COORD"),
                 ARGUMENT_MAP.get("MAJOR_AXIS"),
                 ARGUMENT_MAP.get("MINOR_AXIS"),
                 ARGUMENT_MAP.get("ANGLE"),
                 ARGUMENT_MAP.get("ANGLE")]  
    return 0, args, arg_types, PRIMITIVE_MAP.get("ELLIPSE")

# not implemented yet
def spline_to_attr(spline):
    return 1, [], [], PRIMITIVE_MAP.get("ERROR")
    # get fields with check
    status, fields = _get_dxf_fields(spline, ["center", "major_axis", "ratio", "start_param", "end_param"])
    if not status:
        return 1, [], [], -100
    
    # build outputs
    args = []
    arg_types = []
    return 0, args, arg_types, PRIMITIVE_MAP.get("LINE")

def solid_to_attr(solid):
    return 1, [], [], PRIMITIVE_MAP.get("ERROR")
    args = []
    arg_types = []
    return 0, args, arg_types, PRIMITIVE_MAP.get("LINE")

def point_to_attr(point):
    return 1, [], [], PRIMITIVE_MAP.get("ERROR")
    args = []
    arg_types = []
    return 0, args, arg_types, PRIMITIVE_MAP.get("LINE")

# extract only text with valid dimension values
def text_to_attr(text):
    raise NotImplementedError("don't use it, text is encoded in a different way")

def text_to_tensor(text, arg_vec_length, num_arg_types, num_prim_types):
    return 1

def primitive_to_attr(primitive, debug=False):
    if not hasattr(primitive, "dxftype"):
        if debug:
            print("Primitive has no dxftype:", primitive)
        return 1, [], [], PRIMITIVE_MAP.get("ERROR")
    entity_type = primitive.dxftype()
    if entity_type == "LINE":
        return line_to_attr(primitive)
    elif entity_type == "ARC":
        return arc_to_attr(primitive)
    elif entity_type == "ELLIPSE":
        return ellipse_to_attr(primitive)
    elif entity_type == "SPLINE":
        return spline_to_attr(primitive)
    elif entity_type == "SOLID":
        return solid_to_attr(primitive)
    elif entity_type == "POINT":
        return point_to_attr(primitive)
    else:
        if debug:
            print("Unable to handle dxf type in primitive_to_attr:", primitive.dxftype())
        return 1, [], [], PRIMITIVE_MAP.get("ERROR")
    
# convert the attribute returned by previous functions to a tensor
# 3 tensors: (1, arg_vec_length) for arguments
#            (1, arg_vec_length, num_arg_types) for argument types (one hot vector)
#            (1, num_prim_types) for primitive type (one hot vector)
# the 1 is for stacking
def attr_to_tensor(args, arg_types, prim_type, arg_vec_length, num_arg_types, num_prim_types):
    # fill in args and arg_types to fixed length
    while len(args) < arg_vec_length:
        args.append(-1)
        arg_types.append(ARGUMENT_MAP.get("EMPTY"))
    args_tensor = torch.tensor(args).unsqueeze(0)
    arg_types_tensor = torch.nn.functional.one_hot(torch.tensor(arg_types), num_classes=num_arg_types).unsqueeze(0)
    # prim_type here starts at 1 ("error" primitives can't reach here), so shift down by 1
    prim_type_tensor = torch.nn.functional.one_hot(torch.tensor([prim_type - 1]), num_classes=num_prim_types)
    return args_tensor, arg_types_tensor, prim_type_tensor

def primitives_to_tensor(primitives, arg_vec_length, num_arg_types, num_prim_types, debug=False):
    args_list = []
    arg_types_list = []
    prim_types_list = []
    for primitive in primitives:
        # try converting primitive to attributes
        status, args, arg_types, prim_type = primitive_to_attr(primitive, debug=debug)
        if status:
            if debug:
                print("Skipping primitive_to_tensor function:", primitive)
            continue
        else:
            args_tensor, arg_types_tensor, prim_type_tensor = attr_to_tensor(args, arg_types, prim_type, arg_vec_length, num_arg_types, num_prim_types)
            args_list.append(args_tensor)
            arg_types_list.append(arg_types_tensor)
            prim_types_list.append(prim_type_tensor)
    return torch.cat(args_list, dim=0), torch.cat(arg_types_list, dim=0), torch.cat(prim_types_list, dim=0)

    