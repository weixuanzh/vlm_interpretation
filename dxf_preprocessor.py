from typing import Iterable, Iterator, Tuple, Dict, Set
import ezdxf


PRIMITIVE_TYPES = {
    "LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE",
    "LWPOLYLINE", "POLYLINE",
    "SOLID", "POINT", "TEXT"
}

COMPLEX_TYPES = {"INSERT", "DIMENSION", "HATCH", "LEADER", "MLEADER"}


def _split_prims_complex(children, debug=False):
    prims, complexes = [], []
    for c in children:
        entity_type = c.dxftype()
        if entity_type in PRIMITIVE_TYPES:
            prims.append(c)
        elif entity_type in COMPLEX_TYPES:
            prims.append(c)
        else:
            if debug:
                print("Unknown dxftype encountered:", entity_type)
            continue
    return prims, complexes

# for INSERT
def expand_insert(ins):

    if not hasattr(ins, "virtual_entities"):
        return [], []
    return _split_prims_complex(ins.virtual_entities())

# For DIMENSION
def expand_dimension(msp, dim):
    block_name = getattr(dim.dxf, "block_name", None)
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
    else:
        primitives, complexes = [], []
        if debug:
            print("Unknown dxftype encountered:", entity_type)

    out.extend(primitives)
    for e in complexes:
        expand_entity(msp, e, out)
    
    
def convert_to_primitives(doc):
    msp = doc.modelspace()
    out = []
    for entity in msp:
        expand_entity(msp, entity, out)


    