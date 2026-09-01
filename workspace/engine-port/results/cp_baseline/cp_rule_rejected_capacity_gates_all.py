"""REJECTED rule variant, kept as evidence -- do not pre-register.  Tracks rev2.

The tempting simplification: "the canonical HI phase is above achieved capacity, so
threshold goodput is ill-posed there (PROJECT_STATUS 'methodology gate' #6, 'measure
capacity first; stay off the metric cliff') -> return NO_VERDICT."  It reads as the
conservative choice.  Run `design_reachability.py` on it and it is
NOTHING_PURCHASABLE: the whole campaign is decided before any datum arrives.

Same shape as NSL D4 ("removing the cap axis pre-decides the answer") and the
design-layer form of PROJECT_STATUS "methodology gate" #40 ("the decision quantity
can itself be an identity").  `cp_rule.py` keeps the capacity axis but lets it gate
the SIZE claim only, because canon cites the HE0 RANKING from exactly this operating
point (CONSENSUS §1-7).
"""
import importlib.util as _il, os as _os

_spec = _il.spec_from_file_location(
    "_base", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "cp_rule.py"))
_base = _il.module_from_spec(_spec)
_spec.loader.exec_module(_base)

RULE_REV = "rejected-2"
AXES = _base.AXES
AXIS_ORDER = _base.AXIS_ORDER
SUBSTANTIVE = _base.SUBSTANTIVE


def label(w):
    if _base.incoherent(w):
        return "IMPOSSIBLE_WORLD"
    if w["capacity"] == "above":
        return "CLIFF_ILLPOSED"          # <- swallows every world on the canonical trace
    return _base.label(w)
