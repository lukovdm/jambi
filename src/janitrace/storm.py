"""Optional helpers for checking janitrace models with Storm (needs stormpy)."""
import json

from .core import to_jani


def build(model_cls, properties=None, simplify=False):
    """Compile, parse and build the model with Storm's sparse engine.

    properties defaults to the model's @properties functions.
    Returns (storm model, list of stormpy properties in the order given)."""
    import stormpy
    j = to_jani(model_cls, properties, simplify=simplify)
    jani, props = stormpy.parse_jani_model_from_string(json.dumps(j, ensure_ascii=False))
    return stormpy.build_model(jani, props), props


def check(model_cls, properties=None, simplify=False):
    """Model-check every property; returns {name: value in the initial state}.

    properties defaults to the model's @properties functions."""
    import stormpy
    if properties is None:
        properties = model_cls.all_properties()
    model, props = build(model_cls, properties, simplify)
    init = model.initial_states[0]
    values = {}
    for name, prop in zip(properties, props):
        result = stormpy.model_checking(model, prop)
        values[name] = result.get_value(init) if hasattr(result, "get_value") else result.at(init)
    return values
