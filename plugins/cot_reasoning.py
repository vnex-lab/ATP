"""Legacy plugin shim — CoT reasoning is built into reasoning.py."""

import reasoning

name = "cot_reasoning"
description = "Built-in CoT reasoning (shim — use reasoning.py directly)."

register_routes = lambda app, state, training_state: None  # registered in api.py
def on_startup(app, state, training_state):
    return "CoT reasoning is built into reasoning.py"
