import sys
from pathlib import Path

import plugin_manager


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/create_plugin.py <plugin_name>")
        return
    plugin_name = sys.argv[1].strip()
    if not plugin_name.isidentifier():
        print("Plugin name must be a valid Python identifier.")
        return
    plugin_manager.ensure_extension_dirs()
    path = plugin_manager.PLUGIN_DIR / f"{plugin_name}.py"
    if path.exists():
        print(f"Plugin already exists: {path}")
        return
    content = f'''name = "{plugin_name}"
description = "Custom plugin module for extending the application."


def register_routes(app, state, training_state):
    @app.get("/api/plugins/{plugin_name}")
    async def {plugin_name}_route():
        return {{"plugin": name, "active": True}}


def on_startup(app, state, training_state):
    state["{plugin_name}_loaded"] = True
'''
    path.write_text(content, encoding='utf-8')
    print(f"Created plugin: {path}")


if __name__ == "__main__":
    main()
