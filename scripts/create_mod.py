import sys
from pathlib import Path

import plugin_manager


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/create_mod.py <mod_name>")
        return
    mod_name = sys.argv[1].strip()
    if not mod_name.isidentifier():
        print("Mod name must be a valid Python identifier.")
        return
    plugin_manager.ensure_extension_dirs()
    path = plugin_manager.MOD_DIR / f"{mod_name}.py"
    if path.exists():
        print(f"Mod already exists: {path}")
        return
    content = f'''name = "{mod_name}"
description = "Custom mod module for modifying application code, UI, or behavior."


def apply_mod():
    message = "{mod_name} applied successfully."
    print(message)
    return message
'''
    path.write_text(content, encoding='utf-8')
    print(f"Created mod: {path}")


if __name__ == "__main__":
    main()
