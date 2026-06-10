import json
from pathlib import Path

import plugin_manager


if __name__ == "__main__":
    summary = plugin_manager.plugin_info()
    print("Plugins:")
    for item in summary["plugins"]:
        print(f" - {item['name']}: {item['description']}")
    print("\nMods:")
    for item in summary["mods"]:
        print(f" - {item['name']}: {item['description']}")
    if summary["errors"]["plugins"] or summary["errors"]["mods"]:
        print("\nErrors:")
        print(json.dumps(summary["errors"], indent=2))
