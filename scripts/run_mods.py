import json

import plugin_manager


if __name__ == "__main__":
    results = plugin_manager.apply_mods()
    print("Mod results:")
    print(json.dumps(results, indent=2))
