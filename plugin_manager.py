import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PLUGIN_DIR = BASE_DIR / "plugins"
MOD_DIR = BASE_DIR / "mods"

PLUGIN_REGISTRY: Dict[str, Dict[str, Any]] = {}
MOD_REGISTRY: Dict[str, Dict[str, Any]] = {}


def ensure_extension_dirs() -> None:
    PLUGIN_DIR.mkdir(exist_ok=True)
    MOD_DIR.mkdir(exist_ok=True)
    (PLUGIN_DIR / "__init__.py").touch(exist_ok=True)
    (MOD_DIR / "__init__.py").touch(exist_ok=True)


def _load_module(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    name = path.stem
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            return None, f"Unable to load spec for {path}"
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module, None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error(f"Failed to execute module {path}: {error}")
            return None, error
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.error(f"Failed to load module {path}: {error}")
        return None, error


def load_extensions(path: Path) -> Tuple[List[Any], List[Dict[str, str]]]:
    extensions = []
    errors = []
    if not path.exists():
        return extensions, errors
    for path_item in sorted(path.glob("*.py")):
        if path_item.name.startswith("_"):
            continue
        module, error = _load_module(path_item)
        if module is not None:
            extensions.append(module)
        else:
            errors.append({"file": path_item.name, "error": error or "Unknown error"})
    return extensions, errors


def load_plugins() -> Tuple[List[Any], List[Dict[str, str]]]:
    return load_extensions(PLUGIN_DIR)


def load_mods() -> Tuple[List[Any], List[Dict[str, str]]]:
    return load_extensions(MOD_DIR)


def plugin_info() -> Dict[str, Any]:
    plugins, plugin_errors = load_plugins()
    mods, mod_errors = load_mods()
    summary = {
        "plugins": [],
        "mods": [],
        "errors": {"plugins": plugin_errors, "mods": mod_errors},
    }
    for module in plugins:
        summary["plugins"].append({
            "name": getattr(module, "name", module.__name__),
            "description": getattr(module, "description", ""),
            "filename": getattr(module, "__file__", ""),
        })
    for module in mods:
        summary["mods"].append({
            "name": getattr(module, "name", module.__name__),
            "description": getattr(module, "description", ""),
            "filename": getattr(module, "__file__", ""),
        })
    return summary


def register_plugins(app: Any, state: Dict[str, Any], training_state: Dict[str, Any]) -> None:
    ensure_extension_dirs()
    plugins, load_errors = load_plugins()
    
    for module in plugins:
        plugin_name = getattr(module, "name", module.__name__)
        
        if hasattr(module, "register_routes") and callable(module.register_routes):
            try:
                module.register_routes(app, state, training_state)
                PLUGIN_REGISTRY[plugin_name] = {
                    "name": plugin_name,
                    "status": "registered",
                    "routes": True,
                }
                logger.info(f"Plugin {plugin_name} registered routes successfully.")
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.error(f"Plugin {plugin_name} failed to register routes: {error_msg}")
                PLUGIN_REGISTRY[plugin_name] = {
                    "name": plugin_name,
                    "status": "failed",
                    "error": error_msg,
                }
        
        if hasattr(module, "on_startup") and callable(module.on_startup):
            try:
                async def plugin_startup(module=module, plugin_name=plugin_name):
                    try:
                        result = module.on_startup(app, state, training_state)
                        if hasattr(result, '__await__'):
                            await result
                        logger.info(f"Plugin {plugin_name} startup completed.")
                    except Exception as exc:
                        logger.error(f"Plugin {plugin_name} startup failed: {exc}")

                if hasattr(app, 'on_event'):
                    app.on_event("startup")(plugin_startup)
                else:
                    logger.warning(f"FastAPI version does not support event handlers for plugin {plugin_name}")
            except Exception as exc:
                logger.error(f"Plugin {plugin_name} failed to register startup handler: {exc}")
    
    if load_errors:
        logger.warning(f"Plugin load errors: {load_errors}")


def apply_mods() -> Dict[str, Any]:
    ensure_extension_dirs()
    mods, load_errors = load_mods()
    results = {"mods": [], "errors": load_errors}
    
    for module in mods:
        mod_name = getattr(module, "name", module.__name__)
        
        if hasattr(module, "apply_mod") and callable(module.apply_mod):
            try:
                status = module.apply_mod()
                results["mods"].append({
                    "name": mod_name,
                    "status": "applied",
                    "result": str(status),
                })
                MOD_REGISTRY[mod_name] = {
                    "name": mod_name,
                    "status": "applied",
                    "result": status,
                }
                logger.info(f"Mod {mod_name} applied successfully.")
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                results["errors"].append({"file": getattr(module, "__file__", ""), "error": error_msg})
                MOD_REGISTRY[mod_name] = {
                    "name": mod_name,
                    "status": "failed",
                    "error": error_msg,
                }
                logger.error(f"Mod {mod_name} failed: {error_msg}")
    
    return results


def get_plugin_status() -> Dict[str, Any]:
    return {
        "plugins": PLUGIN_REGISTRY,
        "mods": MOD_REGISTRY,
    }


def call_hook(hook_name: str, *args: Any, **kwargs: Any) -> List[Tuple[str, Any]]:
    results = []
    plugins, _ = load_plugins()
    for module in plugins:
        if hasattr(module, f"on_{hook_name}"):
            hook_func = getattr(module, f"on_{hook_name}")
            if callable(hook_func):
                try:
                    result = hook_func(*args, **kwargs)
                    results.append((getattr(module, "name", module.__name__), result))
                except Exception as exc:
                    logger.error(f"Hook {hook_name} failed in plugin {getattr(module, 'name', module.__name__)}: {exc}")
    return results


def reload_extensions() -> Dict[str, Any]:
    ensure_extension_dirs()
    PLUGIN_REGISTRY.clear()
    MOD_REGISTRY.clear()

    def _clear_module_cache(folder: Path) -> None:
        for module_name, module_obj in list(sys.modules.items()):
            module_file = getattr(module_obj, '__file__', None)
            if module_file is None:
                continue
            try:
                module_path = Path(module_file).resolve()
            except Exception:
                continue
            if module_path.parent == folder:
                del sys.modules[module_name]

    _clear_module_cache(PLUGIN_DIR)
    _clear_module_cache(MOD_DIR)

    plugins, plugin_errors = load_plugins()
    mods, mod_errors = load_mods()
    return {
        "plugins_loaded": len(plugins),
        "mods_loaded": len(mods),
        "errors": {
            "plugins": plugin_errors,
            "mods": mod_errors,
        },
    }
