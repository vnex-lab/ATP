name = "example_plugin"
description = "Example plugin for adding one API route and startup behavior."


def register_routes(app, state, training_state):
    """Register custom API routes for this plugin."""
    @app.get("/api/plugins/example")
    async def example_plugin_route():
        return {
            "plugin": name,
            "description": description,
            "state_has_model": state.get("model") is not None,
        }


def on_startup(app, state, training_state):
    """Called when the application starts up."""
    state["plugin_example_initialized"] = True
    return f"{name} initialized successfully."
