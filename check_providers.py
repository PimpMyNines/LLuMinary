"""
Check which providers are properly registered in the system.
"""

import importlib


def check_provider_registration():
    """Check if providers are properly registered."""

    # Import the router module
    try:
        router = importlib.import_module("lluminary.models.router")
        print("Successfully imported router module")
    except ImportError as e:
        print(f"Failed to import router module: {e}")
        return

    # Check available providers
    try:
        # Check PROVIDER_REGISTRY constant we found
        if hasattr(router, "PROVIDER_REGISTRY"):
            print(f"Provider registry keys: {list(router.PROVIDER_REGISTRY.keys())}")
        else:
            print("No PROVIDER_REGISTRY found")

        # Check MODEL_MAPPINGS constant we found
        if hasattr(router, "MODEL_MAPPINGS"):
            print(f"Model mappings keys: {list(router.MODEL_MAPPINGS.keys())}")
        else:
            print("No MODEL_MAPPINGS found")

        # Check available models using list_available_models function we found
        if hasattr(router, "list_available_models"):
            models = router.list_available_models()
            print(f"Available models: {models}")
        else:
            print("No list_available_models function found")

    except Exception as e:
        print(f"Error checking providers: {e}")


if __name__ == "__main__":
    check_provider_registration()

    # If we get here, let's try to inspect the router module
    try:
        import inspect

        from lluminary.models import router

        print("\nRouter module contents:")
        for name, obj in inspect.getmembers(router):
            if not name.startswith("_"):  # Skip private members
                print(f"  {name}: {type(obj).__name__}")
    except Exception as e:
        print(f"Failed to inspect router module: {e}")
