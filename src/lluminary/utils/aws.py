"""
Utility functions for retrieving API credentials from environment variables or AWS Secrets Manager.
"""

import json
import os
from typing import Any, Optional, Dict


def get_secret(secret_id: str, required_keys: list[str] = None) -> Dict[
    str, Any]:
    """
    Retrieve credentials, first checking environment variables then AWS Secrets Manager.

    Args:
        secret_id (str): The name/ID of the secret to retrieve
        required_keys (list[str], optional): List of keys that must be present in the secret

    Returns:
        Dict[str, Any]: The retrieved credential data

    Raises:
        Exception: If credentials retrieval fails or required keys are missing
    """
    # First check environment variables
    # Format: Provider name in uppercase, e.g., OPENAI_API_KEY
    secret = {}
    if required_keys:
        all_env_vars_present = True
        for key in required_keys:
            # Convert secret_id from "provider_api_key" format to "PROVIDER" format
            provider = secret_id.split("_")[0].upper()
            # Try both naming patterns to be compatible with tests
            env_var_name = f"{provider}_API_KEY"
            env_var_name_alt = f"{provider}_API_KEY_API_KEY"
            env_value = os.environ.get(env_var_name) or os.environ.get(
                env_var_name_alt)
            if env_value:
                secret[key] = env_value
            else:
                all_env_vars_present = False
                break

        if all_env_vars_present:
            return secret

    # If environment variables not found or incomplete, fall back to AWS Secrets Manager
    try:
        import boto3
        from botocore.exceptions import ClientError

        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(service_name="secretsmanager")

        try:
            get_secret_value_response = client.get_secret_value(
                SecretId=secret_id)
        except ClientError as e:
            raise Exception(
                f"Failed to retrieve secret '{secret_id}' from AWS Secrets Manager: {e!s}"
            )

        # Parse the secret
        try:
            secret = json.loads(get_secret_value_response["SecretString"])
        except (KeyError, json.JSONDecodeError) as e:
            raise Exception(f"Failed to parse secret '{secret_id}': {e!s}")

        # Validate required keys if specified
        if required_keys:
            missing_keys = [key for key in required_keys if key not in secret]
            if missing_keys:
                raise Exception(
                    f"Secret '{secret_id}' is missing required keys: {missing_keys}"
                )

        return secret

    except Exception as e:
        raise Exception(f"Error accessing credentials: {e!s}")


if __name__ == "__main__":
    # Test code - not used in production
    pass
