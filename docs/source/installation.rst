Installation
============

LLuMinary is available on PyPI and can be installed with pip:

.. code-block:: bash

    pip install lluminary

Requirements
-----------

LLuMinary requires Python 3.8 or later and includes the following core dependencies:

* requests>=2.31.0
* pydantic>=2.0.0
* tenacity>=8.0.0
* click>=8.0.0
* pyyaml>=6.0.0

Provider Dependencies
--------------------

Depending on which LLM providers you want to use, additional dependencies may be required:

.. code-block:: bash

    # To use OpenAI
    pip install openai>=1.12.0

    # To use Anthropic
    pip install anthropic>=0.18.0

    # To use Google Gemini
    pip install google-genai>=1.0.0

    # To work with images
    pip install Pillow>=10.0.0

Optional Dependencies
--------------------

LLuMinary has several optional dependency groups:

.. code-block:: bash

    # For AWS services (Bedrock)
    pip install "lluminary[aws]"

    # For documentation development
    pip install "lluminary[docs]"

Development Installation
-----------------------

If you want to contribute to LLuMinary, you can install it with development dependencies:

.. code-block:: bash

    git clone https://github.com/PimpMyNines/LLuMinary.git
    cd LLuMinary
    pip install -e ".[dev]"

This will install all the necessary tools for development, testing, and documentation.

Environment Variables
-------------------

LLuMinary uses environment variables for API keys and other configuration:

.. code-block:: bash

    # OpenAI API Key
    export OPENAI_API_KEY="your-api-key"

    # Anthropic API Key
    export ANTHROPIC_API_KEY="your-api-key"

    # Google API Key
    export GOOGLE_API_KEY="your-api-key"

    # Cohere API Key
    export COHERE_API_KEY="your-api-key"

    # AWS Credentials (for Bedrock)
    export AWS_ACCESS_KEY_ID="your-access-key"
    export AWS_SECRET_ACCESS_KEY="your-secret-key"
    export AWS_REGION="us-east-1"

You can also provide these credentials directly when initializing LLuMinary.
