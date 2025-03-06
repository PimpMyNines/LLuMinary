from setuptools import find_packages, setup

setup(
    name="llmhandler",
    version="0.1.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests>=2.31.0",
        "anthropic>=0.18.0",
        "openai>=1.12.0",
        "Pillow>=10.0.0",
        "google-genai>=1.0.0",
        "httpx>=0.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-xdist>=3.5.0",
            "responses>=0.25.0",
        ],
        "aws": [
            "boto3>=1.34.0",
        ],
    },
    author="Chase Brown",
    author_email="chaseb@knowbe4.com",
    description="A package for handling LLM operations across multiple providers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://gitlab.kb4.it/library/packages/python/llm-handler",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    include_package_data=True,
)
