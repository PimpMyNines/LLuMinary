from setuptools import setup, find_packages

setup(
    name="github-project-management",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    install_requires=[
        "pygithub>=1.58.0",
        "click>=8.1.3",
        "pyyaml>=6.0",
        "requests>=2.28.1",
        "python-dotenv>=0.21.0",
    ],
    entry_points={
        "console_scripts": [
            "github-pm=github_project_management.cli:main",
        ],
    },
    author="LLuMinary Team",
    author_email="team@lluminary.dev",
    description="A Python-based GitHub project management tool",
    keywords="github, project, management, issues",
    python_requires=">=3.8",
)
