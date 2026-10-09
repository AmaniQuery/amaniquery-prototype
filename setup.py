"""
Setup script - Install dependencies and initialize project
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a shell command"""
    print(f"\n{'=' * 60}")
    print(f"[PKG] {description}")
    print(f"{'=' * 60}")

    try:
        subprocess.run(command, check=True, shell=True)
        print(f"[OK] {description} - Complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} - Failed: {e}")
        return False


def main():
    """Main setup function"""
    print("=" * 60)
    print("[BUILD] AmaniQuery Setup")
    print("=" * 60)

    project_root = Path(__file__).parent

    # Check Python version
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        return

    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}")

    # Create virtual environment
    venv_path = project_root / "venv"
    if not venv_path.exists():
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return
    else:
        print("\n[OK] Virtual environment already exists")

    # Determine pip command
    if sys.platform == "win32":
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"

    # Upgrade pip
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")

    # Install dependencies
    if not run_command(
        f"{pip_cmd} install -r requirements.txt", "Installing dependencies"
    ):
        print("\n[WARN] Some packages may have failed to install")
        print("   You can install them manually later")

    # Create data directories
    print("\n" + "=" * 60)
    print("[DIR] Creating data directories")
    print("=" * 60)

    directories = [
        project_root / "data" / "raw",
        project_root / "data" / "processed",
        project_root / "data" / "embeddings",
        project_root / "data" / "chroma_db",
        project_root / "logs",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   [OK] {directory}")

    # Create .env from template
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"

    if not env_file.exists() and env_example.exists():
        print("\n" + "=" * 60)
        print("[CONFIG] Creating .env file")
        print("=" * 60)

        with open(env_example, "r") as f:
            content = f.read()

        with open(env_file, "w") as f:
            f.write(content)

        print(f"   [OK] Created .env")
        print(f"   [WARN] Please edit .env and add your API keys")

    # Print next steps
    print("\n" + "=" * 60)
    print("[OK] Setup Complete!")
    print("=" * 60)
    print("\n[INFO] Next Steps:\n")
    print("1. Activate virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")

    print("\n2. Edit .env file with your API keys:")
    print("   - OPENAI_API_KEY (for RAG)")
    print("   - Other configuration as needed")

    print("\n3. Run the modules in order:")
    print("   a. Crawl data:")
    print("      python -m Module1_NiruSpider.crawl_all")
    print("\n   b. Process data:")
    print("      python -m Module2_NiruParser.process_all")
    print("\n   c. Populate database:")
    print("      python -m Module3_NiruDB.populate_db")
    print("\n   d. Start API server:")
    print("      python -m Module4_NiruAPI.api")

    print("\n4. Access the API:")
    print("   - API: http://localhost:8000")
    print("   - Docs: http://localhost:8000/docs")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
