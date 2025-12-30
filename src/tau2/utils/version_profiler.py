# tau2/utils/version_profiler.py
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict
import sys


def get_git_info(repo_path: Path) -> Dict:
    """Get git repository information"""
    info = {"path": str(repo_path)}
    try:
        # Get commit
        commit = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10
        )
        if commit.returncode == 0:
            info["git_commit"] = commit.stdout.strip()
        
        # Get branch
        branch = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=10
        )
        if branch.returncode == 0:
            info["git_branch"] = branch.stdout.strip()
        
        # Check if dirty
        status = subprocess.run(
            ["git", "-C", str(repo_path), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10
        )
        if status.returncode == 0:
            info["git_dirty"] = len(status.stdout.strip()) > 0
        
        # Get remote
        remote = subprocess.run(
            ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=10
        )
        if remote.returncode == 0:
            info["git_remote"] = remote.stdout.strip()
    except Exception:
        pass
    
    return info


def detect_tau2_root(env_path: Optional[str] = None) -> Optional[Path]:
    """
    Detect the tau2-bench root directory from the installed tau2 package in the environment.
    
    Uses the environment's Python to find where the tau2 package is installed,
    then walks up to find the git repository root.
    
    Args:
        env_path: Path to the Python environment (conda or venv)
    
    Returns:
        Path to the tau2-bench git repository, or None if not found
    """
    if not env_path:
        env_path = os.getenv("CONDA_PREFIX") or sys.prefix
    
    python_path = Path(env_path) / "bin" / "python"
    if not python_path.exists():
        return None
    
    package_file = None
    repo_root = None
    
    # Method 1: Use pip show to get package location
    try:
        pip_path = Path(env_path) / "bin" / "pip"
        result = subprocess.run(
            [str(pip_path), "show", "tau2"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Editable project location:"):
                    editable_location = line.split(":", 1)[1].strip()
                    # For editable installs, this is the repo root
                    repo_root = Path(editable_location)
                    if (repo_root / ".git").exists():
                        return repo_root
                    # If no .git at editable location, walk up
                    current = repo_root
                    for _ in range(5):
                        if (current / ".git").exists():
                            return current
                        if current.parent == current:
                            break
                        current = current.parent
                elif line.startswith("Location:"):
                    regular_location = line.split(":", 1)[1].strip()
                    # For regular installs, find the tau2 package and walk up
                    package_dir = Path(regular_location) / "tau2"
                    if package_dir.exists():
                        package_file = package_dir / "__init__.py"
    except Exception:
        pass
    
    # Method 2: Use importlib.metadata to find package location
    if not package_file and not repo_root:
        try:
            metadata_script = '''
import importlib.metadata
import pathlib
try:
    files = importlib.metadata.files("tau2")
    if files:
        for f in files:
            if f.name == "__init__.py" and "tau2" in str(f):
                located = f.locate()
                if located:
                    print(located)
                    break
except Exception:
    pass
'''
            result = subprocess.run(
                [str(python_path), "-c", metadata_script],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                package_file = Path(result.stdout.strip())
        except Exception:
            pass
    
    # Method 3: Direct import
    if not package_file and not repo_root:
        try:
            result = subprocess.run(
                [str(python_path), "-c", "import tau2; print(tau2.__file__)"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                package_file = Path(result.stdout.strip())
        except Exception:
            pass
    
    # If we found a package file, walk up to find git repo
    if package_file:
        current = package_file.parent
        for _ in range(10):  # Limit search depth
            git_dir = current / ".git"
            if git_dir.exists():
                return current
            if current.parent == current:
                break
            current = current.parent
    
    # If we found a repo root but no .git, return it anyway
    if repo_root:
        return repo_root
    
    # Fallback: return None (caller can use file-based detection)
    return None


def get_env_info(env_path: Optional[str] = None) -> Dict:
    """Get environment information"""
    if not env_path:
        env_path = os.getenv("CONDA_PREFIX") or sys.prefix
    
    info = {
        "path": env_path,
        "env_type": "conda" if os.getenv("CONDA_PREFIX") else "venv",
    }
    
    # Get Python version
    try:
        python_bin = Path(env_path) / "bin" / "python"
        result = subprocess.run(
            [str(python_bin), "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            info["python_version"] = result.stdout.strip().replace("Python ", "")
    except Exception:
        pass
    
    # Get packages (conda or pip)
    packages = {}
    try:
        if info["env_type"] == "conda":
            result = subprocess.run(
                ["conda", "list", "--prefix", env_path, "--json"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                pkg_list = json.loads(result.stdout)
                packages = {pkg["name"]: pkg["version"] for pkg in pkg_list if "name" in pkg and "version" in pkg}
        else:
            pip_path = Path(env_path) / "bin" / "pip"
            result = subprocess.run(
                [str(pip_path), "freeze"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if "==" in line:
                        name, version = line.split("==", 1)
                        packages[name.strip()] = version.strip()
        info["packages"] = packages
    except Exception:
        pass
    
    return info


def get_version_info(tau2_root: Optional[Path] = None, env_path: Optional[str] = None) -> Dict:
    """
    Get complete version information for tau2-bench.
    
    Args:
        tau2_root: Path to tau2-bench root directory. If None, will auto-detect from environment.
        env_path: Path to Python environment. If None, will use CONDA_PREFIX or sys.prefix.
    
    Returns:
        Dictionary with version information including harness (git) and environment info.
    """
    # Auto-detect tau2-bench root from environment if not provided
    if tau2_root is None:
        tau2_root = detect_tau2_root(env_path)
        if tau2_root is None:
            # Fallback: try to find from current file location
            # This file is in tau2/utils/, so go up 2 levels to get to tau2-bench root
            tau2_root = Path(__file__).parents[2]
    
    version_info = {
        "profiled_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # Get harness (tau2-bench) git info
    version_info["harness"] = get_git_info(tau2_root)
    
    # Get environment info
    version_info["environment"] = get_env_info(env_path)
    
    return version_info