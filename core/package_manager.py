import os
import subprocess
import sys
import uuid
import zipfile
import shutil
from pathlib import Path

from requests import get

class PackageManager:
    def __init__(self, zip_path, comfy_path, temp_path=None, extract_minimal=False, log_file=None):
        """
        Initialize PackageManager with a ZIP file path and ComfyUI directory.

        Args:
            zip_path (Path): Path to the ZIP file containing workflow and related files.
            comfy_path (Path): Path to the ComfyUI directory.
            temp_path (Path, optional): Parent directory for temporary folder.
            extract_minimal (bool): If True, extract only JSON files from ZIP.
            log_file (Path, optional): Path to log file for logging operations.
        """
        self.zip_path = Path(zip_path).resolve()
        self.comfy_path = Path(comfy_path).resolve()
        self.temp_path = Path(temp_path).resolve() if temp_path else self.comfy_path / "temp"
        self.extract_minimal = extract_minimal
        self.log_file = log_file if log_file else None
        self.temp_dir = None
        self.created_temp_dir = False
        self.package_name = self.zip_path.stem

    def log(self, message):
        """Log a message to the console and optionally to a log file."""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

    def extract_zip(self):
        """Extract the ZIP file and return the path to workflow.json."""
        self.temp_dir = self.temp_path / f"temp_{uuid.uuid4().hex}"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.created_temp_dir = True
        self.log(f"Extracting files to temporary directory: {self.temp_dir}")

        with zipfile.ZipFile(self.zip_path, 'r') as zipf:
            json_files = [f for f in zipf.namelist() if f.lower().endswith('.json')]
            if not json_files:
                error_msg = f"No JSON files found in {self.zip_path}"
                self.log(error_msg)
                raise FileNotFoundError(error_msg)
            root_json_files = [f for f in json_files if not (os.sep in f or '/' in f)]
            target_file = next((f for f in root_json_files if f.lower() == 'workflow.json'), None)
            if not target_file:
                error_msg = f"No workflow.json found at the root of {self.zip_path}"
                self.log(error_msg)
                raise FileNotFoundError(error_msg)

            for file_name in (json_files if self.extract_minimal else zipf.namelist()):
                zipf.extract(file_name, self.temp_dir)
                self.log(f"Extracted file: {Path(file_name).name}")

            zip_basename = self.zip_path.stem
            workflow_dir = self.comfy_path / "user" / "default" / "workflows" / zip_basename
            os.makedirs(workflow_dir, exist_ok=True)
            for json_file in root_json_files:
                src_path = self.temp_dir / json_file
                if src_path.exists():
                    dest_path = workflow_dir / Path(json_file).name
                    shutil.copy2(src_path, dest_path)
                    self.log(f"Saving workflow to: {dest_path}")

            self._run_script_if_exists(self.temp_dir / "pre.py")
            self._install_custom_nodes()
            self._run_script_if_exists(self.temp_dir / "post.py")

            return self.temp_dir / target_file

    def _run_script_if_exists(self, script_path):
        """Run a Python script if it exists."""
        if script_path.exists():
            self.log(f"Running {script_path}...")
            try:
                subprocess.check_call([sys.executable, str(script_path)], cwd=self.temp_dir)
                self.log(f"Successfully ran {script_path}")
            except subprocess.CalledProcessError as e:
                error_msg = f"Error running {script_path}: {e}"
                self.log(error_msg)
                raise

    def _install_custom_nodes(self):
        """Install requirements for custom nodes in the extracted ZIP."""
        custom_nodes_dir = self.temp_dir / "ComfyUI" / "custom_nodes"
        if custom_nodes_dir.exists() and custom_nodes_dir.is_dir():
            for node_dir in custom_nodes_dir.iterdir():
                if node_dir.is_dir():
                    requirements_path = node_dir / "requirements.txt"
                    if requirements_path.exists():
                        node_name = node_dir.name
                        self.log(f"Installing requirements for custom node {node_name}...")
                        try:
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], cwd=node_dir)
                            self.log(f"Successfully installed requirements for custom node {node_name}")
                        except subprocess.CalledProcessError as e:
                            error_msg = f"Failed to install requirements for custom node {node_name}: {e}"
                            self.log(error_msg)
                            raise
        if not self.extract_minimal:
            extracted_comfy_dir = self.temp_dir / "ComfyUI"
            if extracted_comfy_dir.exists() and extracted_comfy_dir.is_dir():
                for subfolder in extracted_comfy_dir.iterdir():
                    if subfolder.is_dir():
                        dest_folder = self.comfy_path / subfolder.name
                        self.log(f"Begin folder copy: {subfolder.name}")
                        shutil.copytree(subfolder, dest_folder, dirs_exist_ok=True)
                        self.log(f"Copied folder {subfolder.name} to {dest_folder}")

    def cleanup(self):
        """Clean up temporary directory if it was created."""
        if self.temp_dir and self.temp_dir.exists() and self.created_temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.log(f"Removed temporary directory: {self.temp_dir}")