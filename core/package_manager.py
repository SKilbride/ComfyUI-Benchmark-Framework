# core/package_manager.py
import os
import subprocess
import sys
import uuid
import shutil
from pathlib import Path
from typing import Optional

from .smart_extractor import SmartExtractor  # <-- NEW IMPORT


class PackageManager:
    def __init__(self,
                 zip_path: Path,
                 comfy_path: Path,
                 temp_path: Optional[Path] = None,
                 extract_minimal: bool = False,
                 force_extract: bool = False,
                 log_file: Optional[Path] = None):
        """
        Initialize PackageManager with smart extraction capabilities.
        """
        self.zip_path = Path(zip_path).resolve()
        self.comfy_path = Path(comfy_path).resolve()
        self.temp_path = Path(temp_path).resolve() if temp_path else self.comfy_path / "temp"
        self.extract_minimal = extract_minimal
        self.force_extract = force_extract
        self.log_file = log_file
        self.temp_dir = None
        self.created_temp_dir = False
        self.package_name = self.zip_path.stem
        self.extractor: Optional[SmartExtractor] = None
        self.custom_nodes_extracted = False

    def log(self, message: str):
        print(message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

    def extract_zip(self) -> Path:
        """
        Extract ZIP using SmartExtractor.
        Returns path to extracted workflow.json
        """
        self.temp_dir = self.temp_path / f"temp_{uuid.uuid4().hex}"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.created_temp_dir = True
        self.log(f"Extracting package to temp dir: {self.temp_dir}")

        # === SMART EXTRACTOR ===
        self.extractor = SmartExtractor(
            zip_path=self.zip_path,
            comfy_path=self.comfy_path,
            temp_dir=self.temp_dir,
            log_file=self.log_file,
            minimal=self.extract_minimal,
            force_extraction=self.force_extract
        )

        try:
            workflow_path = self.extractor.extract()
            self.log(f"Smart extraction complete: {workflow_path}")
        except Exception as e:
            self.log(f"Smart extraction failed: {e}")
            raise

        # === SAVE TO USER WORKFLOWS (for GUI/history) ===
        workflow_dir = self.comfy_path / "user" / "default" / "workflows" / self.package_name
        os.makedirs(workflow_dir, exist_ok=True)
        for json_file in ["workflow.json", "warmup.json", "baseconfig.json"]:
            src = self.temp_dir / json_file
            if src.exists():
                dest = workflow_dir / json_file
                shutil.copy2(src, dest)
                self.log(f"Saved {json_file} to {dest}")

        return workflow_path

    def cleanup(self):
        """Clean up temp directory."""
        if self.extractor:
            self.extractor.cleanup()
        elif self.temp_dir and self.temp_dir.exists() and self.created_temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.log(f"Removed temporary directory: {self.temp_dir}")