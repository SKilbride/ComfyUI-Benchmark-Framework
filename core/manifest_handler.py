# core/manifest_handler.py
import os
import sys
import json
import yaml
import hashlib
import shutil
import subprocess
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download, snapshot_download, HfApi
from tqdm import tqdm
import time


class ManifestHandler:
    """Handles benchmark package manifests for downloading and installing resources."""
    
    SUPPORTED_SOURCES = ['bundled', 'huggingface', 'git', 'url', 'local']
    SUPPORTED_TYPES = ['workflow', 'config', 'model', 'custom_node', 'input', 'script']
    
    def __init__(self, manifest_path: Path, comfy_path: Path, log_file: Optional[Path] = None, 
                 max_workers: int = 4, resume_downloads: bool = True):
        """
        Initialize ManifestHandler.
        
        Args:
            manifest_path: Path to manifest file (YAML or JSON)
            comfy_path: Path to ComfyUI installation
            log_file: Optional path to log file
            max_workers: Number of parallel download workers (default: 4)
            resume_downloads: Enable resume capability for interrupted downloads (default: True)
        """
        self.manifest_path = Path(manifest_path)
        self.comfy_path = Path(comfy_path)
        self.log_file = log_file
        self.manifest = None
        self.hf_token = os.getenv('HF_TOKEN')
        self.max_workers = max_workers
        self.resume_downloads = resume_downloads
        self.partial_download_dir = self.comfy_path / ".partial_downloads"
        
        # Track what was actually downloaded vs skipped
        self.downloaded_items = []
        self.skipped_items = []
        
        # Cache for file status to verify checksums only once
        self._file_status_cache = None
        
        # Create partial download directory if resume is enabled
        if self.resume_downloads:
            self.partial_download_dir.mkdir(parents=True, exist_ok=True)
        
    def log(self, message: str, level: str = "INFO"):
        """Log message to console and file with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"
        print(formatted_message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(formatted_message + '\n')
    
    def load_manifest(self) -> Dict:
        """Load and validate manifest file."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        # Support both JSON and YAML
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            if self.manifest_path.suffix in ['.yaml', '.yml']:
                self.manifest = yaml.safe_load(f)
            else:
                self.manifest = json.load(f)
        
        package_name = self.manifest.get('package', {}).get('name', 'unknown')
        self.log(f"Loaded manifest: {package_name}")
        return self.manifest
    
    def validate_manifest(self) -> bool:
        """Basic manifest validation."""
        if not self.manifest:
            raise ValueError("Manifest not loaded")
        
        required_keys = ['package', 'items']
        for key in required_keys:
            if key not in self.manifest:
                raise ValueError(f"Manifest missing required key: {key}")
        
        # Validate items
        for item in self.manifest['items']:
            if 'name' not in item or 'type' not in item or 'source' not in item:
                raise ValueError(f"Item missing required fields: {item}")
            
            if item['type'] not in self.SUPPORTED_TYPES:
                raise ValueError(f"Unsupported item type: {item['type']}")
            
            if item['source'] not in self.SUPPORTED_SOURCES:
                raise ValueError(f"Unsupported source: {item['source']}")
        
        self.log("Manifest validation passed")
        return True
    
    def check_existing_files(self, force_recheck: bool = False) -> Dict[str, Dict]:
        """
        Check which files already exist and their status.
        Uses internal cache to prevent double-verification unless force_recheck is True.
        """
        if self._file_status_cache is not None and not force_recheck:
            return self._file_status_cache

        existing = {}
        for item in self.manifest['items']:
            if item['source'] == 'bundled':
                continue
            
            path = self.comfy_path / item['path']
            
            status = {
                'exists': False,
                'valid': False,
                'needs_download': True,
                'reason': None,
                'partial_exists': False,
                'partial_size': 0
            }
            
            # Check for partial downloads
            if self.resume_downloads:
                partial_path = self._get_partial_path(item['path'])
                if partial_path.exists():
                    status['partial_exists'] = True
                    status['partial_size'] = partial_path.stat().st_size
            
            if path.exists():
                status['exists'] = True
                
                # Verify checksum if provided (support both 'sha256' and 'sha')
                checksum = item.get('sha256') or item.get('sha')
                if checksum:
                    if self._verify_checksum(path, checksum):
                        status['valid'] = True
                        status['needs_download'] = False
                        status['reason'] = 'exists_and_valid'
                        self.log(f"✓ {item['name']} exists and checksum matches")
                    else:
                        status['valid'] = False
                        status['needs_download'] = True
                        status['reason'] = 'checksum_mismatch'
                        self.log(f"⚠ {item['name']} exists but checksum mismatch - will re-download", "WARNING")
                else:
                    # No checksum provided - trust existing file
                    status['valid'] = True
                    status['needs_download'] = False
                    status['reason'] = 'exists_no_checksum'
                    self.log(f"✓ {item['name']} exists (no checksum to verify)")
            else:
                status['exists'] = False
                status['valid'] = False
                status['needs_download'] = True
                status['reason'] = 'missing'
                resume_msg = f" (partial: {status['partial_size'] / 1024 / 1024:.1f}MB)" if status['partial_exists'] else ""
                self.log(f"⊘ {item['name']} not found - will download{resume_msg}")
            
            existing[item['name']] = status
        
        self._file_status_cache = existing
        return existing
    
    def download_items(self, skip_existing: bool = True, required_only: bool = False, 
                      verify_checksums: bool = True, dry_run: bool = False, parallel: bool = True):
        """
        Download all items from manifest.
        
        Args:
            skip_existing: Skip files that already exist with valid checksums
            required_only: Only download items marked as required
            verify_checksums: Verify SHA256 checksums after download
            dry_run: Preview actions without executing downloads
            parallel: Use parallel downloads for improved speed
        """
        items_to_download = self.manifest['items']
        
        if required_only:
            items_to_download = [i for i in items_to_download if i.get('required', False)]
        
        # This will use the cache populated during print_summary()
        existing = self.check_existing_files() if skip_existing else {}
        
        # Filter items that need downloading
        download_list = []
        for item in items_to_download:
            # Skip bundled items
            if item['source'] == 'bundled':
                continue
            
            # Check if we should skip this file
            if skip_existing and item['name'] in existing:
                file_status = existing[item['name']]
                if not file_status['needs_download']:
                    self.log(f"⊘ Skipping {item['name']} ({file_status['reason']})")
                    continue
                elif file_status['reason'] == 'checksum_mismatch':
                    self.log(f"⚠ Re-downloading {item['name']} due to checksum mismatch", "WARNING")
            
            # Check for gated models
            if item.get('gated', False):
                self.log(f"⚠ {item['name']} is gated - license acceptance required", "WARNING")
                if not self.hf_token:
                    self.log(f"✗ HF_TOKEN not set - cannot download gated model", "ERROR")
                    self.log(f"  Visit: {item.get('license_url', 'Hugging Face')}")
                    if item.get('required', False):
                        raise Exception(f"Required gated model {item['name']} cannot be downloaded without HF_TOKEN")
                    continue
            
            download_list.append((item, existing.get(item['name'], {})))
        
        if dry_run:
            self.log("\n" + "=" * 60)
            self.log("DRY RUN MODE - No downloads will be performed")
            self.log("=" * 60)
            for item, status in download_list:
                source_info = self._get_source_info(item)
                self.log(f"Would download: {item['name']}")
                self.log(f"  Source: {source_info}")
                self.log(f"  Destination: {item['path']}")
                if status.get('partial_exists'):
                    self.log(f"  Resume from: {status['partial_size'] / 1024 / 1024:.1f}MB")
            self.log("=" * 60 + "\n")
            return
        
        # Download items
        if parallel and len(download_list) > 1:
            self._download_parallel(download_list, verify_checksums)
        else:
            self._download_sequential(download_list, verify_checksums)
    
    def _download_sequential(self, download_list: List[Tuple[Dict, Dict]], verify_checksums: bool):
        """Download items sequentially."""
        for item, status in download_list:
            try:
                self._download_item(item, verify_checksums)
            except Exception as e:
                if item.get('required', False):
                    raise Exception(f"Failed to download required item {item['name']}: {e}")
                else:
                    self.log(f"⚠ Failed to download optional item {item['name']}: {e}", "WARNING")
    
    def _download_parallel(self, download_list: List[Tuple[Dict, Dict]], verify_checksums: bool):
        """Download items in parallel using ThreadPoolExecutor."""
        self.log(f"Starting parallel downloads with {self.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_item = {
                executor.submit(self._download_item, item, verify_checksums): item
                for item, status in download_list
            }
            
            # Process completed downloads
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    future.result()
                except Exception as e:
                    if item.get('required', False):
                        # Cancel remaining downloads and raise
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise Exception(f"Failed to download required item {item['name']}: {e}")
                    else:
                        self.log(f"⚠ Failed to download optional item {item['name']}: {e}", "WARNING")
    
    def _download_item(self, item: Dict, verify_checksum: bool = True):
        """
        Download a single item based on its source type.
        
        Note: Checksum verification is only performed for items with type='model'.
        Custom nodes, configs, inputs, etc. don't use SHA256 verification.
        """
        # Only verify checksums for model types
        should_verify = verify_checksum and item.get('type') == 'model'
        
        if item['source'] == 'huggingface':
            self._download_from_huggingface(item, verify_checksum=should_verify)
        elif item['source'] == 'git':
            self._download_from_git(item)
        elif item['source'] == 'url':
            self._download_from_url(item, verify_checksum=should_verify)
        elif item['source'] == 'local':
            self._copy_from_local(item)
        else:
            self.log(f"⚠ Unknown source type: {item['source']}", "WARNING")
    
    def _get_source_info(self, item: Dict) -> str:
        """Get human-readable source information."""
        if item['source'] == 'huggingface':
            return f"HuggingFace: {item['repo']}/{item['file']}"
        elif item['source'] == 'git':
            return f"Git: {item['url']} (ref: {item.get('ref', 'main')})"
        elif item['source'] == 'url':
            return f"URL: {item['url']}"
        elif item['source'] == 'local':
            return f"Local: {item.get('source_path', 'unknown')}"
        return item['source']
    
    def _get_partial_path(self, target_path: str) -> Path:
        """Get path for partial download file."""
        # Create a safe filename from the target path
        safe_name = str(target_path).replace('/', '_').replace('\\', '_')
        return self.partial_download_dir / f"{safe_name}.partial"
    
    def _download_from_huggingface(self, item: Dict, verify_checksum: bool = True):
        """Download from Hugging Face Hub with resume capability."""
        self.log(f"↓ Downloading {item['name']} from Hugging Face...")
        
        repo_id = item['repo']
        filename = item['file']
        local_path = self.comfy_path / item['path']
        partial_path = self._get_partial_path(item['path'])
        
        # Parse remote_path to get subfolder if provided
        subfolder = None
        if 'remote_path' in item:
            remote_path = item['remote_path']
            if '/tree/' in remote_path:
                parts = remote_path.split('/tree/')
                if len(parts) > 1:
                    branch_and_path = parts[1].split('/', 1)
                    if len(branch_and_path) > 1:
                        subfolder = branch_and_path[1]
                        self.log(f"  Using subfolder: {subfolder}")
        
        # Build the full filename path in the repo
        repo_filename = f"{subfolder}/{filename}" if subfolder else filename
        
        # Create parent directory
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get file info for progress bar
        try:
            api = HfApi(token=self.hf_token)
            file_info = api.model_info(repo_id, files_metadata=True)
            total_size = None
            for sibling in file_info.siblings:
                if sibling.rfilename == repo_filename:
                    total_size = sibling.size
                    break
        except Exception as e:
            self.log(f"  Could not get file size: {e}", "WARNING")
            total_size = None
        
        # Check for remote SHA256
        remote_checksum = None
        try:
            api = HfApi(token=self.hf_token)
            file_metadata = api.get_paths_info(repo_id, paths=[repo_filename], repo_type="model")
            if file_metadata and len(file_metadata) > 0:
                remote_checksum = file_metadata[0].lfs.get('sha256') if hasattr(file_metadata[0], 'lfs') else None
        except Exception:
            pass
        
        manifest_checksum = item.get('sha256') or item.get('sha')
        checksum_to_verify = manifest_checksum or remote_checksum
        
        try:
            # Download with progress bar
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=repo_filename,
                local_dir=local_path.parent,
                local_dir_use_symlinks=False,
                token=self.hf_token,
                resume_download=self.resume_downloads
            )
            
            # Move to correct location if needed
            if Path(downloaded_path) != local_path:
                shutil.move(downloaded_path, local_path)
            
        except Exception as e:
            self.log(f"✗ Download failed: {e}", "ERROR")
            raise
        
        # Verify checksum
        if verify_checksum and checksum_to_verify:
            self.log(f"  Verifying checksum...")
            if self._verify_checksum(local_path, checksum_to_verify):
                self.log(f"✓ {item['name']} downloaded and verified")
                
                # Warn if manifest checksum doesn't match remote
                if manifest_checksum and remote_checksum and manifest_checksum != remote_checksum:
                    self.log(f"  ⚠ WARNING: Manifest SHA256 differs from remote!", "WARNING")
                    self.log(f"    Manifest: {manifest_checksum[:16]}...")
                    self.log(f"    Remote:   {remote_checksum[:16]}...")
                
                # Clean up partial file
                if partial_path.exists():
                    partial_path.unlink()
            else:
                raise Exception(f"Checksum mismatch for {item['name']}")
        else:
            self.log(f"✓ {item['name']} downloaded (no checksum verification)")
    
    def _download_from_git(self, item: Dict) -> bool:
        """
        Clone from Git repository or update existing clone.
        Returns True if downloaded/updated, False if skipped.
        
        Strategy:
        1. If doesn't exist → clone
        2. If exists but not a git repo → delete and clone
        3. If exists and is git repo → try to update with git pull
        4. If update fails → delete and clone as fallback
        """
        url = item['url']
        ref = item.get('ref', 'main')
        local_path = self.comfy_path / item['path']
        
        # Case 1: Directory doesn't exist - fresh clone
        if not local_path.exists():
            self.log(f"↓ Cloning {item['name']} from Git...")
            return self._git_clone_fresh(item, url, ref, local_path)
        
        # Case 2: Directory exists but is not a Git repo - delete and clone
        if not (local_path / '.git').exists():
            self.log(f"↓ Cloning {item['name']} from Git...")
            self.log(f"  Removing non-git directory...")
            self._remove_directory_safely(local_path)
            return self._git_clone_fresh(item, url, ref, local_path)
        
        # Case 3: It's a Git repo - check if already at correct version
        self.log(f"↓ Checking {item['name']}...")
        
        try:
            # Get local commit
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=local_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            local_commit = result.stdout.strip()
            
            # Get remote commit for ref
            result = subprocess.run(
                ['git', 'ls-remote', url, ref],
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            if result.stdout:
                remote_commit = result.stdout.split()[0]
                
                # Already at correct version - skip
                if local_commit == remote_commit:
                    self.log(f"⊘ Skipping {item['name']} (already at {ref})")
                    self.skipped_items.append(item)
                    return False
                
                # Different version - update needed
                self.log(f"  Updating {item['name']} from {local_commit[:8]} to {remote_commit[:8]}...")
                return self._git_update_existing(item, url, ref, local_path)
            else:
                # Ref doesn't exist remotely - try to clone fresh
                self.log(f"  Ref {ref} not found on remote, re-cloning...", "WARNING")
                self._remove_directory_safely(local_path)
                return self._git_clone_fresh(item, url, ref, local_path)
                
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError) as e:
            # Git commands failed - try to update anyway
            self.log(f"  Could not verify version ({e}), attempting update...", "WARNING")
            return self._git_update_existing(item, url, ref, local_path)
    
    def _git_update_existing(self, item: Dict, url: str, ref: str, local_path: Path) -> bool:
        """
        Update existing Git repository using fetch + checkout + reset.
        Falls back to fresh clone if update fails.
        """
        try:
            # Fetch latest from remote
            self.log(f"  Fetching updates...")
            subprocess.run(
                ['git', 'fetch', 'origin'],
                cwd=local_path,
                check=True,
                capture_output=True,
                timeout=60
            )
            
            # Checkout the ref
            self.log(f"  Checking out {ref}...")
            subprocess.run(
                ['git', 'checkout', ref],
                cwd=local_path,
                check=True,
                capture_output=True,
                timeout=30
            )
            
            # Reset to match remote exactly
            try:
                subprocess.run(
                    ['git', 'reset', '--hard', f'origin/{ref}'],
                    cwd=local_path,
                    check=True,
                    capture_output=True,
                    timeout=30
                )
            except subprocess.CalledProcessError:
                # origin/{ref} might not work for tags/commits, try without origin/
                subprocess.run(
                    ['git', 'reset', '--hard', ref],
                    cwd=local_path,
                    check=True,
                    capture_output=True,
                    timeout=30
                )
            
            self.log(f"✓ {item['name']} updated")
            self.downloaded_items.append(item)
            
            # Install requirements if specified
            self._install_requirements_if_needed(item, local_path)
            
            return True
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # Update failed - fall back to fresh clone
            self.log(f"  Git update failed, re-cloning...", "WARNING")
            self._remove_directory_safely(local_path)
            return self._git_clone_fresh(item, url, ref, local_path)
    
    def _git_clone_fresh(self, item: Dict, url: str, ref: str, local_path: Path) -> bool:
        """
        Perform a fresh git clone.
        """
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.log(f"  Cloning from {url} (ref: {ref})...")
        try:
            subprocess.run([
                'git', 'clone', '--depth', '1',
                '--branch', ref,
                '--progress',
                url, str(local_path)
            ], check=True, capture_output=False)
        except subprocess.CalledProcessError as e:
            self.log(f"✗ Git clone failed: {e}", "ERROR")
            raise
        
        self.log(f"✓ {item['name']} cloned")
        self.downloaded_items.append(item)
        
        # Install requirements if specified
        self._install_requirements_if_needed(item, local_path)
        
        return True
    
    def _remove_directory_safely(self, path: Path):
        """
        Safely remove a directory, handling Windows file locks and permissions.
        """
        try:
            shutil.rmtree(path)
        except (OSError, PermissionError) as e:
            # On Windows, files might be locked - try harder
            self.log(f"  Retrying removal with error handler...", "WARNING")
            import stat
            
            def handle_remove_readonly(func, path_to_remove, exc):
                """Error handler for Windows readonly files."""
                try:
                    os.chmod(path_to_remove, stat.S_IWRITE)
                    func(path_to_remove)
                except:
                    pass
            
            try:
                shutil.rmtree(path, onerror=handle_remove_readonly)
            except Exception as e2:
                # Last resort: rename to .backup
                self.log(f"  Could not remove, renaming to .backup...", "WARNING")
                import uuid
                backup_path = path.parent / f".backup_{path.name}_{uuid.uuid4().hex[:8]}"
                try:
                    if backup_path.exists():
                        shutil.rmtree(backup_path, ignore_errors=True)
                    os.rename(path, backup_path)
                    self.log(f"  Renamed to {backup_path.name}", "WARNING")
                except Exception as e3:
                    raise Exception(f"Cannot remove existing directory {path}: {e3}")
    
    def _install_requirements_if_needed(self, item: Dict, local_path: Path):
        """
        Install requirements.txt if specified in manifest item.
        """
        if item.get('install_requirements', False):
            req_file = local_path / 'requirements.txt'
            if req_file.exists():
                self.log(f"  Installing requirements for {item['name']}...")
                try:
                    subprocess.run([
                        sys.executable, '-m', 'pip', 'install', '-r', str(req_file)
                    ], check=True, capture_output=False)
                    self.log(f"  ✓ Requirements installed")
                except subprocess.CalledProcessError as e:
                    self.log(f"  ⚠ Requirements installation failed: {e}", "WARNING")
                    if item.get('required', False):
                        raise
    
    def _download_from_url(self, item: Dict, verify_checksum: bool = True):
        """Download from direct URL with progress bar and resume capability."""
        self.log(f"↓ Downloading {item['name']} from URL...")
        
        url = item['url']
        local_path = self.comfy_path / item['path']
        partial_path = self._get_partial_path(item['path'])
        
        # Create parent directory
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check for partial download
        resume_header = {}
        initial_pos = 0
        if self.resume_downloads and partial_path.exists():
            initial_pos = partial_path.stat().st_size
            resume_header = {'Range': f'bytes={initial_pos}-'}
            self.log(f"  Resuming from {initial_pos / 1024 / 1024:.1f}MB")
        
        # Get file size
        try:
            head_response = requests.head(url, allow_redirects=True)
            total_size = int(head_response.headers.get('content-length', 0))
            supports_resume = head_response.headers.get('accept-ranges') == 'bytes'
        except Exception:
            total_size = 0
            supports_resume = False
        
        # Download with progress bar
        try:
            response = requests.get(url, stream=True, headers=resume_header if supports_resume else {})
            response.raise_for_status()
            
            # Open file in appropriate mode
            mode = 'ab' if self.resume_downloads and supports_resume and initial_pos > 0 else 'wb'
            temp_path = partial_path if self.resume_downloads else local_path
            
            chunk_size = 8192
            with open(temp_path, mode) as f:
                if total_size > 0:
                    with tqdm(total=total_size, initial=initial_pos, unit='B', unit_scale=True, 
                             desc=item['name'][:30], ncols=80) as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # No size info, download without progress bar
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            
            # Move partial to final location if using resume
            if self.resume_downloads and temp_path != local_path:
                shutil.move(temp_path, local_path)
        
        except Exception as e:
            self.log(f"✗ Download failed: {e}", "ERROR")
            raise
        
        # Make executable if specified
        if item.get('executable', False):
            local_path.chmod(0o755)
        
        # Verify checksum if provided
        if verify_checksum:
            manifest_checksum = item.get('sha256') or item.get('sha')
            if manifest_checksum:
                self.log(f"  Verifying checksum...")
                if self._verify_checksum(local_path, manifest_checksum):
                    self.log(f"✓ {item['name']} downloaded and verified")
                    # Clean up partial file
                    if partial_path.exists():
                        partial_path.unlink()
                else:
                    raise Exception(f"Checksum mismatch for {item['name']}")
            else:
                self.log(f"✓ {item['name']} downloaded (no checksum verification)")
        else:
            self.log(f"✓ {item['name']} downloaded")
    
    def _copy_from_local(self, item: Dict):
        """Copy from local path."""
        self.log(f"↓ Copying {item['name']} from local...")
        
        source = Path(item['source_path'])
        dest = self.comfy_path / item['path']
        
        if not source.exists():
            raise FileNotFoundError(f"Source path not found: {source}")
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        if source.is_dir():
            # Copy directory with progress
            self.log(f"  Copying directory...")
            shutil.copytree(source, dest, dirs_exist_ok=True)
        else:
            # Copy file with progress
            file_size = source.stat().st_size
            if file_size > 10 * 1024 * 1024:  # Show progress for files > 10MB
                with tqdm(total=file_size, unit='B', unit_scale=True, 
                         desc=item['name'][:30], ncols=80) as pbar:
                    with open(source, 'rb') as fsrc:
                        with open(dest, 'wb') as fdst:
                            while True:
                                buf = fsrc.read(8192)
                                if not buf:
                                    break
                                fdst.write(buf)
                                pbar.update(len(buf))
            else:
                shutil.copy2(source, dest)
        
        self.log(f"✓ {item['name']} copied")
    
    def _verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify file SHA256 checksum with progress bar for large files."""
        if not file_path.exists():
            return False
        
        file_size = file_path.stat().st_size
        sha256_hash = hashlib.sha256()
        
        # Show progress for large files (> 100MB)
        if file_size > 100 * 1024 * 1024:
            with open(file_path, "rb") as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, 
                         desc="Verifying", ncols=80) as pbar:
                    while True:
                        data = f.read(4096)
                        if not data:
                            break
                        sha256_hash.update(data)
                        pbar.update(len(data))
        else:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest() == expected_sha256
    
    def custom_nodes_were_downloaded(self) -> bool:
        """
        Check if any custom nodes were actually downloaded (not skipped).
        Returns True only if custom nodes were installed or updated.
        """
        for item in self.downloaded_items:
            # Check if item is marked as custom_node type
            if item.get('type') == 'custom_node':
                return True
            
            # Check if destination contains 'custom_nodes'
            dest_path = item.get('dest', '') or item.get('path', '')
            if 'custom_nodes' in dest_path.lower():
                return True
        
        return False
    
    def get_download_summary(self) -> Dict:
        """Get summary of downloads needed."""
        existing = self.check_existing_files()
        
        total_items = len(self.manifest['items'])
        bundled = sum(1 for i in self.manifest['items'] if i['source'] == 'bundled')
        to_download = sum(1 for name, status in existing.items() if status['needs_download'])
        already_exist = sum(1 for status in existing.values() if not status['needs_download'])
        partial_downloads = sum(1 for status in existing.values() if status.get('partial_exists', False))
        
        # Calculate total size from metadata or items
        metadata_size = self.manifest.get('metadata', {}).get('total_size_mb')
        if metadata_size:
            total_size = metadata_size
        else:
            total_size = sum(
                item.get('size_mb', 0)
                for item in self.manifest['items']
                if item['source'] != 'bundled' and existing.get(item['name'], {}).get('needs_download', False)
            )
        
        return {
            'total_items': total_items,
            'bundled': bundled,
            'to_download': to_download,
            'already_exist': already_exist,
            'partial_downloads': partial_downloads,
            'total_download_size_mb': total_size,
            'gated_models': [
                i['name'] for i in self.manifest['items']
                if i.get('gated', False) and existing.get(i['name'], {}).get('needs_download', False)
            ]
        }
    
    def print_summary(self):
        """Print download summary."""
        summary = self.get_download_summary()
        package = self.manifest.get('package', {})
        metadata = self.manifest.get('metadata', {})
        
        print("\n" + "=" * 60)
        print("MANIFEST DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"Package: {package.get('name', 'unknown')} v{package.get('version', '?')}")
        print(f"Description: {package.get('description', 'N/A')}")
        if metadata.get('details'):
            print(f"Details: {metadata['details']}")
        print(f"\nTotal items: {summary['total_items']}")
        print(f"  Bundled in ZIP: {summary['bundled']}")
        print(f"  Already exist: {summary['already_exist']}")
        print(f"  To download: {summary['to_download']}")
        if summary['partial_downloads'] > 0:
            print(f"  Partial downloads: {summary['partial_downloads']} (can be resumed)")
        print(f"Total download size: {summary['total_download_size_mb']:.1f} MB")
        if metadata.get('estimated_time'):
            print(f"Estimated time: {metadata['estimated_time']}")
        
        if summary['gated_models']:
            print(f"\n⚠ Gated models (require license acceptance):")
            for model in summary['gated_models']:
                print(f"  - {model}")
            if not self.hf_token:
                print("\n  Set HF_TOKEN environment variable to download gated models")
        
        if metadata.get('tags'):
            print(f"\nTags: {', '.join(metadata['tags'])}")
        
        print("=" * 60 + "\n")
    
    def cleanup_partial_downloads(self):
        """Clean up all partial download files."""
        if not self.partial_download_dir.exists():
            return
        
        partial_files = list(self.partial_download_dir.glob("*.partial"))
        if partial_files:
            self.log(f"Cleaning up {len(partial_files)} partial download(s)...")
            for partial_file in partial_files:
                partial_file.unlink()
            self.log("✓ Partial downloads cleaned up")
        else:
            self.log("No partial downloads to clean up")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ComfyUI Manifest Handler')
    parser.add_argument('manifest', type=Path, help='Path to manifest file (YAML or JSON)')
    parser.add_argument('--comfy-path', type=Path, default=Path('C:/ComfyUI'),
                       help='Path to ComfyUI installation (default: C:/ComfyUI)')
    parser.add_argument('--log-file', type=Path, help='Path to log file')
    parser.add_argument('--required-only', action='store_true',
                       help='Only download required items')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download all files')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip checksum verification')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview actions without downloading')
    parser.add_argument('--sequential', action='store_true',
                       help='Disable parallel downloads')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel download workers (default: 4)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Disable resume capability for downloads')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up partial download files and exit')
    
    args = parser.parse_args()
    
    # Initialize handler
    handler = ManifestHandler(
        manifest_path=args.manifest,
        comfy_path=args.comfy_path,
        log_file=args.log_file,
        max_workers=args.workers,
        resume_downloads=not args.no_resume
    )
    
    # Cleanup mode
    if args.cleanup:
        handler.cleanup_partial_downloads()
        sys.exit(0)
    
    # Load and validate
    handler.load_manifest()
    handler.validate_manifest()
    handler.print_summary()
    
    # Download items
    handler.download_items(
        skip_existing=not args.force,
        required_only=args.required_only,
        verify_checksums=not args.no_verify,
        dry_run=args.dry_run,
        parallel=not args.sequential
    )
    
    print("\n✓ All downloads completed!")