import argparse
import subprocess
import time
import requests
import json
import random
import uuid
import os
import sys
import zipfile
import shutil
import signal
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Thread, Event, Lock
from queue import Queue
from datetime import datetime

def queue_prompt(prompt, client_id, server_address):
    data = {"prompt": prompt, "client_id": client_id}
    response = requests.post(f"http://{server_address}/prompt", json=data)
    if response.status_code != 200:
        raise Exception(f"Failed to queue prompt: {response.text}")
    return response.json()["prompt_id"]

def wait_for_completion(prompt_id, server_address):
    start_time = time.time()
    while True:
        response = requests.get(f"http://{server_address}/history/{prompt_id}")
        if response.status_code == 200:
            history = response.json()
            if history:
                print(f"Prompt {prompt_id} completed in {time.time() - start_time:.2f} seconds")
                return history
        time.sleep(1)

def apply_overrides(workflow, override_path, log_file=None):
    if not override_path or not os.path.exists(override_path):
        return workflow, []
    try:
        with open(override_path, 'r', encoding='utf-8') as f:
            overrides = json.load(f).get("overrides", {})
        print(f"Override file specified: {override_path}")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Override file specified: {override_path}\n")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error: Failed to parse override file {override_path}: {e}")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Error: Failed to parse override file {override_path}: {e}\n")
        raise
    modified_workflow = json.loads(json.dumps(workflow))  # Deep copy
    applied_overrides = []  # Track overrides for summary
    for override_key, override in overrides.items():
        override_item = override.get("override_item")
        override_value = override.get("override_value")
        restrict = override.get("restrict", {})
        if not override_item or override_value is None:
            print(f"Warning: Skipping invalid override {override_key}: missing override_item or override_value")
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Warning: Skipping invalid override {override_key}: missing override_item or override_value\n")
            continue
        matched_nodes = []
        for node_id, node in modified_workflow.items():
            # Check if node matches restriction criteria
            matches = True
            if restrict:
                for restrict_key, restrict_value in restrict.items():
                    if restrict_key == "id":
                        if node_id != str(restrict_value):  # Ensure string comparison
                            matches = False
                            break
                    elif restrict_key in node.get("_meta", {}):
                        if node["_meta"].get(restrict_key) != restrict_value:
                            matches = False
                            break
                    elif restrict_key in node:
                        if node.get(restrict_key) != restrict_value:
                            matches = False
                            break
                    else:
                        matches = False
                        break
            if matches:
                node_title = node.get("_meta", {}).get("title", "Untitled")
                if override_item == "bypass":
                    if override_value is True:
                        # Add or update bypass: true
                        modified_workflow[node_id]["bypass"] = True
                        print(f"Applying override {override_key}: setting bypass to true in node {node_id} ({node_title})")
                        if log_file:
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(f"Applying override {override_key}: setting bypass to true in node {node_id} ({node_title})\n")
                        matched_nodes.append(f"{node_id} ({node_title})")
                    elif override_value is False and node.get("bypass", False) is True:
                        # Only update to false if bypass exists and is true
                        modified_workflow[node_id]["bypass"] = False
                        print(f"Applying override {override_key}: setting bypass to false in node {node_id} ({node_title})")
                        if log_file:
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(f"Applying override {override_key}: setting bypass to false in node {node_id} ({node_title})\n")
                        matched_nodes.append(f"{node_id} ({node_title})")
                elif override_item in node.get("inputs", {}):
                    # Handle regular input overrides
                    print(f"Applying override {override_key}: setting {override_item} to {override_value} in node {node_id} ({node_title})")
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"Applying override {override_key}: setting {override_item} to {override_value} in node {node_id} ({node_title})\n")
                    modified_workflow[node_id]["inputs"][override_item] = override_value
                    matched_nodes.append(f"{node_id} ({node_title})")
        if matched_nodes:
            applied_overrides.append({
                "key": override_key,
                "item": override_item,
                "value": override_value,
                "nodes": matched_nodes
            })
        else:
            print(f"Warning: Override {override_key} matched no nodes")
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Warning: Override {override_key} matched no nodes\n")
    return modified_workflow, applied_overrides

def load_workflow(workflow_path):
    if not os.path.exists(workflow_path):
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        if not isinstance(workflow, dict):
            raise ValueError(f"Error: {workflow_path} does not contain a valid JSON object (expected a dictionary).")
        for node_id in workflow:
            if not isinstance(node_id, str):
                raise ValueError(f"Error: Invalid node ID {node_id} in {workflow_path}. Node IDs must be strings.")
            if not isinstance(workflow[node_id], dict):
                raise ValueError(f"Error: Node {node_id} in {workflow_path} is not a valid node dictionary.")
            # Check for bypassed nodes
            if workflow[node_id].get("bypass", False):
                node_title = workflow[node_id].get("_meta", {}).get("title", "Untitled")
                print(f"Node {node_id} ({node_title}) is bypassed")
        # Validate KSampler and KSamplerAdvanced parameters
        for node_id, node in workflow.items():
            class_type = node.get("class_type")
            if class_type in ["KSampler", "KSamplerAdvanced"]:
                inputs = node.get("inputs", {})
                if class_type == "KSampler":
                    steps = inputs.get("steps")
                    start_step = inputs.get("start_step", 0)
                    last_step = inputs.get("last_step", steps)
                    cfg = inputs.get("cfg")
                    denoise = inputs.get("denoise")
                    sampler_name = inputs.get("sampler_name")
                    scheduler = inputs.get("scheduler")
                    print(f"DEBUG: KSampler node {node_id} - steps: {steps}, start_step: {start_step}, last_step: {last_step}, cfg: {cfg}, denoise: {denoise}, sampler_name: {sampler_name}, scheduler: {scheduler}")
                    if not isinstance(steps, int) or steps <= 0:
                        raise ValueError(f"Invalid KSampler steps ({steps}) in node {node_id}. Must be a positive integer (e.g., 20).")
                    if not isinstance(start_step, int) or start_step < 0:
                        raise ValueError(f"Invalid KSampler start_step ({start_step}) in node {node_id}. Must be a non-negative integer.")
                    if last_step is not None and (not isinstance(last_step, int) or last_step <= start_step or last_step > steps):
                        raise ValueError(f"Invalid KSampler last_step ({last_step}) in node {node_id}. Must be an integer > start_step and <= steps.")
                    if not isinstance(cfg, (int, float)) or cfg <= 0:
                        raise ValueError(f"Invalid KSampler cfg ({cfg}) in node {node_id}. Must be a positive number.")
                    if not isinstance(denoise, (int, float)) or denoise < 0 or denoise > 1:
                        raise ValueError(f"Invalid KSampler denoise ({denoise}) in node {node_id}. Must be between 0 and 1.")
                    if not sampler_name:
                        raise ValueError(f"Missing KSampler sampler_name in node {node_id}.")
                    if not scheduler:
                        raise ValueError(f"Missing KSampler scheduler in node {node_id}.")
                elif class_type == "KSamplerAdvanced":
                    steps = inputs.get("steps")
                    cfg = inputs.get("cfg")
                    sampler_name = inputs.get("sampler_name")
                    scheduler = inputs.get("scheduler")
                    start_at_step = inputs.get("start_at_step")
                    end_at_step = inputs.get("end_at_step")
                    print(f"DEBUG: KSamplerAdvanced node {node_id} - steps: {steps}, cfg: {cfg}, sampler_name: {sampler_name}, scheduler: {scheduler}, start_at_step: {start_at_step}, end_at_step: {end_at_step}")
                    if not isinstance(steps, int) or steps <= 0:
                        raise ValueError(f"Invalid KSamplerAdvanced steps ({steps}) in node {node_id}. Must be a positive integer (e.g., 20).")
                    if not isinstance(cfg, (int, float)) or cfg <= 0:
                        raise ValueError(f"Invalid KSamplerAdvanced cfg ({cfg}) in node {node_id}. Must be a positive number.")
                    if not sampler_name:
                        raise ValueError(f"Missing KSamplerAdvanced sampler_name in node {node_id}.")
                    if not scheduler:
                        raise ValueError(f"Missing KSamplerAdvanced scheduler in node {node_id}.")
                    if not isinstance(start_at_step, int) or start_at_step < 0:
                        raise ValueError(f"Invalid KSamplerAdvanced start_at_step ({start_at_step}) in node {node_id}. Must be a non-negative integer.")
                    if not isinstance(end_at_step, int) or end_at_step <= start_at_step:
                        raise ValueError(f"Invalid KSamplerAdvanced end_at_step ({end_at_step}) in node {node_id}. Must be an integer > start_at_step.")
        return workflow
    except UnicodeDecodeError as e:
        print(f"Error: Failed to decode {workflow_path}. Ensure the file is encoded in UTF-8.")
        raise e
    except json.JSONDecodeError as e:
        print(f"Error: {workflow_path} is not a valid JSON file.")
        raise e

def load_baseconfig(comfy_path, temp_dir=None, log_file=None):
    baseconfig_path = Path(comfy_path) / "baseconfig.json"
    paths_to_check = [baseconfig_path]
    if temp_dir:
        paths_to_check.append(Path(temp_dir) / "baseconfig.json")
    selected_path = None
    for path in paths_to_check:
        if path.exists():
            selected_path = path
            break
    if not selected_path:
        print(f"Warning: baseconfig.json not found in {comfy_path} or temporary directory. Using default values.")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Warning: baseconfig.json not found in {comfy_path} or temporary directory. Using default values.\n")
        return {"NUM_INSTANCES": 1, "GENERATIONS": 1}
    try:
        with open(selected_path, 'r', encoding='utf-8') as f:
            baseconfig = json.load(f)
        if not isinstance(baseconfig, dict):
            raise ValueError(f"Error: baseconfig.json at {selected_path} does not contain a valid JSON object (expected a dictionary).")
        config_values = {
            "NUM_INSTANCES": baseconfig.get("NUM_INSTANCES", 1),
            "GENERATIONS": baseconfig.get("GENERATIONS", 1)
        }
        print(f"Using values from baseconfig.json at {selected_path}: NUM_INSTANCES={config_values['NUM_INSTANCES']}, GENERATIONS={config_values['GENERATIONS']}")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Using values from baseconfig.json at {selected_path}: NUM_INSTANCES={config_values['NUM_INSTANCES']}, GENERATIONS={config_values['GENERATIONS']}\n")
        return config_values
    except Exception as e:
        print(f"Warning: Failed to load baseconfig.json at {selected_path}: {e}. Using default values.")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Warning: Failed to load baseconfig.json at {selected_path}: {e}. Using default values.\n")
        return {"NUM_INSTANCES": 1, "GENERATIONS": 1}

def extract_zip(zip_path, extract_to, comfy_path, extract_minimal=False, log_file=None):
    os.makedirs(extract_to, exist_ok=True)
    extract_to_path = Path(extract_to).resolve()
    print(f"Extracting files to temporary directory: {extract_to_path}")
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Extracting files to temporary directory: {extract_to_path}\n")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        json_files = [f for f in zipf.namelist() if f.lower().endswith('.json')]
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {zip_path}")
        # Prioritize workflow.json at the root of the ZIP
        root_json_files = [f for f in json_files if not (os.sep in f or '/' in f)]
        target_file = next((f for f in root_json_files if f.lower() == 'workflow.json'), None)
        if not target_file:
            raise FileNotFoundError(f"No workflow.json found at the root of {zip_path}")
        if extract_minimal:
            # Extract all .json files when --extract_minimal is used
            for json_file in json_files:
                zipf.extract(json_file, extract_to)
                print(f"Extracted file: {Path(json_file).name}")
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"Extracted file: {Path(json_file).name} to {extract_to}\n")
        else:
            # Extract all files
            for file_name in zipf.namelist():
                zipf.extract(file_name, extract_to)
                print(f"Extracted file: {Path(file_name).name}")
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"Extracted file: {Path(file_name).name} to {extract_to}\n")
        
        # Copy only .json files at the root of the temporary directory to ComfyUI/user/default/workflows/<zip_basename>
        zip_basename = Path(zip_path).stem
        workflow_dir = comfy_path / "user" / "default" / "workflows" / zip_basename
        os.makedirs(workflow_dir, exist_ok=True)
        # Filter json_files to only include files at the root (no directory separators)
        root_json_files = [f for f in json_files if not (os.sep in f or '/' in f)]
        for json_file in root_json_files:
            src_path = Path(extract_to) / json_file
            if src_path.exists():  # Only copy files that were extracted
                dest_path = workflow_dir / Path(json_file).name
                shutil.copy2(src_path, dest_path)
                print(f"Saving workflow to: {dest_path}")
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"Saving workflow to: {dest_path}\n")
        # Run pre.py if it exists
        pre_script_path = Path(extract_to) / "pre.py"
        if pre_script_path.exists():
            print(f"Running {pre_script_path}...")
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Running {pre_script_path}...\n")
            run_python_script(pre_script_path, extract_to)
            print(f"Successfully ran {pre_script_path}")
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Successfully ran {pre_script_path}\n")
        # Check for ComfyUI/custom_nodes and install requirements.txt if present
        custom_nodes_dir = Path(extract_to) / "ComfyUI" / "custom_nodes"
        if custom_nodes_dir.exists() and custom_nodes_dir.is_dir():
            for node_dir in custom_nodes_dir.iterdir():
                if node_dir.is_dir():
                    requirements_path = node_dir / "requirements.txt"
                    if requirements_path.exists():
                        node_name = node_dir.name
                        print(f"Installing requirements for custom node {node_name}...")
                        if log_file:
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(f"Installing requirements for custom node {node_name}...\n")
                        try:
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], cwd=node_dir)
                            print(f"Successfully installed requirements for custom node {node_name}")
                            if log_file:
                                with open(log_file, 'a', encoding='utf-8') as f:
                                    f.write(f"Successfully installed requirements for custom node {node_name}\n")
                        except subprocess.CalledProcessError as e:
                            print(f"Failed to install requirements for custom node {node_name}: {e}")
                            if log_file:
                                with open(log_file, 'a', encoding='utf-8') as f:
                                    f.write(f"Failed to install requirements for custom node {node_name}: {e}\n")
                            raise
        # Copy subfolders from extracted ComfyUI/ to local comfy_path
        extracted_comfy_dir = Path(extract_to) / "ComfyUI"
        if extracted_comfy_dir.exists() and extracted_comfy_dir.is_dir() and not extract_minimal:
            for subfolder in extracted_comfy_dir.iterdir():
                if subfolder.is_dir():
                    folder_name = subfolder.name
                    dest_folder = comfy_path / folder_name
                    print(f"Begin folder copy: {folder_name}")
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"Begin folder copy: {folder_name}\n")
                    shutil.copytree(subfolder, dest_folder, dirs_exist_ok=True)
                    print(f"Copied folder {folder_name} to {dest_folder}")
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"Copied folder {folder_name} to {dest_folder}\n")
        # Run post.py if it exists
        post_script_path = Path(extract_to) / "post.py"
        if post_script_path.exists():
            print(f"Running {post_script_path}...")
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Running {post_script_path}...\n")
            run_python_script(post_script_path, extract_to)
            print(f"Successfully ran {post_script_path}")
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Successfully ran {post_script_path}\n")
        return Path(extract_to) / target_file

def run_python_script(script_path, cwd):
    if script_path.exists():
        print(f"Running {script_path}...")
        try:
            subprocess.check_call([sys.executable, str(script_path)], cwd=cwd)
            print(f"Successfully ran {script_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_path}: {e}")
            raise

def interrupt_process(port):
    """Attempt to gracefully interrupt a ComfyUI instance via its API."""
    server_address = f"127.0.0.1:{port}"
    try:
        response = requests.post(f"http://{server_address}/interrupt")
        if response.status_code == 200:
            print(f"Sent interrupt to {server_address}")
            return True
        else:
            print(f"Failed to send interrupt to {server_address}: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"Error sending interrupt to {server_address}: {e}")
        return False

def check_server_ready(port, timeout=60, interval=2):
    """Check if the ComfyUI server is ready by polling its root endpoint."""
    server_address = f"127.0.0.1:{port}"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{server_address}/")
            if response.status_code == 200:
                print(f"Server on port {port} is ready")
                return True
        except requests.ConnectionError:
            time.sleep(interval)
    print(f"Error: Server on port {port} not ready after {timeout} seconds")
    return False

def capture_execution_times(proc, output_queue, capture_event, print_lock, log_file=None):
    """Capture execution times and handle progress bars from ComfyUI output."""
    pattern = re.compile(r"Prompt executed in (\d+\.\d+) seconds")
    progress_pattern = re.compile(r"\d+%\|.*?\| \d+/\d+ \[\d+:\d+.*?\d+\.\d+(?:it/s|s/it)\]")
    error_pattern = re.compile(r"!!! Exception during processing !!!")
    progress_buffer = []
    error_detected = False
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            try:
                line = line.decode('utf-8').strip()
            except AttributeError:
                line = line.strip()
            except UnicodeDecodeError:
                line = line.decode('cp1252', errors='ignore').strip()
            line = line.replace('█', '#').replace('▌', '#').replace('▎', '#')
            line = line.replace('â–ˆ', '#').replace('â–Œ','>')
            with print_lock:
                if error_pattern.search(line):
                    error_detected = True
                    print(line, flush=True)
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(line + '\n')
                elif progress_pattern.search(line):
                    progress_buffer.append(line)
                    print(f"\r{' ' * 120}\r{progress_buffer[-1]}", end='', flush=True)
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(progress_buffer[-1] + '\n')
                else:
                    if progress_buffer:
                        print(f"\n{progress_buffer[-1]}", flush=True)
                        if log_file:
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(progress_buffer[-1] + '\n')
                        progress_buffer = []
                    print(line, flush=True)
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(line + '\n')
            if capture_event.is_set() and not error_detected:
                match = pattern.search(line)
                if match:
                    exec_time = float(match.group(1))
                    if exec_time > 1.0:  # Ignore invalid times
                        print("\n")
                        if log_file:
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write("\n")
                        output_queue.put(exec_time)
            if error_pattern.search(line):
                print("\nDEBUG: Detected ComfyUI processing error", flush=True)
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write("DEBUG: Detected ComfyUI processing error\n")
    with print_lock:
        if progress_buffer:
            print(f"\n{progress_buffer[-1]}", flush=True)
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(progress_buffer[-1] + '\n')

def main():
    # Set console encoding to UTF-8
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Run multiple ComfyUI instances for simultaneous image generations.")
    parser.add_argument("-n", "--num_instances", type=int, default=1, help="Number of simultaneous ComfyUI instances.")
    parser.add_argument("-c", "--comfy_path", required=True, help="Path to the ComfyUI directory.")
    parser.add_argument("-w", "--workflow_path", required=True, help="Path to the JSON workflow file, ZIP file, or directory containing workflow.json.")
    parser.add_argument("-g", "--generations", type=int, default=1, help="Number of generations per instance.")
    parser.add_argument("-e", "--extract_minimal", action="store_true", help="Extract only JSON files from ZIP.")
    parser.add_argument("-r", "--run_default", action="store_true", help="Load default values for num_instances and generations from baseconfig.json if not provided.")
    parser.add_argument("-o", "--override", type=str, help="Path to JSON file with override parameters.")
    parser.add_argument("-l", "--log", nargs='?', const=True, default=False, help="Log console output to a file. If no path is provided, use workflow basename + timestamp (yymmdd_epochtime.txt). If a path is provided, use it as is (if file) or append timestamp (if directory).")
    parser.add_argument("-t", "--temp_path", type=str, help="Path to parent directory for temporary folder (defaults to ComfyUI/temp).")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Additional arguments to pass to main.py (e.g., --cuda-device 1).")
    args = parser.parse_args()
    # Validate arguments
    if args.num_instances < 1:
        print("Error: --num_instances must be at least 1.")
        sys.exit(1)
    if args.generations < 1:
        print("Error: --generations must be at least 1.")
        sys.exit(1)
    # Handle extra_args
    extra_args = args.extra_args if args.extra_args else []
    # Handle logging
    log_file = None
    if args.log is not False:
        workflow_basename = Path(args.workflow_path).stem
        timestamp = datetime.now().strftime("%y%m%d_") + str(int(time.time()))
        if args.log is True:
            # Default log file: workflow basename + timestamp
            log_file = Path(f"{workflow_basename}_{timestamp}.txt").resolve()
        else:
            # User provided a path
            log_path = Path(args.log).resolve()
            if log_path.is_dir():
                log_file = log_path / f"{workflow_basename}_{timestamp}.txt"
            else:
                log_file = log_path
        print(f"Logging to: {log_file}")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Starting run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    # Initialize variables
    temp_dir = None
    print_lock = Lock()
    comfy_path = Path(args.comfy_path).resolve()
    workflow_path = Path(args.workflow_path).resolve()
    processes = []  # Initialize early to avoid UnboundLocalError
    created_temp_dir = False  # Track if we created the unique temp dir
    try:
        # Set default temp_dir to comfy_path/temp/<unique>
        temp_base = Path(args.temp_path).resolve() if args.temp_path else comfy_path / "temp"
        temp_dir = temp_base / f"temp_{uuid.uuid4().hex}"
        os.makedirs(temp_dir, exist_ok=True)
        created_temp_dir = True  # We created the unique temp dir
        # Handle ZIP, JSON, or directory for workflow
        if workflow_path.is_dir():
            workflow_path = workflow_path / "workflow.json"
            if not workflow_path.exists():
                print(f"Error: workflow.json not found in directory {args.workflow_path}")
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"Error: workflow.json not found in directory {args.workflow_path}\n")
                sys.exit(1)
        elif workflow_path.suffix.lower() == '.zip':
            workflow_path = extract_zip(workflow_path, temp_dir, comfy_path, args.extract_minimal, log_file)
        elif workflow_path.suffix.lower() != '.json':
            print(f"Error: --workflow_path must be a .json file, a .zip file, or a directory containing workflow.json.")
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write("Error: --workflow_path must be a .json file, a .zip file, or a directory containing workflow.json.\n")
            sys.exit(1)
        print(f"Using workflow: {workflow_path}")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Using workflow: {workflow_path}\n")
        # Handle -r flag: Load defaults from baseconfig.json if not provided
        if args.run_default:
            baseconfig = load_baseconfig(comfy_path, temp_dir, log_file)
            if args.num_instances == 1:  # Only override if default value
                args.num_instances = baseconfig["NUM_INSTANCES"]
            if args.generations == 1:  # Only override if default value
                args.generations = baseconfig["GENERATIONS"]
        num_instances = args.num_instances
        generations = args.generations
        # Load and validate workflow
        main_workflow = load_workflow(workflow_path)
        # Apply overrides if provided
        main_workflow, main_applied_overrides = apply_overrides(main_workflow, args.override, log_file)
        # Use main workflow for warmup
        warmup_workflow = main_workflow
        warmup_applied_overrides = main_applied_overrides
        processes = []
        ports = []
        base_port = 8188
        output_queues = [Queue() for _ in range(num_instances)]
        capture_events = [Event() for _ in range(num_instances)]
        print(f"Starting {num_instances} ComfyUI instances...")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Starting {num_instances} ComfyUI instances...\n")
        for i in range(num_instances):
            port = base_port + i
            ports.append(port)
            command = ["python", "main.py", "--port", str(port), "--listen", "127.0.0.1"]
            command.extend(extra_args)
            proc = subprocess.Popen(
                command,
                cwd=comfy_path,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            processes.append(proc)
            print(f"Started instance {i+1} on port {port} (PID: {proc.pid}) with command: {' '.join(command)}")
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Started instance {i+1} on port {port} (PID: {proc.pid}) with command: {' '.join(command)}\n")
            Thread(target=capture_execution_times, args=(proc, output_queues[i], capture_events[i], print_lock, log_file), daemon=True).start()
        # Wait for servers to start
        for port in ports:
            if not check_server_ready(port, timeout=60):
                error_msg = f"ComfyUI server on port {port} failed to start."
                print(error_msg)
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(error_msg + "\n")
                raise RuntimeError(error_msg)
        client_ids = [str(uuid.uuid4()) for _ in ports]
        def generation_task(idx, gen, is_warmup=False):
            port = ports[idx]
            server_address = f"127.0.0.1:{port}"
            client_id = client_ids[idx]
            workflow = warmup_workflow if is_warmup else main_workflow
            prompt = json.loads(json.dumps(workflow))
            for node_id, node in prompt.items():
                class_type = node.get("class_type")
                inputs = node.get("inputs", {})
                # Randomize seed for KSampler
                if class_type == "KSampler" and "seed" in inputs:
                    prompt[node_id]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
                # Randomize noise_seed for KSamplerAdvanced
                elif class_type == "KSamplerAdvanced" and "noise_seed" in inputs:
                    prompt[node_id]["inputs"]["noise_seed"] = random.randint(0, 2**32 - 1)
                # Randomize value for PrimitiveInt nodes with title matching "Random" (case-insensitive)
                elif class_type == "PrimitiveInt" and "value" in inputs:
                    meta = node.get("_meta", {})
                    title = meta.get("title", "")
                    if re.search(r"Random", title, re.IGNORECASE):
                        prompt[node_id]["inputs"]["value"] = random.randint(0, 2**32 - 1)
            try:
                prompt_id = queue_prompt(prompt, client_id, server_address)
                if is_warmup:
                    print(f"Queued warmup on instance {idx+1} (prompt_id: {prompt_id})")
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"Queued warmup on instance {idx+1} (prompt_id: {prompt_id})\n")
                else:
                    capture_events[idx].set()
                    print(f"Queued generation {gen+1} of {generations} on instance {idx+1} (prompt_id: {prompt_id})")
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"Queued generation {gen+1} of {generations} on instance {idx+1} (prompt_id: {prompt_id})\n")
                history = wait_for_completion(prompt_id, server_address)
                if is_warmup:
                    print(f"Completed warmup on instance {idx+1}")
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"Completed warmup on instance {idx+1}\n")
                else:
                    print(f"Completed generation {gen+1} of {generations} on instance {idx+1}")
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"Completed generation {gen+1} of {generations} on instance {idx+1}\n")
            except Exception as e:
                error_msg = f"Error during {'warmup' if is_warmup else f'generation {gen+1}'} on instance {idx+1}: {e}"
                if "ZeroDivisionError" in str(e) or "integer division or modulo by zero" in str(e):
                    error_msg += "\nThis error suggests an issue with the KSampler or KSamplerAdvanced node in workflow.json. Please verify 'steps' (> 0), 'start_step' or 'start_at_step' (>= 0), 'last_step' or 'end_at_step' (> start_step/start_at_step and <= steps for KSampler), 'cfg' (> 0), 'denoise' (0–1 for KSampler), 'sampler_name' (e.g., 'dpmpp_2m' or 'euler'), and 'scheduler' (e.g., 'normal' or 'simple'). Alternatively, test the workflow in the ComfyUI GUI."
                    print(error_msg)
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(error_msg + "\n")
                    raise RuntimeError(error_msg) from e
                raise e
            finally:
                if not is_warmup:
                    capture_events[idx].clear()
        # Warmup step
        print("Performing warmup...")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("Performing warmup...\n")
        with ThreadPoolExecutor(max_workers=num_instances) as executor:
            futures = [executor.submit(generation_task, idx, -1, is_warmup=True) for idx in range(num_instances)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception during warmup: {e}")
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"Exception during warmup: {e}\n")
                    raise
        print("Warmup completed.")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("Warmup completed.\n")
        # Clear queues to ensure no stale data
        for queue in output_queues:
            while not queue.empty():
                queue.get()
        # Capture start timestamp
        start_time = time.time()
        try:
            for gen in range(generations):
                print(f"Starting generation round {gen+1}/{generations}")
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"Starting generation round {gen+1}/{generations}\n")
                with ThreadPoolExecutor(max_workers=num_instances) as executor:
                    futures = [executor.submit(generation_task, idx, gen, is_warmup=False) for idx in range(num_instances)]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Exception in generation round {gen+1}: {e}")
                            if log_file:
                                with open(log_file, 'a', encoding='utf-8') as f:
                                    f.write(f"Exception in generation round {gen+1}: {e}\n")
                            raise
        except KeyboardInterrupt:
            print("Interrupted by user. Calculating partial metrics...")
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write("Interrupted by user. Calculating partial metrics...\n")
            raise
        # Capture end timestamp
        end_time = time.time()
        total_time = end_time - start_time
        total_images = num_instances * generations
        # Collect execution times
        execution_times = []
        for queue in output_queues:
            while not queue.empty():
                execution_times.append(queue.get())
        total_execution_time = sum(execution_times)
        avg_execution_time = total_execution_time / len(execution_times) if execution_times else 0
        if total_time > 0 and total_images > 0:
            images_per_minute = total_images / (total_time / 60)
            avg_time_per_image = total_time / total_images
        else:
            images_per_minute = 0
            avg_time_per_image = 0
        # Prepare results summary
        print('####_RESULTS_SUMMARY_####\n')
        print(f"Total time to generate {total_images} images: {total_time:.2f} seconds")
        print(f"Number of images per minute: {images_per_minute:.2f}")
        print(f"Average time (secs) per image: {avg_time_per_image:.2f}")
        print(f"Total Execution Time (main generations): {total_execution_time:.2f} seconds")
        print(f"Average Execution Time Per Image (main generations): {avg_execution_time:.2f} seconds")
        # Add override summary
        if main_applied_overrides:
            print("\nApplied Overrides:")
            print("  Main Workflow Overrides:")
            for override in main_applied_overrides:
                print(f"    - {override['key']}: Set {override['item']} to {override['value']} in nodes {override['nodes']}")
        else:
            print("\nNo overrides applied.")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write('####_RESULTS_SUMMARY_####\n')
                f.write(f"Total time to generate {total_images} images: {total_time:.2f} seconds\n")
                f.write(f"Number of images per minute: {images_per_minute:.2f}\n")
                f.write(f"Average time (secs) per image: {avg_time_per_image:.2f}\n")
                f.write(f"Total Execution Time (main generations): {total_execution_time:.2f} seconds\n")
                f.write(f"Average Execution Time Per Image (main generations): {avg_execution_time:.2f} seconds\n")
                if main_applied_overrides:
                    f.write("\nApplied Overrides:\n")
                    f.write("  Main Workflow Overrides:\n")
                    for override in main_applied_overrides:
                        f.write(f"    - {override['key']}: Set {override['item']} to {override['value']} in nodes {override['nodes']}\n")
                else:
                    f.write("\nNo overrides applied.\n")
    except KeyboardInterrupt:
        print("Cleaning up after user interrupt...")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("Cleaning up after user interrupt...\n")
        # Calculate partial metrics
        end_time = time.time()
        total_time = end_time - start_time
        total_images = num_instances * sum(1 for queue in output_queues if not queue.empty())
        execution_times = []
        for queue in output_queues:
            while not queue.empty():
                execution_times.append(queue.get())
        total_execution_time = sum(execution_times)
        avg_execution_time = total_execution_time / len(execution_times) if execution_times else 0
        if total_time > 0 and total_images > 0:
            images_per_minute = total_images / (total_time / 60)
            avg_time_per_image = total_time / total_images
        else:
            images_per_minute = 0
            avg_time_per_image = 0
        print(f"Partial metrics for {total_images} images:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Number of images per minute: {images_per_minute:.2f}")
        print(f"Average time (secs) per image: {avg_time_per_image:.2f}")
        print(f"Total Execution Time (main generations): {total_execution_time:.2f} seconds")
        print(f"Average Execution Time Per Image (main generations): {avg_execution_time:.2f} seconds")
        # Add override summary for partial metrics
        if main_applied_overrides:
            print("\nApplied Overrides:")
            print("  Main Workflow Overrides:")
            for override in main_applied_overrides:
                print(f"    - {override['key']}: Set {override['item']} to {override['value']} in nodes {override['nodes']}")
        else:
            print("\nNo overrides applied.")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Partial metrics for {total_images} images:\n")
                f.write(f"Total time: {total_time:.2f} seconds\n")
                f.write(f"Number of images per minute: {images_per_minute:.2f}\n")
                f.write(f"Average time (secs) per image: {avg_time_per_image:.2f}\n")
                f.write(f"Total Execution Time (main generations): {total_execution_time:.2f} seconds\n")
                f.write(f"Average Execution Time Per Image (main generations): {avg_execution_time:.2f} seconds\n")
                if main_applied_overrides:
                    f.write("\nApplied Overrides:\n")
                    f.write("  Main Workflow Overrides:\n")
                    for override in main_applied_overrides:
                        f.write(f"    - {override['key']}: Set {override['item']} to {override['value']} in nodes {override['nodes']}\n")
                else:
                    f.write("\nNo overrides applied.\n")
    finally:
        # Clean up: Terminate processes and remove temp directory
        for i, proc in enumerate(processes):
            port = base_port + i
            try:
                if proc.poll() is None:
                    if interrupt_process(port):
                        time.sleep(5)
                    if proc.poll() is None:
                        with open(os.devnull, 'w') as devnull:
                            proc.stderr = devnull
                            proc.terminate()
                            proc.wait(timeout=5)
            except Exception as e:
                print(f"Error terminating process {proc.pid}: {e}")
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"Error terminating process {proc.pid}: {e}\n")
                with open(os.devnull, 'w') as devnull:
                    proc.stderr = devnull
                    proc.terminate()
                    proc.wait(timeout=2)
        if temp_dir and temp_dir.exists() and created_temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Removed temporary directory: {temp_dir}")
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"Removed temporary directory: {temp_dir}\n")
        print("Done.")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Logfile written to: {log_file}\n")

if __name__ == "__main__":
    main()