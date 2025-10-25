import argparse
import subprocess
import time
import requests
import json
import uuid
import os
import sys
import yaml
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Thread, Event, Lock
from queue import Queue
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.workflow_manager import WorkflowManager
from core.package_manager import PackageManager

class YamlObject:
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.data = None
        self.load()

    def load(self):
        if os.path.exists(self.yaml_path):
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                self.data = yaml.safe_load(f) or {}
        else:
            self.data = {}

    def exists(self):
        return os.path.exists(self.yaml_path)

    def get(self, key, default=None):
        if self.data is None:
            self.load()
        keys = key.split('.')
        value = self.data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, key, value):
        if self.data is None:
            self.load()
        keys = key.split('.')
        d = self.data
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
            if not isinstance(d, dict):
                raise ValueError(f"Cannot set nested key '{key}' because intermediate '{k}' is not a dict")
        d[keys[-1]] = value
        self.save()

    def save(self):
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.data, f)

    def key_exists(self, key):
        if self.data is None:
            self.load()
        keys = key.split('.')
        value = self.data
        for k in keys:
            if not isinstance(value, dict):
                return False
            if k not in value:
                return False
            value = value[k]
        return True

class BenchmarkNodeManager:
    def __init__(self, custom_nodes_path):
        self.custom_nodes_path = custom_nodes_path
        self.benchmark_path = os.path.join(custom_nodes_path, 'comfyui-benchmark')
        self.exists = os.path.exists(self.benchmark_path) and os.path.isdir(self.benchmark_path)
        self.yaml = None
        if self.exists:
            yaml_path = os.path.join(self.benchmark_path, 'config.yaml')
            self.yaml = YamlObject(yaml_path)

def wait_for_completion(prompt_id, server_address):
    """Wait for a prompt to complete and return its history."""
    start_time = time.time()
    while True:
        response = requests.get(f"http://{server_address}/history/{prompt_id}")
        if response.status_code == 200:
            history = response.json()
            if history:
                print(f"Prompt {prompt_id} completed in {time.time() - start_time:.2f} seconds")
                return history
        time.sleep(1)

def load_baseconfig(comfy_path, temp_dir=None, log_file=None):
    """Load configuration from baseconfig.json."""
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

def check_server_running(port):
    """Check if a ComfyUI server is already running on the specified port."""
    server_address = f"127.0.0.1:{port}"
    try:
        response = requests.get(f"http://{server_address}/")
        if response.status_code == 200:
            print(f"ComfyUI server already running on port {port}")
            return True
    except requests.ConnectionError:
        return False
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
    parser.add_argument("-l", "--log", nargs='?', const=True, default=False, help="Log console output to a file.")
    parser.add_argument("-t", "--temp_path", type=str, help="Path to parent directory for temporary folder.")
    parser.add_argument("-p", "--port", type=int, default=8188, help="Starting base port for ComfyUI instances.")
    parser.add_argument("-u", "--use_main_workflow_only", action="store_true", help="Use workflow.json for warmup even if warmup.json exists.")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Additional arguments to pass to main.py.")
    args = parser.parse_args()

    # Validate arguments
    if args.num_instances < 1:
        print("Error: --num_instances must be at least 1.")
        sys.exit(1)
    if args.generations < 1:
        print("Error: --generations must be at least 1.")
        sys.exit(1)
    if args.port < 1024 or args.port > 65535:
        print("Error: --port must be between 1024 and 65535.")
        sys.exit(1)

    # Handle logging
    log_file = None
    if args.log is not False:
        workflow_basename = Path(args.workflow_path).stem
        timestamp = datetime.now().strftime("%y%m%d_") + str(int(time.time()))
        if args.log is True:
            log_file = Path(f"{workflow_basename}_{timestamp}.txt").resolve()
        else:
            log_path = Path(args.log).resolve()
            if log_path.is_dir():
                log_file = log_path / f"{workflow_basename}_{timestamp}.txt"
            else:
                log_file = log_path
        print(f"Logging to: {log_file}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Starting run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize variables
    workflow_path = Path(args.workflow_path).resolve()
    warmup_workflow_path = None
    comfy_path = Path(args.comfy_path).resolve()
    package_manager = None
    extra_args = args.extra_args if args.extra_args else []

    try:
        # Handle workflow path (ZIP, directory, or JSON)
        if workflow_path.is_dir():
            workflow_path = workflow_path / "workflow.json"
            if not args.use_main_workflow_only:
                warmup_workflow_path = workflow_path.parent / "warmup.json"
                if not warmup_workflow_path.exists():
                    warmup_workflow_path = workflow_path
            else:
                warmup_workflow_path = workflow_path
            if not workflow_path.exists():
                error_msg = f"Error: workflow.json not found in directory {args.workflow_path}"
                print(error_msg)
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(error_msg + "\n")
                sys.exit(1)
        elif workflow_path.suffix.lower() == '.zip':
            package_manager = PackageManager(
                zip_path=workflow_path,
                comfy_path=comfy_path,
                temp_path=args.temp_path,
                extract_minimal=args.extract_minimal,
                log_file=log_file
            )
            workflow_path = package_manager.extract_zip()
            if not args.use_main_workflow_only:
                warmup_workflow_path = workflow_path.parent / "warmup.json"
                if not warmup_workflow_path.exists():
                    warmup_workflow_path = workflow_path
            else:
                warmup_workflow_path = workflow_path
        elif workflow_path.suffix.lower() != '.json':
            error_msg = "Error: --workflow_path must be a .json file, a .zip file, or a directory containing workflow.json."
            print(error_msg)
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg + "\n")
            sys.exit(1)
        else:
            warmup_workflow_path = workflow_path

        # Initialize WorkflowManager
        workflow_manager = WorkflowManager(workflow_path=workflow_path, log_file=log_file)
        workflow_manager.load_workflow()
        warmup_workflow_manager = WorkflowManager(workflow_path=warmup_workflow_path, log_file=log_file)
        warmup_workflow_manager.load_workflow()

        # Apply overrides if provided
        if args.override:
            workflow_manager.apply_overrides(args.override)
            warmup_workflow_manager.apply_overrides(args.override)

        # Initialize BenchmarkNodeManager
        benchmark_node_manager = BenchmarkNodeManager(custom_nodes_path=comfy_path / "custom_nodes")

        # Handle -r flag: Load defaults from baseconfig.json
        num_instances = args.num_instances
        generations = args.generations
        if args.run_default:
            baseconfig = load_baseconfig(comfy_path, package_manager.temp_dir if package_manager else None, log_file)
            if args.num_instances == 1:
                num_instances = baseconfig["NUM_INSTANCES"]
            if args.generations == 1:
                generations = baseconfig["GENERATIONS"]

        # Initialize processes, ports, and queues
        processes = []
        ports = []
        base_port = args.port
        output_queues = [Queue() for _ in range(num_instances)]
        capture_events = [Event() for _ in range(num_instances)]
        print_lock = Lock()

        print(f"Starting {num_instances} ComfyUI instances...")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Starting {num_instances} ComfyUI instances...\n")

        for i in range(num_instances):
            port = base_port + i
            ports.append(port)
            if check_server_running(port):
                print(f"Using existing ComfyUI instance on port {port}")
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"Using existing ComfyUI instance on port {port}\n")
                processes.append(None)
                continue
            command = ["python", "main.py", "--port", str(port), "--listen", "127.0.0.1"] + extra_args
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
        for i, port in enumerate(ports):
            if processes[i] is None:
                continue
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
            try:
                # Use warmup_workflow_manager for warmup, workflow_manager for main generations
                manager = warmup_workflow_manager if is_warmup else workflow_manager
                workflow = manager.get_workflow(randomize_seeds=True)  # Assuming always randomize, adjust if needed for warmup

                if benchmark_node_manager.exists:
                    benchmark_node_id = None
                    for node_id, node in workflow.items():
                        meta = node.get("_meta", {})
                        title = meta.get("title", "")
                        if title == "Benchmark Workflow":
                            benchmark_node_id = node_id
                            break

                    if benchmark_node_id is None:
                        benchmark_node_id = str(len(workflow) + 1)
                        workflow[benchmark_node_id] = {
                            "class_type": "BenchmarkWorkflow",
                            "_meta": {"title": "Benchmark Workflow"},
                            "inputs": {
                                "capture_benchmark": True,
                                "outfile_postfix1": "",
                                "outfile_postfix2": ""
                            }
                        }
                        # Optional: log creation if desired
                        # print(f"Created new Benchmark Workflow node with ID: {benchmark_node_id} for instance {idx+1}")

                    node = workflow[benchmark_node_id]
                    if is_warmup:
                        node["inputs"]["outfile_postfix1"] = "_warmup_"
                        node["inputs"]["outfile_postfix2"] = f"{idx+1}"
                    else:
                        node["inputs"]["outfile_postfix1"] = f"_RUN_{gen+1}."
                        node["inputs"]["outfile_postfix2"] = f"{idx+1}"

                # Queue the prompt with the modified workflow
                data = {"prompt": workflow, "client_id": client_id}
                response = requests.post(f"http://{server_address}/prompt", json=data)
                if response.status_code != 200:
                    raise Exception(f"Failed to queue prompt: {response.text}")
                prompt_id = response.json()["prompt_id"]

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

        # Clear queues
        for queue in output_queues:
            while not queue.empty():
                queue.get()

        # Main generations
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

        # Calculate metrics
        end_time = time.time()
        total_time = end_time - start_time
        total_images = num_instances * generations
        execution_times = []
        for queue in output_queues:
            while not queue.empty():
                execution_times.append(queue.get())
        total_execution_time = sum(execution_times)
        avg_execution_time = total_execution_time / len(execution_times) if execution_times else 0
        images_per_minute = total_images / (total_time / 60) if total_time > 0 else 0
        avg_time_per_image = total_time / total_images if total_images > 0 else 0

        # Print results
        print('####_RESULTS_SUMMARY_####\n')
        print(f"Total time to generate {total_images} images: {total_time:.2f} seconds")
        print(f"Number of images per minute: {images_per_minute:.2f}")
        print(f"Average time (secs) per image: {avg_time_per_image:.2f}")
        print(f"Total Execution Time (main generations): {total_execution_time:.2f} seconds")
        print(f"Average Execution Time Per Image (main generations): {avg_execution_time:.2f} seconds")
        applied_overrides = workflow_manager.get_applied_overrides()
        if applied_overrides:
            print("\nApplied Overrides:")
            print("  Main Workflow Overrides:")
            for override in applied_overrides:
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
                if applied_overrides:
                    f.write("\nApplied Overrides:\n")
                    f.write("  Main Workflow Overrides:\n")
                    for override in applied_overrides:
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
        images_per_minute = total_images / (total_time / 60) if total_time > 0 else 0
        avg_time_per_image = total_time / total_images if total_images > 0 else 0
        print(f"Partial metrics for {total_images} images:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Number of images per minute: {images_per_minute:.2f}")
        print(f"Average time (secs) per image: {avg_time_per_image:.2f}")
        print(f"Total Execution Time (main generations): {total_execution_time:.2f} seconds")
        print(f"Average Execution Time Per Image (main generations): {avg_execution_time:.2f} seconds")
        applied_overrides = workflow_manager.get_applied_overrides()
        if applied_overrides:
            print("\nApplied Overrides:")
            print("  Main Workflow Overrides:")
            for override in applied_overrides:
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
                if applied_overrides:
                    f.write("\nApplied Overrides:\n")
                    f.write("  Main Workflow Overrides:\n")
                    for override in applied_overrides:
                        f.write(f"    - {override['key']}: Set {override['item']} to {override['value']} in nodes {override['nodes']}\n")
                else:
                    f.write("\nNo overrides applied.\n")

    finally:
        # Clean up processes
        for i, proc in enumerate(processes):
            if proc is None:
                continue
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
        # Clean up temporary directory
        if package_manager:
            package_manager.cleanup()
        print("Done.")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Logfile written to: {log_file}\n")

if __name__ == "__main__":
    main()