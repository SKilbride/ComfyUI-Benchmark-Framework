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

def load_workflow(workflow_path):
    if not os.path.exists(workflow_path):
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        # Validate KSampler parameters
        for node_id, node in workflow.items():
            if node.get("class_type") == "KSampler":
                inputs = node.get("inputs", {})
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
        return workflow
    except UnicodeDecodeError as e:
        print(f"Error: Failed to decode {workflow_path}. Ensure the file is encoded in UTF-8.")
        raise e
    except json.JSONDecodeError as e:
        print(f"Error: {workflow_path} is not a valid JSON file.")
        raise e

def extract_zip(zip_path, extract_to, extract_minimal=False):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        if extract_minimal:
            json_files = [f for f in zipf.namelist() if f.lower().endswith('.json')]
            for file_name in json_files:
                zipf.extract(file_name, extract_to)
                print(f"Extracted {file_name} to: {extract_to}")
        else:
            zipf.extractall(extract_to)
            print(f"Extracted all files to: {extract_to}")
    return extract_to

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

def capture_execution_times(proc, output_queue, capture_event, print_lock):
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
                elif progress_pattern.search(line):
                    progress_buffer.append(line)
                    print(f"\r{' ' * 120}\r{progress_buffer[-1]}", end='', flush=True)
                else:
                    if progress_buffer:
                        print(f"\n{progress_buffer[-1]}", flush=True)
                        progress_buffer = []
                    print(line, flush=True)
            if capture_event.is_set() and not error_detected:
                match = pattern.search(line)
                if match:
                    exec_time = float(match.group(1))
                    if exec_time > 1.0:  # Ignore invalid times
                        print(f"\nDEBUG: Captured execution time: {exec_time} seconds", flush=True)
                        output_queue.put(exec_time)
            if error_pattern.search(line):
                print("\nDEBUG: Detected ComfyUI processing error", flush=True)
    with print_lock:
        if progress_buffer:
            print(f"\n{progress_buffer[-1]}", flush=True)

def main():
    # Force tqdm to use ASCII characters
    # os.environ['TQDM_ASCII'] = '1'  # Temporarily commented to test impact
    # Set console encoding to UTF-8
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Run multiple ComfyUI instances for simultaneous image generations.")
    parser.add_argument("-n", "--num_instances", type=int, default=1, help="Number of simultaneous ComfyUI instances.")
    parser.add_argument("-c", "--comfy_path", required=True, help="Path to the ComfyUI directory.")
    parser.add_argument("-w", "--workflow_path", required=True, help="Path to the JSON workflow file or ZIP file.")
    parser.add_argument("-g", "--generations", type=int, default=1, help="Number of generations per instance.")
    parser.add_argument("-e", "--extract_minimal", action="store_true", help="Extract only .json files from ZIP.")
    parser.add_argument("-r", "--run_default", action="store_true", help="Use default recipe (baseconfig.json)")
    args = parser.parse_args()

    # Validate arguments
    if args.num_instances < 1:
        print("Error: --num_instances must be at least 1.")
        sys.exit(1)
    if args.generations < 1:
        print("Error: --generations must be at least 1.")
        sys.exit(1)

    comfy_path = Path(args.comfy_path).resolve()
    if not (comfy_path / "main.py").exists():
        print("Error: main.py not found in the provided ComfyUI path.")
        sys.exit(1)

    # Handle workflow path
    workflow_path = Path(args.workflow_path).resolve()
    main_workflow = None
    warmup_workflow = None
    temp_dir = None
    processes = []
    output_queues = []
    capture_events = []
    print_lock = Lock()

    # Initialize default values
    num_instances = args.num_instances
    generations = args.generations

    # Check for --run_default flag and load JSON file
    if args.run_default:
        config_filename = "baseconfig.json"  # Default config file
        config_path = None
        if workflow_path.suffix.lower() == '.zip':
            # Create temp directory with ZIP basename
            zip_basename = workflow_path.stem
            temp_dir = comfy_path / "temp" / zip_basename
            print(f"Extracting ZIP file: {workflow_path} to {temp_dir}")
            # Extract ZIP to temp directory
            extract_zip(workflow_path, temp_dir, extract_minimal=args.extract_minimal)
            config_path = temp_dir / config_filename
        elif workflow_path.suffix.lower() == '.json':
            config_path = comfy_path / config_filename

        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                num_instances = config.get("NUM_INSTANCES", args.num_instances)
                generations = config.get("GENERATIONS", args.generations)
                print(f"Loaded config from {config_path}: num_instances={num_instances}, generations={generations}")
            except json.JSONDecodeError as e:
                print(f"Error: {config_path} is not a valid JSON file.")
                raise e
            except Exception as e:
                print(f"Error reading {config_path}: {e}")
                raise
        else:
            print(f"Error: Config file {config_filename} not found in {temp_dir if workflow_path.suffix.lower() == '.zip' else comfy_path}.")
            sys.exit(1)

    try:
        if workflow_path.suffix.lower() == '.zip':
            # Temp directory is already created if run_default was processed
            if not temp_dir:
                zip_basename = workflow_path.stem
                temp_dir = comfy_path / "temp" / zip_basename
                print(f"Extracting ZIP file: {workflow_path} to {temp_dir}")
                extract_zip(workflow_path, temp_dir, extract_minimal=args.extract_minimal)

            # Run pre.py if it exists
            pre_script_path = temp_dir / "pre.py"
            run_python_script(pre_script_path, temp_dir)

            # Copy folders from extracted ComfyUI folder (only if not minimal extraction)
            if not args.extract_minimal:
                comfyui_extracted = temp_dir / "ComfyUI"
                if comfyui_extracted.exists() and comfyui_extracted.is_dir():
                    for item in comfyui_extracted.iterdir():
                        if item.is_dir():
                            target_path = comfy_path / item.name
                            shutil.copytree(item, target_path, dirs_exist_ok=True)
                            print(f"Copied folder {item.name} to: {target_path}")

                    # Check for custom_nodes in extracted ComfyUI and install requirements
                    custom_nodes_extracted = comfyui_extracted / "custom_nodes"
                    if custom_nodes_extracted.exists() and custom_nodes_extracted.is_dir():
                        for node_folder in custom_nodes_extracted.iterdir():
                            if node_folder.is_dir():
                                requirements_path = node_folder / "requirements.txt"
                                if requirements_path.exists():
                                    print(f"Installing requirements for {node_folder.name}...")
                                    try:
                                        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], cwd=node_folder)
                                        print(f"Installed requirements for {node_folder.name}")
                                    except subprocess.CalledProcessError as e:
                                        print(f"Error installing requirements for {node_folder.name}: {e}. Continuing...")

            # Run post.py if it exists
            post_script_path = temp_dir / "post.py"
            run_python_script(post_script_path, temp_dir)

            # Warn if ComfyUI folder is missing in minimal extraction
            if args.extract_minimal and not (temp_dir / "ComfyUI").exists():
                print("Warning: --extract_minimal was specified, but ComfyUI folder is missing. Ensure workflow.json is at the root of the ZIP.")

            # Load main workflow
            main_workflow_path = temp_dir / "workflow.json"
            if not main_workflow_path.exists():
                print(f"Error: workflow.json not found in extracted ZIP.")
                sys.exit(1)
            main_workflow = load_workflow(main_workflow_path)

            # Load warmup workflow if it exists
            warmup_workflow_path = temp_dir / "warmup.json"
            if warmup_workflow_path.exists():
                warmup_workflow = load_workflow(warmup_workflow_path)
            else:
                warmup_workflow = main_workflow

        elif workflow_path.suffix.lower() == '.json':
            main_workflow = load_workflow(workflow_path)
            warmup_workflow = main_workflow
        else:
            print("Error: --workflow_path must be a .json or .zip file.")
            sys.exit(1)

        processes = []
        ports = []
        base_port = 8188
        output_queues = [Queue() for _ in range(num_instances)]
        capture_events = [Event() for _ in range(num_instances)]

        print(f"Starting {num_instances} ComfyUI instances...")

        for i in range(num_instances):
            port = base_port + i
            ports.append(port)
            proc = subprocess.Popen(
                ["python", "main.py", "--port", str(port), "--listen", "127.0.0.1"],
                cwd=comfy_path,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            processes.append(proc)
            print(f"Started instance {i+1} on port {port} (PID: {proc.pid})")
            Thread(target=capture_execution_times, args=(proc, output_queues[i], capture_events[i], print_lock), daemon=True).start()

        # Wait for servers to start
        for port in ports:
            if not check_server_ready(port, timeout=60):
                raise RuntimeError(f"ComfyUI server on port {port} failed to start.")

        client_ids = [str(uuid.uuid4()) for _ in ports]

        def generation_task(idx, gen, is_warmup=False):
            port = ports[idx]
            server_address = f"127.0.0.1:{port}"
            client_id = client_ids[idx]
            workflow = warmup_workflow if is_warmup else main_workflow
            prompt = json.loads(json.dumps(workflow))
            for node_id, node in prompt.items():
                if node.get("class_type") == "KSampler" and "seed" in node.get("inputs", {}):
                    prompt[node_id]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
                    break
            try:
                prompt_id = queue_prompt(prompt, client_id, server_address)
                if is_warmup:
                    print(f"Queued warmup on instance {idx+1} (prompt_id: {prompt_id})")
                else:
                    capture_events[idx].set()
                    print(f"Queued generation {gen+1} on instance {idx+1} (prompt_id: {prompt_id})")
                history = wait_for_completion(prompt_id, server_address)
                if is_warmup:
                    print(f"Completed warmup on instance {idx+1}")
                else:
                    print(f"Completed generation {gen+1} on instance {idx+1}")
            except Exception as e:
                error_msg = f"Error during {'warmup' if is_warmup else f'generation {gen+1}'} on instance {idx+1}: {e}"
                if "ZeroDivisionError" in str(e) or "integer division or modulo by zero" in str(e):
                    error_msg += "\nThis error suggests an issue with the KSampler node in workflow.json. Please verify 'steps' (> 0), 'start_step' (>= 0), 'last_step' (> start_step and <= steps), 'cfg' (> 0), 'denoise' (0–1), 'sampler_name' (e.g., 'dpmpp_2m'), and 'scheduler' (e.g., 'normal'). Alternatively, test the workflow in the ComfyUI GUI."
                    raise RuntimeError(error_msg) from e
                raise e
            finally:
                if not is_warmup:
                    capture_events[idx].clear()

        # Warmup step
        print("Performing warmup...")
        with ThreadPoolExecutor(max_workers=num_instances) as executor:
            futures = [executor.submit(generation_task, idx, -1, is_warmup=True) for idx in range(num_instances)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception during warmup: {e}")
                    raise

        print("Warmup completed.")

        # Clear queues to ensure no stale data
        for queue in output_queues:
            while not queue.empty():
                queue.get()

        # Capture start timestamp
        start_time = time.time()

        try:
            for gen in range(generations):
                print(f"Starting generation round {gen+1}/{generations}")
                with ThreadPoolExecutor(max_workers=num_instances) as executor:
                    futures = [executor.submit(generation_task, idx, gen, is_warmup=False) for idx in range(num_instances)]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Exception in generation round {gen+1}: {e}")
                            raise
        except KeyboardInterrupt:
            print("Interrupted by user. Calculating partial metrics...")
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

        print(f"Total time to generate {total_images} images: {total_time:.2f} seconds")
        print(f"Number of images per minute: {images_per_minute:.2f}")
        print(f"Average time (secs) per image: {avg_time_per_image:.2f}")
        print(f"Total Execution Time (main generations): {total_execution_time:.2f} seconds")
        print(f"Average Execution Time Per Image (main generations): {avg_execution_time:.2f} seconds")

    except KeyboardInterrupt:
        print("Cleaning up after user interrupt...")
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
    finally:
        # Clean up: Terminate processes and remove temp directory
        print("Cleaning up...")
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
                with open(os.devnull, 'w') as devnull:
                    proc.stderr = devnull
                    proc.terminate()
                    proc.wait(timeout=2)
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Removed temporary directory: {temp_dir}")

        print("Done.")

if __name__ == "__main__":
    main()