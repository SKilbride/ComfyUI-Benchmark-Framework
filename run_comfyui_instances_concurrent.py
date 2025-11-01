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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
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

def wait_for_completion(prompt_id, server_address, timeout=600, instance_id=None, debug=False):
    """Wait for prompt with timeout, health checks, and conditional debug logging."""
    start_time = time.time()
    last_health_check = 0
    health_check_interval = 15

    if debug:
        print(f"[DEBUG] [{instance_id}] Waiting for prompt {prompt_id} on {server_address} (timeout: {timeout}s)")

    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time

        # Health check (only in debug mode)
        if debug and elapsed - last_health_check > health_check_interval:
            try:
                health = requests.get(f"http://{server_address}/system_stats", timeout=5)
                if health.status_code == 200:
                    stats = health.json()
                    vram_total = stats.get("vram_total", "N/A")
                    vram_free = stats.get("vram_free", "N/A")
                    print(f"[HEALTH] [{instance_id}] VRAM: {vram_total} MB, Free: {vram_free} MB")
                last_health_check = time.time()
            except Exception as e:
                print(f"[HEALTH] [{instance_id}] Failed: {e}")

        # Poll history
        try:
            response = requests.get(f"http://{server_address}/history/{prompt_id}", timeout=8)
            if response.status_code == 200:
                history = response.json()
                if prompt_id in history:
                    exec_time = time.time() - start_time
                    print(f"Prompt {prompt_id} completed in {exec_time:.2f} seconds")
                    return history
                elif debug:
                    print(f"[DEBUG] [{instance_id}] Prompt in history but not complete")
            elif response.status_code == 404 and debug:
                print(f"[DEBUG] [{instance_id}] History not found yet")
            elif debug:
                print(f"[DEBUG] [{instance_id}] History status: {response.status_code}")
        except requests.Timeout:
            if debug:
                print(f"[TIMEOUT] [{instance_id}] Request timeout for /history/{prompt_id}")
        except requests.ConnectionError:
            if debug:
                print(f"[ERROR] [{instance_id}] Connection lost to {server_address}")
        except Exception as e:
            if debug:
                print(f"[ERROR] [{instance_id}] Wait error: {e}")

        time.sleep(2)

    raise TimeoutError(f"[{instance_id}] Prompt {prompt_id} timed out after {timeout}s on {server_address}")

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
        print(f"Warning: baseconfig.json not found. Using defaults.")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Warning: baseconfig.json not found. Using defaults.\n")
        return {"NUM_INSTANCES": 1, "GENERATIONS": 1}
    try:
        with open(selected_path, 'r', encoding='utf-8') as f:
            baseconfig = json.load(f)
        if not isinstance(baseconfig, dict):
            raise ValueError("Invalid baseconfig.json")
        config_values = {
            "NUM_INSTANCES": baseconfig.get("NUM_INSTANCES", 1),
            "GENERATIONS": baseconfig.get("GENERATIONS", 1)
        }
        print(f"Loaded baseconfig: NUM_INSTANCES={config_values['NUM_INSTANCES']}, GENERATIONS={config_values['GENERATIONS']}")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Loaded baseconfig from {selected_path}\n")
        return config_values
    except Exception as e:
        print(f"Failed to load baseconfig: {e}. Using defaults.")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Failed to load baseconfig: {e}\n")
        return {"NUM_INSTANCES": 1, "GENERATIONS": 1}

def interrupt_process(port):
    server_address = f"127.0.0.1:{port}"
    try:
        response = requests.post(f"http://{server_address}/interrupt", timeout=5)
        if response.status_code == 200:
            print(f"Sent interrupt to {server_address}")
            return True
        else:
            print(f"Interrupt failed: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"Interrupt error: {e}")
        return False

def check_server_ready(port, timeout=60, interval=2):
    server_address = f"127.0.0.1:{port}"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{server_address}/", timeout=5)
            if response.status_code == 200:
                print(f"Server on port {port} is ready")
                return True
        except requests.ConnectionError:
            time.sleep(interval)
    print(f"Error: Server on port {port} not ready after {timeout}s")
    return False

def check_server_running(port):
    server_address = f"127.0.0.1:{port}"
    try:
        response = requests.get(f"http://{server_address}/", timeout=3)
        if response.status_code == 200:
            print(f"Server already running on port {port}")
            return True
    except requests.ConnectionError:
        return False
    return False

def capture_execution_times(proc, output_queue, capture_event, print_lock, log_file=None):
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
            except:
                line = line.strip()
            line = line.replace('█', '#').replace('▌', '#').replace('▎', '#')
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
                    if exec_time > 1.0:
                        print("\n")
                        if log_file:
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write("\n")
                        output_queue.put(exec_time)
    with print_lock:
        if progress_buffer:
            print(f"\n{progress_buffer[-1]}", flush=True)

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Run multiple ComfyUI instances concurrently.")
    parser.add_argument("-n", "--num_instances", type=int, default=1)
    parser.add_argument("-c", "--comfy_path", required=True)
    parser.add_argument("-w", "--workflow_path", required=True)
    parser.add_argument("-g", "--generations", type=int, default=1)
    parser.add_argument("-e", "--extract_minimal", action="store_true")
    parser.add_argument("-r", "--run_default", action="store_true")
    parser.add_argument("-o", "--override", type=str)
    parser.add_argument("-l", "--log", nargs='?', const=True, default=False)
    parser.add_argument("-t", "--temp_path", type=str)
    parser.add_argument("-p", "--port", type=int, default=8188)
    parser.add_argument("-u", "--use_main_workflow_only", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--debug-warmup", action="store_true", help="Enable verbose debug logging ONLY during warmup")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.num_instances < 1 or args.generations < 1 or not (1024 <= args.port <= 65535):
        print("Invalid arguments.")
        sys.exit(1)

    log_file = None
    if args.log is not False:
        workflow_basename = Path(args.workflow_path).stem
        timestamp = datetime.now().strftime("%y%m%d_") + str(int(time.time()))
        log_file = Path(args.log).resolve() if args.log is not True else Path(f"{workflow_basename}_{timestamp}.txt").resolve()
        if log_file.is_dir():
            log_file = log_file / f"{workflow_basename}_{timestamp}.txt"
        print(f"Logging to: {log_file}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Run started: {datetime.now()}\n")

    workflow_path = Path(args.workflow_path).resolve()
    warmup_workflow_path = None
    comfy_path = Path(args.comfy_path).resolve()
    package_manager = None
    extra_args = args.extra_args or []

    try:
        # Handle workflow
        if workflow_path.is_dir():
            workflow_path = workflow_path / "workflow.json"
            warmup_workflow_path = workflow_path.parent / "warmup.json" if not args.use_main_workflow_only else workflow_path
            if not warmup_workflow_path.exists():
                warmup_workflow_path = workflow_path
            if not workflow_path.exists():
                raise FileNotFoundError("workflow.json not found")
        elif workflow_path.suffix.lower() == '.zip':
            package_manager = PackageManager(
                zip_path=workflow_path, comfy_path=comfy_path,
                temp_path=args.temp_path, extract_minimal=args.extract_minimal, log_file=log_file
            )
            workflow_path = package_manager.extract_zip()
            warmup_workflow_path = workflow_path.parent / "warmup.json" if not args.use_main_workflow_only else workflow_path
            if not warmup_workflow_path.exists():
                warmup_workflow_path = workflow_path
        elif workflow_path.suffix.lower() != '.json':
            raise ValueError("Invalid workflow path")

        workflow_manager = WorkflowManager(workflow_path=workflow_path, log_file=log_file)
        workflow_manager.load_workflow()
        warmup_workflow_manager = WorkflowManager(workflow_path=warmup_workflow_path, log_file=log_file)
        warmup_workflow_manager.load_workflow()

        if args.override:
            workflow_manager.apply_overrides(args.override)
            warmup_workflow_manager.apply_overrides(args.override)

        benchmark_node_manager = BenchmarkNodeManager(custom_nodes_path=comfy_path / "custom_nodes")

        num_instances = args.num_instances
        generations = args.generations
        if args.run_default:
            bc = load_baseconfig(comfy_path, package_manager.temp_dir if package_manager else None, log_file)
            if args.num_instances == 1: num_instances = bc["NUM_INSTANCES"]
            if args.generations == 1: generations = bc["GENERATIONS"]

        processes, ports = [], []
        base_port = args.port
        output_queues = [Queue() for _ in range(num_instances)]
        capture_events = [Event() for _ in range(num_instances)]
        print_lock = Lock()

        print(f"Starting {num_instances} instances...")
        for i in range(num_instances):
            port = base_port + i
            ports.append(port)
            if check_server_running(port):
                processes.append(None)
                continue
            cmd = ["python", "main.py", "--port", str(port), "--listen", "127.0.0.1"] + extra_args
            proc = subprocess.Popen(
                cmd, cwd=comfy_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
                text=True, bufsize=1
            )
            processes.append(proc)
            print(f"Started instance {i+1} on port {port} (PID: {proc.pid})")
            Thread(target=capture_execution_times, args=(proc, output_queues[i], capture_events[i], print_lock, log_file), daemon=True).start()

        for i, port in enumerate(ports):
            if processes[i] is None: continue
            if not check_server_ready(port): raise RuntimeError(f"Server {port} failed")

        client_ids = [str(uuid.uuid4()) for _ in ports]

        def generation_task(idx, gen, is_warmup=False):
            instance_id = f"Inst{idx+1}"
            port = ports[idx]
            server_address = f"127.0.0.1:{port}"
            client_id = client_ids[idx]
            debug = args.debug_warmup and is_warmup  # ← ONLY during warmup if flag set

            try:
                print(f"[{instance_id}] Starting {'warmup' if is_warmup else f'gen {gen+1}'}")
                manager = warmup_workflow_manager if is_warmup else workflow_manager
                workflow = manager.get_workflow(randomize_seeds=True)

                if benchmark_node_manager.exists:
                    bid = next((nid for nid, n in workflow.items() if n.get("_meta", {}).get("title") == "Benchmark Workflow"), None)
                    if not bid:
                        bid = str(len(workflow) + 1)
                        workflow[bid] = {"class_type": "BenchmarkWorkflow", "_meta": {"title": "Benchmark Workflow"}, "inputs": {"capture_benchmark": True, "file_prefix": "", "file_postfix": ""}}
                    node = workflow[bid]
                    node["inputs"]["file_postfix"] = "_warmup_" if is_warmup else f"_RUN_{gen+1}.{idx+1}"
                    

                print(f"[{instance_id}] Queueing...")
                resp = requests.post(f"http://{server_address}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
                if resp.status_code != 200:
                    raise Exception(f"Queue failed: {resp.text}")
                prompt_id = resp.json()["prompt_id"]
                print(f"[{instance_id}] Queued: {prompt_id}")

                if not is_warmup:
                    capture_events[idx].set()

                history = wait_for_completion(prompt_id, server_address, timeout=900 if not is_warmup else 600, instance_id=instance_id, debug=debug)

                print(f"[{instance_id}] {'Warmup' if is_warmup else f'Gen {gen+1}'} completed")
            except Exception as e:
                print(f"[{instance_id}] ERROR: {e}")
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{instance_id}] ERROR: {e}\n")
                raise
            finally:
                if not is_warmup:
                    capture_events[idx].clear()

        # === SAFE SEQUENTIAL WARMUP ===
        print("Performing SAFE sequential warmup to prevent VRAM OOM...")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("SAFE sequential warmup started\n")

        for idx in range(num_instances):
            port = ports[idx]
            instance_id = f"Inst{idx+1}"
            print(f"[WARMUP] Starting {instance_id} on port {port}")
            try:
                generation_task(idx, -1, is_warmup=True)
                print(f"[WARMUP] {instance_id} completed")
            except Exception as e:
                error_msg = f"[WARMUP] {instance_id} FAILED: {e}"
                print(error_msg)
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(error_msg + "\n")
                if "memory" in str(e).lower() or "cuda" in str(e).lower():
                    raise RuntimeError(f"CUDA OOM on {instance_id}. Reduce instances or use --lowvram.")
                raise
            finally:
                print(f"[WARMUP] Freeing VRAM on {instance_id}")
                interrupt_process(port)
                time.sleep(3)

        print("All warmups completed safely.")

        # Clear queues
        for q in output_queues:
            while not q.empty(): q.get()

        # MAIN RUN
        start_time = time.time()
        try:
            for gen in range(generations):
                print(f"Generation round {gen+1}/{generations}")
                with ThreadPoolExecutor(max_workers=num_instances) as executor:
                    futures = [executor.submit(generation_task, idx, gen, False) for idx in range(num_instances)]
                    for f in as_completed(futures):
                        f.result()
        except KeyboardInterrupt:
            print("Interrupted. Partial metrics...")
            raise

        # METRICS
        total_time = time.time() - start_time
        total_images = num_instances * generations
        exec_times = [q.get() for q in output_queues if not q.empty() for _ in range(q.qsize())]
        total_exec = sum(exec_times)
        avg_exec = total_exec / len(exec_times) if exec_times else 0
        ipm = total_images / (total_time / 60) if total_time > 0 else 0
        atpi = total_time / total_images if total_images > 0 else 0

        print('####_RESULTS_SUMMARY_####')
        print(f"Total time: {total_time:.2f}s | Images: {total_images} | IPM (Images Per Minute): {ipm:.2f} | Avg/sec Per Image: {atpi:.2f}")
        print(f"Total exec: {total_exec:.2f}s | Avg exec: {avg_exec:.2f}s")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write('####_RESULTS_SUMMARY_####\n')
                f.write(f"Total: {total_time:.2f}s, IPM: {ipm:.2f}, Avg: {atpi:.2f}s\n")

    except KeyboardInterrupt:
        print("User interrupt.")
    finally:
        if args.no_cleanup:
            print("Skipping cleanup (--no-cleanup)")
        else:
            print("Cleaning up...")
            for i, proc in enumerate(processes):
                if not proc: continue
                port = base_port + i
                server = f"127.0.0.1:{port}"
                try:
                    q = requests.get(f"http://{server}/queue", timeout=3)
                    if q.status_code == 200 and (q.json().get("queue_running") or q.json().get("queue_pending")):
                        interrupt_process(port)
                        time.sleep(3)
                except: pass
                if proc.poll() is None:
                    proc.terminate()
                    try: proc.wait(10)
                    except: proc.kill()
            if package_manager: package_manager.cleanup()
            print("Cleanup done.")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Log: {log_file}\n")
        print("Done.")

if __name__ == "__main__":
    main()