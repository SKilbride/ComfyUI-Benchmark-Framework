import json
import random
import re
import requests
from pathlib import Path

class WorkflowManager:
    def __init__(self, workflow_path, log_file=None):
        """
        Initialize WorkflowManager with a workflow JSON path.

        Args:
            workflow_path (Path): Path to the workflow JSON file.
            log_file (Path, optional): Path to log file for logging operations.
        """
        self.workflow_path = Path(workflow_path).resolve()
        self.log_file = log_file if log_file else None
        self.workflow = None
        self.applied_overrides = []

    def log(self, message):
        """Log a message to the console and optionally to a log file."""
        print(message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

    def load_workflow(self):
        """Load and validate the workflow JSON file."""
        if self.workflow_path.suffix.lower() != '.json':
            error_msg = f"Error: {self.workflow_path} must be a .json file."
            self.log(error_msg)
            raise ValueError(error_msg)
        if not self.workflow_path.exists():
            error_msg = f"Error: Workflow file not found: {self.workflow_path}"
            self.log(error_msg)
            raise FileNotFoundError(error_msg)

        self.log(f"Loading workflow: {self.workflow_path}")
        try:
            with open(self.workflow_path, 'r', encoding='utf-8') as f:
                self.workflow = json.load(f)
            if not isinstance(self.workflow, dict):
                error_msg = f"Error: {self.workflow_path} does not contain a valid JSON object (expected a dictionary)."
                self.log(error_msg)
                raise ValueError(error_msg)
            self._validate_workflow()
        except UnicodeDecodeError as e:
            error_msg = f"Error: Failed to decode {self.workflow_path}. Ensure the file is encoded in UTF-8."
            self.log(error_msg)
            raise e
        except json.JSONDecodeError as e:
            error_msg = f"Error: {self.workflow_path} is not a valid JSON file."
            self.log(error_msg)
            raise e

    def _validate_workflow(self):
        """Validate the workflow JSON, focusing on KSampler and KSamplerAdvanced nodes."""
        for node_id, node in self.workflow.items():
            if not isinstance(node_id, str):
                error_msg = f"Error: Invalid node ID {node_id} in {self.workflow_path}. Node IDs must be strings."
                self.log(error_msg)
                raise ValueError(error_msg)
            if not isinstance(node, dict):
                error_msg = f"Error: Node {node_id} in {self.workflow_path} is not a valid node dictionary."
                self.log(error_msg)
                raise ValueError(error_msg)
            if node.get("bypass", False):
                node_title = node.get("_meta", {}).get("title", "Untitled")
                self.log(f"Node {node_id} ({node_title}) is bypassed")
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
                    self.log(f"DEBUG: KSampler node {node_id} - steps: {steps}, start_step: {start_step}, last_step: {last_step}, cfg: {cfg}, denoise: {denoise}, sampler_name: {sampler_name}, scheduler: {scheduler}")
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
                    self.log(f"DEBUG: KSamplerAdvanced node {node_id} - steps: {steps}, cfg: {cfg}, sampler_name: {sampler_name}, scheduler: {scheduler}, start_at_step: {start_at_step}, end_at_step: {end_at_step}")
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
                    if not isinstance(end_at_step, int) or end_at_step <= start_step:
                        raise ValueError(f"Invalid KSamplerAdvanced end_at_step ({end_at_step}) in node {node_id}. Must be an integer > start_at_step.")

    def apply_overrides(self, override_path):
        """Apply overrides from a JSON file to the workflow."""
        if not override_path or not Path(override_path).exists():
            self.log("No override file provided or file does not exist.")
            return

        try:
            with open(override_path, 'r', encoding='utf-8') as f:
                overrides = json.load(f).get("overrides", {})
            self.log(f"Override file specified: {override_path}")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            error_msg = f"Error: Failed to parse override file {override_path}: {e}"
            self.log(error_msg)
            raise

        modified_workflow = json.loads(json.dumps(self.workflow))  # Deep copy
        self.applied_overrides = []

        for override_key, override in overrides.items():
            override_item = override.get("override_item")
            override_value = override.get("override_value")
            restrict = override.get("restrict", {})
            if not override_item or override_value is None:
                self.log(f"Warning: Skipping invalid override {override_key}: missing override_item or override_value")
                continue
            matched_nodes = []
            for node_id, node in modified_workflow.items():
                matches = True
                if restrict:
                    for restrict_key, restrict_value in restrict.items():
                        if restrict_key == "id":
                            if node_id != str(restrict_value):
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
                            modified_workflow[node_id]["bypass"] = True
                            self.log(f"Applying override {override_key}: setting bypass to true in node {node_id} ({node_title})")
                            matched_nodes.append(f"{node_id} ({node_title})")
                        elif override_value is False and node.get("bypass", False) is True:
                            modified_workflow[node_id]["bypass"] = False
                            self.log(f"Applying override {override_key}: setting bypass to false in node {node_id} ({node_title})")
                            matched_nodes.append(f"{node_id} ({node_title})")
                    elif override_item in node.get("inputs", {}):
                        modified_workflow[node_id]["inputs"][override_item] = override_value
                        self.log(f"Applying override {override_key}: setting {override_item} to {override_value} in node {node_id} ({node_title})")
                        matched_nodes.append(f"{node_id} ({node_title})")
            if matched_nodes:
                self.applied_overrides.append({
                    "key": override_key,
                    "item": override_item,
                    "value": override_value,
                    "nodes": matched_nodes
                })
            else:
                self.log(f"Warning: Override {override_key} matched no nodes")
        self.workflow = modified_workflow

    def get_workflow(self, randomize_seeds=True):
        """
        Get a copy of the workflow, optionally randomizing seeds for KSampler/PrimitiveInt nodes.

        Args:
            randomize_seeds (bool): If True, randomize seeds for KSampler and PrimitiveInt nodes.

        Returns:
            dict: A deep copy of the workflow JSON.
        """
        if self.workflow is None:
            error_msg = "Error: Workflow not loaded. Call load_workflow() first."
            self.log(error_msg)
            raise ValueError(error_msg)

        workflow = json.loads(json.dumps(self.workflow))  # Deep copy
        if randomize_seeds:
            for node_id, node in workflow.items():
                class_type = node.get("class_type")
                inputs = node.get("inputs", {})
                if class_type == "KSampler" and "seed" in inputs:
                    workflow[node_id]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
                elif class_type == "KSamplerAdvanced" and "noise_seed" in inputs:
                    workflow[node_id]["inputs"]["noise_seed"] = random.randint(0, 2**32 - 1)
                elif class_type == "PrimitiveInt" and "value" in inputs:
                    meta = node.get("_meta", {})
                    title = meta.get("title", "")
                    if re.search(r"Random", title, re.IGNORECASE):
                        workflow[node_id]["inputs"]["value"] = random.randint(0, 2**32 - 1)
        return workflow

    def queue_prompt(self, client_id, server_address):
        """
        Queue a workflow prompt to a ComfyUI server.

        Args:
            client_id (str): Unique client ID for the prompt.
            server_address (str): Address of the ComfyUI server (e.g., "127.0.0.1:8188").

        Returns:
            str: The prompt ID returned by the server.
        """
        workflow = self.get_workflow()
        data = {"prompt": workflow, "client_id": client_id}
        response = requests.post(f"http://{server_address}/prompt", json=data)
        if response.status_code != 200:
            error_msg = f"Failed to queue prompt: {response.text}"
            self.log(error_msg)
            raise Exception(error_msg)
        return response.json()["prompt_id"]

    def get_applied_overrides(self):
        """Return the list of applied overrides."""
        return self.applied_overrides

    def set_benchmarknode_value(self, benchmark_node_manager, field_name, field_value):
        """
        Set a value on the Benchmark Workflow node if the comfyui-benchmark custom node exists.
        If the node doesn't exist in the workflow JSON, create it.

        Args:
            benchmark_node_manager (BenchmarkNodeManager): Instance to check for comfyui-benchmark existence.
            field_name (str): The field to set. Can be 'capture_benchmark', 'outfile_postfix1', or 'outfile_postfix2'.
            field_value: The value to set. Boolean for 'capture_benchmark', string for 'outfile_postfix1' and 'outfile_postfix2'.
        """
        if not benchmark_node_manager.exists:
            self.log("comfyui-benchmark custom node not found. Skipping Benchmark Workflow node modification.")
            return

        if self.workflow is None:
            error_msg = "Error: Workflow not loaded. Call load_workflow() first."
            self.log(error_msg)
            raise ValueError(error_msg)

        if field_name not in ['capture_benchmark', 'outfile_postfix1', 'outfile_postfix2']:
            error_msg = f"Invalid field_name: {field_name}. Must be one of 'capture_benchmark', 'outfile_postfix1', 'outfile_postfix2'."
            self.log(error_msg)
            raise ValueError(error_msg)

        if field_name == 'capture_benchmark' and not isinstance(field_value, bool):
            error_msg = f"Invalid field_value for {field_name}: must be boolean."
            self.log(error_msg)
            raise ValueError(error_msg)

        if field_name in ['outfile_postfix1', 'outfile_postfix2'] and not isinstance(field_value, str):
            error_msg = f"Invalid field_value for {field_name}: must be string."
            self.log(error_msg)
            raise ValueError(error_msg)

        benchmark_node_id = None
        for node_id, node in self.workflow.items():
            meta = node.get("_meta", {})
            title = meta.get("title", "")
            if title == "Benchmark Workflow":
                benchmark_node_id = node_id
                break

        modified_workflow = json.loads(json.dumps(self.workflow))  # Deep copy

        if benchmark_node_id is None:
            # Create a new Benchmark Workflow node
            benchmark_node_id = str(len(modified_workflow) + 1)  # Simple ID generation
            modified_workflow[benchmark_node_id] = {
                "class_type": "BenchmarkWorkflow",
                "_meta": {"title": "Benchmark Workflow"},
                "inputs": {
                    "capture_benchmark": True,
                    "outfile_postfix1": "",
                    "outfile_postfix2": ""
                }
            }
            self.log(f"Created new Benchmark Workflow node with ID: {benchmark_node_id}")

        node = modified_workflow[benchmark_node_id]
        if field_name not in node.get("inputs", {}):
            error_msg = f"Field '{field_name}' not found in inputs of Benchmark Workflow node (ID: {benchmark_node_id})."
            self.log(error_msg)
            raise ValueError(error_msg)

        node["inputs"][field_name] = field_value
        self.log(f"Set {field_name} to {field_value} in Benchmark Workflow node (ID: {benchmark_node_id})")
        self.workflow = modified_workflow