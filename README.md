# ComfyUI Benchmarking Framework

## Overview
This framework automates benchmarking of ComfyUI workflows by running prebuilt benchmarking packages. These packages are ZIP archives containing workflow JSON files, models, and assets needed for image or video generation tasks. The script launches ComfyUI instances, performs a warmup run (to load models into memory), and then executes the specified number of generations while measuring performance metrics.

Key features:
- Supports concurrent ComfyUI instances for multi-GPU or advanced benchmarking.
- Handles ZIP package extraction, custom node installation, and script execution (pre.py/post.py).
- Applies overrides to workflows via JSON files.
- Integrates with the `comfyui-benchmark` custom node for detailed statistics (if installed).
- Generates metrics like images per minute, average time per image, and execution times.
- Logs output and handles cleanup, including process termination and temporary directories.

## Installation/Setup
1. Install ComfyUI (portable or standard installation).
2. Place the script (`run_comfyui_instances_concurrent.py`), `workflow_manager.py`, and `package_manager.py` in the same directory as your ComfyUI executable (e.g., next to `run_nvidia_gpu.bat`).
3. Ensure Python dependencies are available (e.g., via `python_embedded` in portable installs): `argparse`, `subprocess`, `requests`, `json`, `uuid`, `os`, `sys`, `yaml`, `re`, `concurrent.futures`, `pathlib`, `threading`, `queue`, `datetime`.
4. Optionally install the `comfyui-benchmark` custom node for enhanced benchmarking output (JSON statistics files).

## Running the Benchmark
The benchmarking framework works by running prebuilt benchmarking packages. These packages contain all the necessary primary and secondary model files required to run a dedicated benchmark workflow. Workflow files are ComfyUI JSON workflow files configured for running using the ComfyUI API mode.

**Warmup:** Each benchmarking run performs an initial warmup run to load models into memory and initialize the pipeline. The warmup time is not included in any benchmarking metrics. By default, it uses `warmup.json` if available; use `-u` to force `workflow.json` for warmup.

Usage:
```
python run_comfyui_instances_concurrent.py [options]
```

Sample command (assumes a ComfyUI portable installation with the script in the same folder as `run_nvidia_gpu.bat`):
```
python_embedded\python run_comfyui_instances_concurrent.py -c .\ComfyUI -w c:\workflows\wan_2.2_i2v_14b_640x640x81x20s.zip -r -l
```

Another sample with overrides and multiple instances:
```
python_embedded\python run_comfyui_instances_concurrent.py -c .\ComfyUI -w c:\workflows\flux_workflow.zip -n 2 -g 5 -o overrides.json -l logs/ -p 8190 --extra_args --cpu
```

## Command Line Arguments
The ComfyUI path (`-c`) and workflow path (`-w`) are required. All other arguments are optional; defaults are used if omitted. Below is a detailed explanation of key arguments, including their purpose, usage, and examples.

- **`-h, --help`**
  - **Description**: Displays the help message with all available arguments and their descriptions, then exits.
  - **Usage**: Use to quickly check available options without running the script.
  - **Example**: `python run_comfyui_instances_concurrent.py --help`

- **`-c, --comfy_path`** (Required)
  - **Description**: Specifies the path to the ComfyUI installation directory. This must point to the folder containing the ComfyUI executable (`main.py`) and subdirectories like `custom_nodes`, `models`, and `user`.
  - **Usage**: Provide the absolute or relative path to the ComfyUI directory.
  - **Example**: `-c ./ComfyUI` or `-c /path/to/ComfyUI`

- **`-w, --workflow_path`** (Required)
  - **Description**: Specifies the path to the workflow file or package. Can be a `.json` workflow file, a `.zip` benchmark package, or a directory containing `workflow.json` (and optionally `warmup.json`).
  - **Usage**: Ensure the path points to a valid file or directory. For ZIP files, the package must contain `workflow.json` at the root.
  - **Example**: `-w workflow.json`, `-w workflows/package.zip`, `-w workflows/`

- **`-g, --generations`** (Default: 1)
  - **Description**: Specifies the number of generations (image or video outputs) per ComfyUI instance. For image workflows, this is typically >1; for video workflows, 1 is common due to longer processing times.
  - **Usage**: Provide an integer value. Can be overridden by `baseconfig.json` with `-r`.
  - **Example**: `-g 5` (runs 5 generations per instance)

- **`-e, --extract_minimal`**
  - **Description**: When set, extracts only JSON files (`workflow.json`, `warmup.json`, `baseconfig.json`) from ZIP packages, assuming models and assets are already installed. Useful for repeated runs to save disk space and time.
  - **Usage**: Use on subsequent runs after the first full extraction. Requires prior installation of models/assets.
  - **Example**: `-e` (extracts only JSON files from the ZIP)

- **`-r, --run_default`**
  - **Description**: Uses default `num_instances` and `generations` values from `baseconfig.json` (if present) instead of command-line values. Useful for running benchmarks with predefined settings.
  - **Usage**: Ensure `baseconfig.json` exists in the ComfyUI directory or extracted ZIP.
  - **Example**: `-r` (uses values from `baseconfig.json`)

- **`-o, --override`**
  - **Description**: Specifies a JSON file containing override parameters to modify workflow node inputs or bypass nodes. Overrides allow dynamic adjustment of workflow settings (e.g., changing `steps` in a KSampler node) without editing the original JSON file. The override file can target specific nodes or multiple nodes of the same type, and supports bypassing nodes to disable their execution.
  - **Purpose**: Enables flexible customization of workflows for testing different configurations (e.g., varying sampling steps, CFG scale, or enabling/disabling nodes) without modifying the source workflow.
  - **Override File Structure**: The file must be a JSON object with an `"overrides"` key mapping to a dictionary of override entries. Each entry has:
    - `override_item`: The node input field to modify (e.g., `steps`, `cfg`, `bypass`).
    - `override_value`: The new value for the field.
    - `restrict` (optional): A dictionary to filter nodes by specific attributes (e.g., `id`, `_meta.title`, or other node properties).
  - **Node Specification Logic**:
    - Without `restrict`, the override applies to all nodes with the specified `override_item` in their `inputs`.
    - With `restrict`, the override applies only to nodes matching all specified criteria (e.g., `id: "3"` or `_meta.title: "KSampler1"`).
    - To target a single node, use `restrict` with unique identifiers like `id` or `_meta.title`.
    - To target multiple nodes, omit `restrict` or use `restrict` with a common attribute (e.g., `class_type: "KSampler"`).
  - **Bypassing Nodes**: Set `override_item: "bypass"` and `override_value: true` to disable a node, or `override_value: false` to re-enable a bypassed node. Bypassing skips a node’s execution in the workflow.
  - **Examples**:
    - **Change Steps for All KSampler Nodes**:
      ```json
      {
        "overrides": {
          "ksampler_steps": {
            "override_item": "steps",
            "override_value": 30
          }
        }
      }
      ```
      - Applies `steps: 30` to all KSampler nodes’ `inputs.steps`.
    - **Change Steps for a Specific Node by ID**:
      ```json
      {
        "overrides": {
          "ksampler_node_3_steps": {
            "override_item": "steps",
            "override_value": 25,
            "restrict": {"id": "3"}
          }
        }
      }
      ```
      - Sets `steps: 25` only for the node with ID `"3"`.
    - **Change Steps for a Specific Node by Title**:
      ```json
      {
        "overrides": {
          "ksampler1_steps": {
            "override_item": "steps",
            "override_value": 40,
            "restrict": {"_meta.title": "KSampler1"}
          }
        }
      }
      ```
      - Sets `steps: 40` only for the node with `_meta.title: "KSampler1"`.
    - **Bypass a Specific Node**:
      ```json
      {
        "overrides": {
          "bypass_ksampler": {
            "override_item": "bypass",
            "override_value": true,
            "restrict": {"_meta.title": "KSampler1"}
          }
        }
      }
      ```
      - Disables the node titled `"KSampler1"`.
    - **Change Multiple Nodes of the Same Type**:
      ```json
      {
        "overrides": {
          "ksampler_cfg": {
            "override_item": "cfg",
            "override_value": 7.5,
            "restrict": {"class_type": "KSampler"}
          }
        }
      }
      ```
      - Sets `cfg: 7.5` for all KSampler nodes.
    - **Override Multiple Different Nodes (Complex Example)**:
      ```json
      {
        "overrides": {
          "ksampler_steps_all": {
            "override_item": "steps",
            "override_value": 10,
            "restrict": {"class_type": "KSamplerAdvanced"}
          },
          "ksampler1_end_at_step": {
            "override_item": "end_at_step",
            "override_value": 5,
            "restrict": {"id": "86"}
          },
          "ksampler2_start_at_step": {
            "override_item": "start_at_step",
            "override_value": 5,
            "restrict": {"id": "85"}
          },
          "ksampler2_end_at_step": {
            "override_item": "end_at_step",
            "override_value": 10,
            "restrict": {"id": "85"}
          },
          "bypass_lora_all": {
            "override_item": "bypass",
            "override_value": true,
            "restrict": {"class_type": "LoraLoaderModelOnly"}
          },
          "bypass_sampling_all": {
            "override_item": "bypass",
            "override_value": true,
            "restrict": {"class_type": "ModelSamplingSD3"}
          }
        }
      }
      ```
      - Applies steps: 10 to all KSamplerAdvanced nodes.
      - Sets end_at_step: 5 for the first KSamplerAdvanced (ID "86").
      - Sets start_at_step: 5 and end_at_step: 10 for the second KSamplerAdvanced (ID "85").
      - Bypasses all LoraLoaderModelOnly nodes (disables LoRA loading).
      - Bypasses all ModelSamplingSD3 nodes (disables SD3 sampling adjustments).
  - **Usage**: Provide the path to the override JSON file. The script logs applied overrides in the output and log file.
  - **Example**: `-o overrides.json` (applies overrides from `overrides.json`)

- **`-l, --log`**
  - **Description**: Enables logging of console output to a file. If no path is provided, the log file is named `<workflow_basename>_<yymmdd_epochtime>.txt` (e.g., `flux.1_krea_fp8_1024x1024x20_251024_1755227444.txt`). If a path is provided, it uses the file directly (if a file) or appends the timestamped name to the directory.
  - **Usage**: Use `-l` alone for default naming or specify a file/directory path.
  - **Example**: `-l` or `-l logs/` or `-l output.log`

- **`-n, --num_instances`** (Default: 1)
  - **Description**: Specifies the number of concurrent ComfyUI instances to run. Each instance runs on a separate port, starting from the base port (`-p`). Useful for multi-GPU setups or stress testing.
  - **Usage**: Set to >1 for concurrent execution. Ensure sufficient system resources (e.g., GPU memory).
  - **Example**: `-n 2` (runs two instances on ports 8188 and 8189 by default)

- **`-t, --temp_path`**
  - **Description**: Specifies an alternate parent directory for temporary files extracted from ZIP packages. By default, temporary files are stored in `<comfy_path>/temp`. This is useful when disk space is limited on the drive where ComfyUI is installed or to organize temporary files elsewhere.
  - **Usage**: Provide a directory path where the script can create a temporary folder (named `temp_<uuid>`). The directory must be writable. Temporary files are cleaned up automatically after the run unless interrupted abnormally.
  - **Example**: `-t /mnt/external/temp` (extracts ZIP contents to `/mnt/external/temp/temp_<uuid>`)
  - **Note**: Ensure the specified path has sufficient space for extracted models and assets, especially for large packages without `-e`.

- **`-p, --port`** (Default: 8188)
  - **Description**: Specifies the starting base port for ComfyUI instances. Each instance uses a sequential port (e.g., `-n 3` with `-p 8190` uses ports 8190, 8191, 8192). If a port is already in use by a running ComfyUI instance (e.g., a desktop ComfyUI session), the script reuses it instead of launching a new process.
  - **Purpose**: Allows benchmarking on an actively running ComfyUI instance, which is particularly useful when testing with the ComfyUI desktop version (e.g., running via GUI). This avoids launching redundant processes and leverages existing server instances.
  - **Usage**: Set to match the port of a running ComfyUI instance (check the ComfyUI GUI or server logs for the port). Ensure ports are free or correctly identify running instances to avoid conflicts.
  - **Example**: `-p 8190` (starts instances at port 8190; reuses if already running)
  - **Note**: If using a running desktop ComfyUI instance, ensure it’s in API mode (listening on `127.0.0.1:<port>`). Use `-n 1` to target a single running instance.

- **`-u, --use_main_workflow_only`**
  - **Description**: Forces the warmup phase to use `workflow.json` even if `warmup.json` exists in the package or directory. By default, the script uses `warmup.json` for warmup if available.
  - **Usage**: Use when you want consistent behavior between warmup and main generations or if `warmup.json` is unsuitable.
  - **Example**: `-u` (uses `workflow.json` for both warmup and main generations)

- ** `--extra_args`**
  - **Description**: Passes additional command-line arguments to ComfyUI’s `main.py` when launching new instances. Useful for customizing ComfyUI behavior (e.g., forcing CPU execution, specifying GPUs, or enabling debug options).
  - **Usage**: Provide arguments as a space-separated list after `--extra_args`. This must be the last argument in the command, as all subsequent tokens are treated as part of `extra_args`. Arguments are passed directly to `main.py` for each instance launched (ignored for reused instances).
  - **Requirements**: Arguments must be valid for ComfyUI’s `main.py`. Common options include `--cpu`, `--num_gpus`, `--force-fp16`, or `--disable-xformers`. Check ComfyUI documentation for supported options.
  - **Examples**:
    - `--extra_args --cpu` (forces CPU execution for all instances)
    - `--extra_args --num_gpus 2 --force-fp16` (uses 2 GPUs with FP16 precision)
    - `--extra_args --disable-xformers` (disables xformers optimizations)
  - **Recommendation**: Always place `--extra_args` last to avoid parsing issues, as it consumes all remaining arguments.
  - **Note**: Does not affect existing ComfyUI instances reused via `-p`.

## Benchmark Package Format
The benchmark package is a zipped archive containing all necessary model files, assets, and workflows. Files listed below must be at the root level (no extra folders in the ZIP).

<style>
    /* Style all table cells, including headers, with minimal padding */
    .minimal-padding-table th, 
    .minimal-padding-table td {
        border: 1px solid #ddd;
        padding: 4px 8px; /* 4px for top/bottom, 8px for left/right */
        text-align: left;
    }
    
    /* Remove default margins from heading tags within table headers */
    .minimal-padding-table th h3 {
        margin: 0;
    }

    /* Style the header rows with the background color */
    .minimal-padding-table .header-row {
        background-color: #c7ba5b;
    }
</style>

<table class="minimal-padding-table" style="border-collapse: collapse; width: 100%; margin: 20px 0;">
  <thead>
    <tr class="header-row">
      <th><h3>File</h3></th>
      <th><h3>Description</h3></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>workflow.json</strong></td>
      <td><strong>Required</strong>. The main ComfyUI workflow JSON file (exported via API format).</td>
    </tr>
    <tr>
      <td><strong>warmup.json</strong></td>
      <td><strong>Optional</strong>. A simplified workflow for warmup (e.g., single frame for video tasks). Uses <code>workflow.json</code> if not provided. Ignored if <code>-u</code> is set.</td>
    </tr>
    <tr>
      <td><strong>baseconfig.json</strong></td>
      <td><strong>Optional</strong>. JSON with default <code>NUM_INSTANCES</code> and <code>GENERATIONS</code> for <code>-r</code>. Example: <code>{"NUM_INSTANCES": 1, "GENERATIONS": 1}</code>.</td>
    </tr>
    <tr>
      <td><strong>pre.py</strong></td>
      <td><strong>Optional</strong>. Python script run before copying models/assets (e.g., create directories). Skipped with <code>-e</code>.</td>
    </tr>
    <tr>
      <td><strong>post.py</strong></td>
      <td><strong>Optional</strong>. Python script run after copying models/assets (e.g., install requirements). Skipped with <code>-e</code>.</td>
    </tr>
  </tbody>

  <!-- Second table section with its own header and body -->
  <thead>
    <tr class="header-row">
      <th><h3>Folder</h3></th>
      <th><h3>Description</h3></th>
    </tr>
  </thead>
  <tbody>   
    <tr>
      <td><strong>ComfyUI</strong></td>
      <td><strong>Required Folder</strong>. Contains subfolders/files to install into the local ComfyUI (e.g., <code>models/diffusion_models/</code>, <code>custom_nodes/</code>).</td>
    </tr>
  </tbody>

  <!-- Final header section for "Additional Notes" -->
  <thead>
    <tr class="header-row">
      <th colspan="2"><h3>Additional Notes</h3></th>
    </tr>
  </thead> 
  <tbody>
    <tr>
      <td colspan="2"><strong>Models, LoRAs, custom nodes, and inputs</strong> should be placed into their respective folders within the <code>ComfyUI</code> folder in the zipped package to mimic the installed location in ComfyUI.</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Images, videos, and similar assets</strong> should be set up in the workflow to reference these assets from the <code>inputs</code> folder; include any required assets in the package.</td>
    </tr>
  </tbody>
</table>

Example package structure (unzipped view):
```
- baseconfig.json
- warmup.json
- workflow.json
- pre.py
- post.py
- ComfyUI/
  - models/
    - diffusion_models/
      - model.safetensors
    - vae/
      - vae.safetensors
  - custom_nodes/
    - custom_node_dir/
      - requirements.txt
  - input/
    - input.jpg
```

Zip the contents directly (no top-level folder in the ZIP).

## Output and Metrics
The script outputs progress (queuing/completion) and a results summary:
```
####_RESULTS_SUMMARY_####

Total time to generate X images: Y.YY seconds
Number of images per minute: Z.ZZ
Average time (secs) per image: A.AA
Total Execution Time (main generations): B.BB seconds
Average Execution Time Per Image (main generations): C.CC seconds

Applied Overrides:
  Main Workflow Overrides:
    - override_key: Set override_item to override_value in nodes [node_id (node_title)]
```
- If `comfyui-benchmark` is installed, it generates JSON files like `flux.1_krea_fp8_1024x1024x20_YYYYMMDD_HHMMSS_RUN_N.M.json` (N=generation, M=instance).
- Logs include detailed execution times, errors, and applied overrides.

## Troubleshooting
- **Server not starting**: Check port conflicts or increase timeout in `check_server_ready`. Use `-p` to match running ComfyUI instances.
- **Missing models**: Ensure package includes all required files; avoid `-e` on first run.
- **Overrides not applying**: Verify `overrides.json` format and node identifiers. Check logs for warnings about unmatched nodes.
- **Keyboard Interrupt**: Partial metrics are calculated and logged.
- **Custom Nodes**: Required Custom node should have their directories added to the ComfyUI/custom_nodes folder within the zipped benchmark package.  Custom nodes which require addiition setup can include any setup commands/processes via an included `post.py` or manually; errors may occur if dependencies are missing.
- **Disk Space Issues**: Use `-t` to specify an alternate temporary directory with sufficient space.

For issues, check the log file or run without `-l` for console output.