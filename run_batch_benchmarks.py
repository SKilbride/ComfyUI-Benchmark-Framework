r"""
Batch Benchmark Runner - Automate running multiple benchmarks with different configurations.

This script allows you to run multiple workflows with different parameter combinations
(e.g., normal, --lowvram, --lowvram --async-offload) and collect all results automatically.

Usage:
    python run_batch_benchmarks.py --config batch_config.yaml
    python run_batch_benchmarks.py -c <comfy_path> -w <workflow_path> --configs normal,lowvram,lowvram_async
    python run_batch_benchmarks.py -c <comfy_path> -w workflow1.zip -w workflow2.zip --configs normal,lowvram
    python run_batch_benchmarks.py -c <comfy_path> -w workflow1.zip,workflow2.zip --configs normal,lowvram

Examples:
    # Single workflow with multiple configs
    python run_batch_benchmarks.py -c E:\ComfyUI -w workflow.zip --configs normal,lowvram,lowvram_async
    
    # Multiple workflows - repeat -w flag
    python run_batch_benchmarks.py -c E:\ComfyUI -w wf1.zip -w wf2.zip -w wf3.zip --configs normal,lowvram
    
    # Multiple workflows - comma-separated
    python run_batch_benchmarks.py -c E:\ComfyUI -w wf1.zip,wf2.zip,wf3.zip --configs normal,lowvram
    
    # Using a directory of workflows
    python run_batch_benchmarks.py -c E:\ComfyUI --workflow-dir E:\workflows --configs normal,lowvram
    
    # Run with YAML config file
    python run_batch_benchmarks.py --config my_benchmarks.yaml
"""

import argparse
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Predefined benchmark configurations
PRESET_CONFIGS = {
    "normal": {
        "name": "Normal (default)",
        "extra_args": []
    },
    "lowvram": {
        "name": "Low VRAM",
        "extra_args": ["--lowvram"]
    },
    "lowvram_async": {
        "name": "Low VRAM + Async Offload",
        "extra_args": ["--lowvram", "--async-offload"]
    },
    "gpu_only": {
        "name": "GPU Only",
        "extra_args": ["--gpu-only"]
    },
    "highvram": {
        "name": "High VRAM",
        "extra_args": ["--highvram"]
    },
    "cpu": {
        "name": "CPU Mode",
        "extra_args": ["--cpu"]
    },
    "fp16": {
        "name": "FP16 VAE",
        "extra_args": ["--fp16-vae"]
    },
    "bf16": {
        "name": "BF16 VAE",
        "extra_args": ["--bf16-vae"]
    },
}


def load_yaml_config(config_path: str) -> dict:
    """Load benchmark configuration from YAML file."""
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except ImportError:
        print("PyYAML not installed. Using JSON fallback.")
        # Try JSON
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def discover_workflows(workflow_dir: str, patterns: List[str] = None) -> List[str]:
    """
    Discover workflow files in a directory.
    
    Args:
        workflow_dir: Directory to search
        patterns: File patterns to match (default: *.zip, *.json)
        
    Returns:
        List of workflow file paths
    """
    if patterns is None:
        patterns = ["*.zip", "*.json"]
    
    workflows = []
    workflow_dir = Path(workflow_dir)
    
    for pattern in patterns:
        matches = list(workflow_dir.glob(pattern))
        workflows.extend([str(m) for m in matches])
    
    # Sort by name for consistent ordering
    workflows.sort()
    return workflows


def run_single_benchmark(
    comfy_path: str,
    workflow_path: str,
    output_dir: str,
    config_name: str,
    extra_args: List[str],
    generations: int = 5,
    port: int = 8188,
    vram_monitor: bool = True,
    timeout: int = 4000,
    no_cleanup: bool = False,
    capture_output: bool = True,
) -> Dict[str, Any]:
    """
    Run a single benchmark with the specified configuration.
    
    Returns:
        Dictionary with benchmark results and metadata
    """
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    workflow_basename = Path(workflow_path).stem
    
    # Generate output filenames
    log_filename = f"benchmark_{workflow_basename}_{config_name}_{timestamp}.txt"
    vram_log_filename = f"gpu_log_{workflow_basename}_{config_name}_{timestamp}.csv"
    full_output_filename = f"full_output_{workflow_basename}_{config_name}_{timestamp}.txt"
    
    log_path = os.path.join(output_dir, log_filename)
    vram_log_path = os.path.join(output_dir, vram_log_filename)
    full_output_path = os.path.join(output_dir, full_output_filename)
    
    # Build command
    cmd = [
        sys.executable,
        "run_comfyui_benchmark_framework.py",
        "-c", comfy_path,
        "-w", workflow_path,
        "-g", str(generations),
        "-p", str(port),
        "-l", log_path,
        "--timeout", str(timeout),
    ]
    
    if vram_monitor:
        cmd.extend(["--vram-monitor", "--vram-log", vram_log_path])
    
    if no_cleanup:
        cmd.append("--no-cleanup")
    
    # Add extra args (like --lowvram, --async-offload)
    if extra_args:
        cmd.append("--extra_args")
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Workflow: {workflow_basename}")
    print(f"Config: {config_name}")
    print(f"Extra args: {' '.join(extra_args) if extra_args else '(none)'}")
    print(f"Log file: {log_path}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    full_output = []
    
    try:
        # Use Popen to capture output while also streaming to console
        process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace',
        )
        
        # Stream output to console and capture it
        for line in process.stdout:
            print(line, end='', flush=True)
            if capture_output:
                full_output.append(line)
        
        process.wait()
        elapsed = time.time() - start_time
        success = process.returncode == 0
        
        # Save full output to file
        if capture_output and full_output:
            try:
                with open(full_output_path, 'w', encoding='utf-8') as f:
                    f.writelines(full_output)
            except Exception as e:
                print(f"Warning: Failed to save full output: {e}")
        
        return {
            "workflow": workflow_basename,
            "workflow_path": workflow_path,
            "config_name": config_name,
            "success": success,
            "return_code": process.returncode,
            "elapsed_time": elapsed,
            "log_file": log_path,
            "full_output_file": full_output_path if capture_output else None,
            "vram_log_file": vram_log_path if vram_monitor else None,
            "extra_args": extra_args,
        }
        
    except Exception as e:
        return {
            "workflow": workflow_basename,
            "workflow_path": workflow_path,
            "config_name": config_name,
            "success": False,
            "error": str(e),
            "log_file": log_path,
            "extra_args": extra_args,
        }


def parse_benchmark_log(log_path: str) -> Dict[str, Any]:
    """Parse a benchmark log file to extract results."""
    results = {}
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for the results summary
        if "####_RESULTS_SUMMARY_####" in content:
            lines = content.split("####_RESULTS_SUMMARY_####")[1].split("\n")
            for line in lines:
                if "Benchmarking Package:" in line:
                    results["package"] = line.split(":", 1)[1].strip()
                elif "VRAM:" in line:
                    results["vram_line"] = line.strip()
                elif "Number of Generations:" in line:
                    results["generations"] = line.split(":", 1)[1].strip()
                elif "Workflow Execution Time:" in line:
                    results["timing_line"] = line.strip()
                elif "Assets Generated:" in line:
                    results["assets_line"] = line.strip()
                    
    except Exception as e:
        results["parse_error"] = str(e)
        
    return results


def run_batch_benchmarks(
    comfy_path: str,
    workflows: List[str],
    configs: List[str],
    output_dir: str = None,
    generations: int = 5,
    port: int = 8188,
    vram_monitor: bool = True,
    timeout: int = 4000,
    no_cleanup: bool = False,
    delay_between: int = 30,
    capture_output: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run multiple benchmarks with different workflows and configurations.
    
    Args:
        comfy_path: Path to ComfyUI installation
        workflows: List of workflow file paths
        configs: List of config names (from PRESET_CONFIGS or custom)
        output_dir: Directory for output files (default: current dir)
        generations: Number of generations per benchmark
        port: ComfyUI server port
        vram_monitor: Enable VRAM monitoring
        timeout: Timeout in seconds
        no_cleanup: Don't cleanup after each run
        delay_between: Seconds to wait between benchmarks
        capture_output: Capture and save full console output
        
    Returns:
        List of result dictionaries
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    total_runs = len(workflows) * len(configs)
    current_run = 0
    
    print(f"\n{'#'*70}")
    print(f"BATCH BENCHMARK - {len(workflows)} workflow(s) x {len(configs)} config(s) = {total_runs} total runs")
    print(f"{'#'*70}")
    print(f"\nWorkflows:")
    for w in workflows:
        print(f"  - {Path(w).name}")
    print(f"\nConfigs: {', '.join(configs)}")
    print(f"Output directory: {output_dir}")
    print(f"{'#'*70}\n")
    
    workflow_results = {}  # Group results by workflow
    
    for workflow_path in workflows:
        workflow_name = Path(workflow_path).stem
        workflow_results[workflow_name] = []
        
        print(f"\n{'='*70}")
        print(f"WORKFLOW: {workflow_name}")
        print(f"{'='*70}")
        
        for config_name in configs:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] {workflow_name} + {config_name}")
            
            # Get config details
            if config_name in PRESET_CONFIGS:
                config = PRESET_CONFIGS[config_name]
                extra_args = config["extra_args"]
                display_config = config_name
            else:
                # Treat as custom args (comma or space separated)
                extra_args = config_name.replace(",", " ").split()
                display_config = config_name.replace(" ", "_").replace(",", "_")
            
            result = run_single_benchmark(
                comfy_path=comfy_path,
                workflow_path=workflow_path,
                output_dir=output_dir,
                config_name=display_config,
                extra_args=extra_args,
                generations=generations,
                port=port,
                vram_monitor=vram_monitor,
                timeout=timeout,
                no_cleanup=no_cleanup,
                capture_output=capture_output,
            )
            
            # Parse log file for results
            if result.get("log_file") and os.path.exists(result["log_file"]):
                result["parsed_results"] = parse_benchmark_log(result["log_file"])
            
            all_results.append(result)
            workflow_results[workflow_name].append(result)
            
            # Print quick summary
            status = "✓ SUCCESS" if result.get("success") else "✗ FAILED"
            print(f"\n[{current_run}/{total_runs}] {workflow_name} + {display_config}: {status}")
            
            if result.get("parsed_results"):
                parsed = result["parsed_results"]
                if "vram_line" in parsed:
                    print(f"  {parsed['vram_line']}")
                if "assets_line" in parsed:
                    print(f"  {parsed['assets_line']}")
            
            # Wait between benchmarks (except for last one)
            if current_run < total_runs and delay_between > 0:
                print(f"\nWaiting {delay_between}s before next benchmark...")
                time.sleep(delay_between)
    
    # Print final summary grouped by workflow
    print(f"\n{'#'*70}")
    print("BATCH BENCHMARK COMPLETE - SUMMARY")
    print(f"{'#'*70}\n")
    
    for workflow_name, results in workflow_results.items():
        print(f"\n{'='*50}")
        print(f"WORKFLOW: {workflow_name}")
        print(f"{'='*50}")
        
        for result in results:
            status = "✓" if result.get("success") else "✗"
            config = result["config_name"]
            print(f"\n{status} {config}")
            
            if result.get("parsed_results"):
                parsed = result["parsed_results"]
                if "vram_line" in parsed:
                    print(f"    {parsed['vram_line']}")
                if "timing_line" in parsed:
                    print(f"    {parsed['timing_line']}")
                if "assets_line" in parsed:
                    print(f"    {parsed['assets_line']}")
            
            print(f"    Log: {result.get('log_file', 'N/A')}")
            if result.get("full_output_file"):
                print(f"    Full output: {result.get('full_output_file')}")
    
    # Save summary to JSON
    summary_path = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%y%m%d_%H%M%S')}.json")
    try:
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "comfy_path": comfy_path,
            "workflows": workflows,
            "configs": configs,
            "generations": generations,
            "total_runs": total_runs,
            "results": all_results,
            "results_by_workflow": {k: v for k, v in workflow_results.items()},
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, default=str)
        print(f"\n\nSummary saved to: {summary_path}")
    except Exception as e:
        print(f"Failed to save summary: {e}")
    
    # Also save a CSV summary for easy comparison
    csv_path = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%y%m%d_%H%M%S')}.csv")
    try:
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Workflow', 'Config', 'Success', 'VRAM (MB)', 'VRAM (GB)', 'Delta (MB)', 
                           'Exec Time (s)', 'Avg per Asset (s)', 'APM', 'Log File'])
            
            for result in all_results:
                parsed = result.get("parsed_results", {})
                
                # Extract VRAM values
                vram_mb = ""
                vram_gb = ""
                delta_mb = ""
                if "vram_line" in parsed:
                    vram_line = parsed["vram_line"]
                    # Parse "VRAM: 51223 MB (50.02 GB) | Delta: 45382 MB"
                    import re
                    match = re.search(r'VRAM:\s*(\d+)\s*MB\s*\(([\d.]+)\s*GB\)', vram_line)
                    if match:
                        vram_mb = match.group(1)
                        vram_gb = match.group(2)
                    delta_match = re.search(r'Delta:\s*(\d+)\s*MB', vram_line)
                    if delta_match:
                        delta_mb = delta_match.group(1)
                
                # Extract timing values
                exec_time = ""
                avg_per_asset = ""
                apm = ""
                if "assets_line" in parsed:
                    assets_line = parsed["assets_line"]
                    # Parse "Assets Generated: 5 | Avg Execution time per Asset: 25.19s | APM: 2.38"
                    avg_match = re.search(r'Avg.*?:\s*([\d.]+)s', assets_line)
                    if avg_match:
                        avg_per_asset = avg_match.group(1)
                    apm_match = re.search(r'APM:\s*([\d.]+)', assets_line)
                    if apm_match:
                        apm = apm_match.group(1)
                
                if "timing_line" in parsed:
                    timing_line = parsed["timing_line"]
                    time_match = re.search(r'Workflow Execution Time:\s*([\d.]+)s', timing_line)
                    if time_match:
                        exec_time = time_match.group(1)
                
                writer.writerow([
                    result.get("workflow", ""),
                    result.get("config_name", ""),
                    "Yes" if result.get("success") else "No",
                    vram_mb, vram_gb, delta_mb,
                    exec_time, avg_per_asset, apm,
                    result.get("log_file", "")
                ])
        
        print(f"CSV summary saved to: {csv_path}")
    except Exception as e:
        print(f"Failed to save CSV summary: {e}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple benchmarks with different workflows and configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available preset configurations:
  normal         - Default settings
  lowvram        - --lowvram flag
  lowvram_async  - --lowvram --async-offload
  gpu_only       - --gpu-only flag
  highvram       - --highvram flag
  cpu            - --cpu mode
  fp16           - --fp16-vae
  bf16           - --bf16-vae

Examples:
  # Single workflow
  python run_batch_benchmarks.py -c E:\\ComfyUI -w workflow.zip --configs normal,lowvram,lowvram_async
  
  # Multiple workflows (repeat -w flag)
  python run_batch_benchmarks.py -c E:\\ComfyUI -w wf1.zip -w wf2.zip -w wf3.zip --configs normal,lowvram
  
  # Multiple workflows (comma-separated)
  python run_batch_benchmarks.py -c E:\\ComfyUI -w wf1.zip,wf2.zip,wf3.zip --configs normal,lowvram
  
  # Workflow directory (all .zip and .json files)
  python run_batch_benchmarks.py -c E:\\ComfyUI --workflow-dir E:\\my_workflows --configs normal,lowvram
  
  # YAML config file
  python run_batch_benchmarks.py --config batch_config.yaml
        """
    )
    
    parser.add_argument("-c", "--comfy_path", type=str, help="Path to ComfyUI folder")
    parser.add_argument("-w", "--workflow_path", type=str, action="append", dest="workflows",
                        help="Path to workflow ZIP or folder (repeat -w or use comma-separated list)")
    parser.add_argument("--workflow-dir", type=str, help="Directory containing workflow files to benchmark")
    parser.add_argument("--configs", type=str, default="normal,lowvram,lowvram_async",
                        help="Comma-separated list of config names (default: normal,lowvram,lowvram_async)")
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    parser.add_argument("-g", "--generations", type=int, default=5, help="Generations per benchmark")
    parser.add_argument("-p", "--port", type=int, default=8188, help="ComfyUI port")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory for logs")
    parser.add_argument("--no-vram-monitor", action="store_true", help="Disable VRAM monitoring")
    parser.add_argument("--no-capture", action="store_true", help="Don't capture full console output")
    parser.add_argument("--timeout", type=int, default=4000, help="Timeout in seconds")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't cleanup after runs")
    parser.add_argument("--delay", type=int, default=5, help="Seconds between benchmarks")
    parser.add_argument("--list-configs", action="store_true", help="List available preset configs")
    
    args = parser.parse_args()
    
    # List configs mode
    if args.list_configs:
        print("Available preset configurations:\n")
        for name, config in PRESET_CONFIGS.items():
            args_str = ' '.join(config['extra_args']) if config['extra_args'] else '(no extra args)'
            print(f"  {name:20} - {config['name']:30} | {args_str}")
        return
    
    # Load from config file if specified
    if args.config:
        config_data = load_yaml_config(args.config)
        comfy_path = config_data.get("comfy_path", args.comfy_path)
        
        # Support both single workflow_path and list of workflows
        workflows = config_data.get("workflows", [])
        if not workflows and "workflow_path" in config_data:
            workflows = [config_data["workflow_path"]]
        if not workflows and args.workflows:
            workflows = args.workflows
            
        configs = config_data.get("configs", args.configs.split(","))
        generations = config_data.get("generations", args.generations)
        output_dir = config_data.get("output_dir", args.output_dir)
    else:
        if not args.comfy_path:
            parser.error("--comfy_path (-c) is required")
        
        comfy_path = args.comfy_path
        
        # Gather workflows from various sources
        workflows = []
        
        # Process -w arguments (support both repeated -w and comma-separated)
        if args.workflows:
            for w in args.workflows:
                # Split by comma in case user passed comma-separated list
                if ',' in w:
                    workflows.extend([p.strip() for p in w.split(',') if p.strip()])
                else:
                    workflows.append(w)
        
        # Add workflows from directory if specified
        if args.workflow_dir:
            dir_workflows = discover_workflows(args.workflow_dir)
            workflows.extend(dir_workflows)
            print(f"Discovered {len(dir_workflows)} workflows in {args.workflow_dir}")
        
        if not workflows:
            parser.error("At least one workflow is required (-w or --workflow-dir)")
        
        configs = [c.strip() for c in args.configs.split(",")]
        generations = args.generations
        output_dir = args.output_dir
    
    run_batch_benchmarks(
        comfy_path=comfy_path,
        workflows=workflows,
        configs=configs,
        output_dir=output_dir,
        generations=generations,
        port=args.port,
        vram_monitor=not args.no_vram_monitor,
        timeout=args.timeout,
        no_cleanup=args.no_cleanup,
        delay_between=args.delay,
        capture_output=not args.no_capture,
    )


if __name__ == "__main__":
    main()
