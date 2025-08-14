usage: run_comfyui_instances_concurrent.py [-h] [-n NUM_INSTANCES] -c COMFY_PATH -w WORKFLOW_PATH [-g GENERATIONS] [-e] [-r]

Run multiple ComfyUI instances for simultaneous image generations.

options:
  -h, --help                                 show this help message and exit
  -n , --num_instances        Number of simultaneous ComfyUI instances.
  -c , --comfy_path               Path to the ComfyUI directory.
  -w, --workflow_path          Path to the JSON workflow file or ZIP file.
  -g , --generations                Number of generations per instance.
  -e, --extract_minimal        Extract only .json files from ZIP.
  -r, --run_default                    Use default recipe (baseconfig.json)

  
