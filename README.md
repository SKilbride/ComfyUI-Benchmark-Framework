

# ComfyUI Benchmarking Framework
## Running the Benchmark
The benchmarking framework works by running prebuilt benchmarking packages. These packages contain all the necessary primary and secondary model files required to run a dedicated benchmark workflow. Workflow files are ComfyUI json workflow files configured for running using the ComfyUI API mode.
Information on creating benchmark package files can be found in the Benchmark Package section of this document.

**Warmup:**  The each benchmarking run will perform an initial warmup run to load models into memory and initialize the pipeline, the warmup time is not included in any of the benchmarking metrics.


usage: run_comfyui_instances_concurrent.py [-h HELP] [-n NUM_INSTANCES] [-c COMFY_PATH] [-w WORKFLOW_PATH] [-g GENERATIONS] [-e EXTRACT MINIMAL] [-r RUN DEFAULT] [-l LOG]

Run multiple ComfyUI instances for simultaneous image generations.

    options: -h, --help show this help message and exit 
	-n , --num_instances  Number of simultaneous ComfyUI instances. 
	-c , --comfy_path  Path to the ComfyUI directory. **Required**
	-w, --workflow_path  Path to the JSON workflow file or ZIP file.  **Required**
	-g , --generations  Number of generations per instance.
	-e, --extract_minimal  Extract only basic work files from ZIP.
	-r, --run_default Use default recipe (baseconfig.json)
	-l, --log Log console output to a file*. *If no path is provided, use workflow basename + timestamp (yymmdd_epochtime.txt). If a path is provided, use it as is (if file) or append timestamp (if directory).

## Benchmarking Options:
The comfy path (-c) and workflow path (-w) are required. All other arguments are optional and if omitted the default values will be used by the benchmark framework.
**COMFY PATH** [-c, --comfy_path]
The comfy path argument must be the path to the ComfyUI folder within your ComfyUI installation. 

**WORKFLOW PATH** [-w, --workflow_path]
The workflow path is the fully qualified path to the .zip benchmark package file. 

**GENERATIONS** [-g, --generations] {Default Value: 1}
An integer value which represents the number of image or video file generations to complete. For image based workflows this will be general a number > 1.  For video files this will generally be =1  since video generation is expected to take significantly longer than image generation, however larger values can be used if desired.

**EXTRACT MINIMAL** [-e, --extract_minimal] 
When the extract minimal flag is set, only the basic files needed to run the workflow will be extracted from the .zip package. Other files such as primary and secondary models and image files will not be extracted from the benchmark package. Extract minimal assumes that the benchmark has previously been run on the system and these necessary files have been installed on the initial run. Failures will occur if extract minimal is used without the other necessary workflow files already existing on the test system. 

**RUN DEFAULT** [-r, --run_default] 
Benchmark packages should contain a baseconfig.json file which provides values to use for the num_instances and generations values for the benchmark run. Setting the -r flag will use the values in the baseconfig.json file so that the user does not have to manually specify these values. A different configuration json file can be provided as an optional parameter when the -r flag is set.  A sample configuration json file format is show below:

    {
    "NUM_INSTANCES": 1,
    "GENERATIONS": 1
    }

**LOG** [-l, --log]
When the log flag is set, the benchmark output will be written to a text file. By default the text file name will use the following format:  
*workflow basename* **+** *timestamp (yymmdd_epochtime)*.txt  (ie. flux.1_krea_1024x1024x20_250814_1755227444.txt) This text file will be saved at the location of the benchmark script. If a path is provided as an optional parameter the benchmark file will be saved to the specified location. 

## Benchmark Package Format
The benchmark package is a zipped archive that contains all of the necessary model files and assets needed to run the workflow, along with the workflow.json file. 
Files:
|File   | Description |
|--|--|
| **workflow.json**  | (REQUIRED) The ComfyUI workflow json file exported using the Export (API) function in ComfyUI  |
| **warmup.json**  | (OPTIONAL) A modified version of the  ComfyUI workflow json file exported using the Export (API) function in ComfyUI to be run once before the main benchmarking workflow. The warmup must utilize all of the same models and features as the workflow.json. If a warmup.json file is not supplied the workflow.json file will be used for warmup. warmup.json is particularly useful for video workloads, where it is not desirable to run an entire video generation as a warmup, in these cases the warmup.json can generally be limited to generating a single video frame. |
| **pre.py**  | (OPTIONAL) If provided the pre.py file is run before model and asset files are copied to the comfyui installation. This allows the script to create any specialized directories or files which may be needed for a workflow.  |
| **post.py**  | (OPTIONAL) If provided the post.py file is run after model and asset files are copied to the comfyui installation. This allows the script to install dependencies (ie. requirements.txt files) which may be needed for installed custom nodes or other requirements needed for the workflow.  |
| **ComfyUI**  | (Required)[Folder] The ComfyUI folder contains other folders and files which will be installed into the local ComfyUI folder of the local Comfy installation. This folder needs to contain all primary models, secondary models, lora, images, etc which are needed to run the workflow.   |
<img width="1166" height="573" alt="image" src="https://github.com/user-attachments/assets/73c65c09-7557-4c98-bef4-d4c883461b9a" />
The image above show the package contents for a Wan2.2 image to video workflow benchmark.

 - The **root folder** contains the *basesconfig.json*, *warmup.json*, *workflow.json*, and *ComfyUI folder*
 - The **ComfyUI folder** contains the subfolders: *models*, and *inputs*
 - The **models** folder contains the *diffusion models subfolder*, which contains the primary Wan diffusion models, the *text_encoders folder* contains the *umt5_xxl text encoder*, and the *vae* folder contains the *wan 2.1 vae model*
 - The **inputs** folder contains the *input.jpg* and *Pose_1.png* image files which are used within the workflow.

The package needs to be zipped so as to create no extra folders under the root folder.
