# How to Build Custom Apptainer

1. Create a new `run_build.sh` script based on the following template:

```bash
#!/usr/bin/env bash
# run_build.sh

export REMOTE="account@login.hpc.virginia.edu:/path/to/build/dir/"
./build.sh
```

2. Modify `face_fft.def` to add or modify any dependencies based on your project.

3. Run `./run_build.sh` inside the `apptainer/` folder to build the apptainer.
