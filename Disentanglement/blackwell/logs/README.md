# Blackwell experiment logs

The shared Blackwell launcher writes combined stdout/stderr and run metadata to
`logs/<RUN_NAME>/`. Commit completed text logs so experiment analysis remains
reproducible from the recorded command, Git revision, device, and training
trace. Do not place checkpoints, TensorBoard event files, datasets, or model
caches here.
