# Start a training with ovhai


## With service ovhai notebook
0. (Login in your terminal `ovhai login`, write 1)

1. Start a notebook with the correct params
```bash
ovhai notebook run conda [jupyterlab/vscode] --gpu 1 --framework-version pytorch1.10.1-py39-cuda10.2-v22-4 # checker espace client pour les options
ovhai notebook run --help
# example
ovhai notebook run pytorch jupyterlab \
	--name basegun-sarah-2CPU \
	--framework-version pytorch1.10.1-py39-cuda10.2-v22-4 \
	--flavor ai1-1-cpu \ # or ai1-1-gpu for GPU
	--cpu 2 \ # or --gpu 1
	--volume basegun@GRA/dataset/v1/:/workspace/data:RWD # or RO for read-only


ovhai capabilities framework ls
ovhai notebook [start/stop/delete] uuid
```

2. Create notebook `train.ipynb`and copy-paste content from `train.py`. Start training

3. Go back to jupyter terminal and run
```bash
tensorboard --logdir models --bind_all
```
4. Copy again job url and paste to browser while changing the end of url
`-8082.job.gra.training.ai.cloud.ovh.net/lab?`
by
`-6006.job.gra.training.ai.cloud.ovh.net`
to open tensorboard in browser

5. Remember to stop your job once finished !

NB:
* volume options same as for job
* difference job/notebook : docker image prepackaged, as long as we haven't deleted it we can start/stop the notebook
* when stopped, we pay the weight of /workspace (without volumes), when started same price as ai training
* we only pay a job/notebook when the status is "RUNNING" https://docs.ovh.com/gb/en/publiccloud/ai/notebooks


### Helper info
If need to test changes on dataset, do symlinks to copy
```bash
# symlink all typology folders from train dataset
find /workspace/data/train/ -maxdepth 1 -mindepth 1 -type d -exec ln -s '{}' /workspace/data_copy/train/ \;
# symlink 1000 random files from train/typo folder
find /workspace/data/train/typo -type f | shuf -n 1000 | xargs -I % ln -s % /workspace/data_copy/train/typo
```

## Legacy
# Run with OVHAi

0. (Login in your terminal `ovhai login`, write 1.)

1. Use Basegun-ml's Docker image to run job
```bash
ovhai job run --gpu 1 --name basegun-yourname-1GPU --volume basegun@GRA/dataset/v0/:/workspace/data:ro --volume basegun-public@GRA/models/:/workspace/models:rw ghcr.io/datalab-mi/basegun-ml:v0.1
```
