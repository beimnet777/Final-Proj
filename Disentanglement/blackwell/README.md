# CBL Blackwell setup

This directory provides machine infrastructure only. Scientific flags live in
tracked scripts under `experiments/`, while the temporary GPU allocation stays
outside Git. No experiment is launched during setup.

## 1. Connect

First enter the Engineering network, then connect to the internal machine:

```bash
ssh YOUR_CRSID@gate.eng.cam.ac.uk
ssh Blackwell
```

You can optionally configure `ProxyJump` later, but the two-step form is the
simplest first-login diagnostic. Reset a forgotten CUED password using the
Engineering Department account instructions linked in the allocation email.

## 2. Clone the GitHub source

The SSDs `/scratch` and `/scratch2` are not backed up. Keep your clone and all
large files inside a directory named after your CRSid:

```bash
mkdir -p /scratch/$USER
cd /scratch/$USER
git clone https://github.com/beimnet777/Final-Proj.git
cd Final-Proj
git fetch origin
git checkout main
git pull --ff-only
git rev-parse HEAD
```

For a reproducible run, use a committed experiment script and record or check
out its exact commit SHA. Never commit datasets, caches, credentials, or model
checkpoints.

## 3. Create the environment

The dependencies used by the Colab notebook are reused here. The setup retains
a compatible system build when available and otherwise installs the repository's
matching `torch==2.11.0` and `torchaudio==2.11.0` pair from the official CUDA
13.0 index. It intentionally does not initialize CUDA before you have a GPU
allocation.

Check which Python versions exist, then run the setup with Python 3.10–3.14:

```bash
command -v python3
python3 --version
./Disentanglement/blackwell/setup.sh --download-librispeech
```

If `python3` is outside the supported range, select another installed Python:

```bash
BLACKWELL_PYTHON=python3.11 ./Disentanglement/blackwell/setup.sh --download-librispeech
```

The optional download fetches the same three OpenSLR splits used by Colab:
`train-clean-100`, `dev-clean`, and `test-clean`. Archives, extracted data,
Hugging Face caches, NLTK data, and the virtual environment all live under
`/scratch/$USER`. Re-run without `--download-librispeech` when only repairing
the Python environment.

The default official wheel index is `cu130`. If the machine administrators
prescribe a different compatible official index, pass it explicitly:

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/NAME \
  ./Disentanglement/blackwell/setup.sh
```

Do not guess `NAME`: Blackwell needs a CUDA 12.8-or-newer build because CUDA
12.8 introduced compiler support for Blackwell `sm_120` ([NVIDIA CUDA 12.8
features](https://docs.nvidia.com/cuda/archive/12.8.2/cuda-features-archive/index.html)).
Use the administrator recommendation or the current command from the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/). The
launcher performs a real CUDA smoke test after an assigned GPU is selected.

For licensed MSP material, follow `Disentanglement/COLAB.md` to create a private
verified bundle, copy it to `/scratch/$USER`, and extract it under the configured
data root. It must not be placed on GitHub.

## 4. Verify the non-GPU setup

```bash
source /scratch/$USER/venvs/final-proj/bin/activate
python -c "import torch, torchaudio, transformers; print(torch.__version__, torch.version.cuda)"
test -d /scratch/$USER/data/LibriSpeech/train-clean-100
test -f /scratch/$USER/data/librispeech-lexicon.txt
cat /scratch/$USER/setup_metadata.txt
deactivate
```

At this point the environment and data are ready. Stop here until a GPU has
been assigned in `blackwell_gpus`.

## 5. Prepare tracked experiment scripts later

Copy, edit, review, and commit the template:

```bash
cp Disentanglement/blackwell/experiments/template.sh \
   Disentanglement/blackwell/experiments/my_experiment.sh
$EDITOR Disentanglement/blackwell/experiments/my_experiment.sh
git add Disentanglement/blackwell/experiments/my_experiment.sh
git commit -m "Add my Blackwell experiment"
git push -u origin YOUR_BRANCH
```

All scientific arguments belong in that file. After Slack assigns (for example)
physical GPU 3, the eventual launch is deliberately short:

```bash
tmux new -s my_experiment
cd /scratch/$USER/Final-Proj
GPU_ID=3 ./Disentanglement/blackwell/experiments/my_experiment.sh
```

The current trainer is single-GPU. Request one GPU: making several devices
visible would not parallelize training. `common.sh` therefore accepts exactly
one allocated ID from 0 through 7, checks that PyTorch sees only logical
`cuda:0`, records the Git commit and fully escaped command, and stores combined
logs under `/scratch/$USER/runs/RUN_NAME/launcher_logs/`.

The tracked `libri_club_hybrid_gradnorm_s42.sh` experiment can first be checked
without training by setting `DRY_RUN=1`; this still requires an allocated GPU
because the shared launcher performs its CUDA smoke test.

## 6. Back up irreplaceable outputs

Scratch is explicitly unbacked. Periodically copy run directories to backed-up
storage with `rsync` or `scp`. Git is appropriate for source and experiment
definitions, not datasets or checkpoints. Release the GPU promptly in Slack
when a run finishes or fails.
