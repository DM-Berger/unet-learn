#!/bin/bash
#SBATCH --comment="======================================================================"
#SBATCH --job-name test_in_job_detect
#SBATCH --comment="==== See https://docs.computecanada.ca/wiki/PyTorch =================="
#SBATCH --gres=gpu:1
#SBATCH --comment="#SBATCH --mem=20000M"
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --time=15:00
#SBATCH --comment="======================================================================"
#SBATCH --output=torchtest__%j_stdout.out
#SBATCH --error=torchtest__%j_stderr.out
#SBATCH --comment="======================================================================"
#SBATCH --verbose
#SBATCH --comment="======================================================================"
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_50

# see:
# https://docs.computecanada.ca/wiki/Tutoriel_Apprentissage_machine/en
# and
# https://docs.computecanada.ca/wiki/PyTorch"
# for references and examples relevant to this submission script

SRC_DIR=/home/$USER/projects/def-jlevman/U-Net_MRI-Data/CalgaryCampinas359
PYSCRIPT=SRC_DIR/torch_test
LOG_DIR=$SLURM_TMPDIR/TensorBoard/"$PYSCRIPT"__"$(date +"%Y-%b%m-%d--%H:%M")"

cd $SLURM_SUBMIT_DIR
module load cuda/10.0.130 python/3.7.4 scipy-stack && echo "$(date +"%T"):  Successfully loaded modules"

# use local torch env with torchio pytorch-lightning tensorflow installed
source /home/$USER/torch/bin/activate && echo "$(date +"%T"):  Activated user local python virtualenv"

# copy data
mkdir -p $SLURM_TMPDIR/data
# --strip-components prevents making double parent directory
echo "$(date +"%T"):  Copying data"
tar xf $SRC_DIR/job_data.tar.gz -C $SLURM_TMPDIR/data --strip-components 1 && echo "$(date +"%T"):  Copied data"

# setup tensorboard live logging and run script
mkdir -p LOG_DIR
echo "$(date +"%T"):  Setting up TensorBoard and executing $PYSCRIPT"
tensorboard --logdir=LOG_DIR --host 0.0.0.0 & python "$PYSCRIPT.py" --logs $LOG_DIR --gpus 1 --epochs 10
# run script