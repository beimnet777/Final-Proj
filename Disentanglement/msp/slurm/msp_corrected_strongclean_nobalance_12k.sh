#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp_sc_fixctl
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.err

# Conservative companion to msp_corrected_strongclean_12k.sh.
# It uses the same data, seed, 4k quota-freeze, 12k DANN/training schedule,
# separate optimizers, per-group clipping, valid-frame AuxK-64, losses, and
# checkpoint cadence.  The sole experimental difference is:
#
#   pcgrad_balance: unit -> none
#
# Submit this file directly with sbatch; the common recipe is executed below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MSP_RUN_NAME=msp_hardqfreeze4000_sc_optfix_nobal_aux64_gn0004_grlp025_dann12000_12k_s42
export MSP_PCGRAD_BALANCE=none

exec bash "${SCRIPT_DIR}/msp_corrected_strongclean_12k.sh"
