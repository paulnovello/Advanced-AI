#!/bin/bash
#SBATCH -J mon_job
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p shared
#SBATCH --time=00:15:00 
#SBATCH --output=/users/formation/tpirtbgchs/job_results/out/job_%j.out
#SBATCH --error=/users/formation/tpirtbgchs/job_results/err/job_%j.err


apptainer exec \
--env PATH=$HOME/.local/bin:$PATH \
--env UV_PROJECT_ENVIRONMENT=/tmpdir/tpirtbgchs/envs/aai \
--bind /tmpdir,/work \
--nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif \
<<<<<<< HEAD
<<<<<<< HEAD
uv run --no-sync python mon_script.py
=======
=======
>>>>>>> 5d1b774ff2e662ba4f7076f9404deb20b7f7a9e1
<<<<<<< HEAD
uv run python test_sbatch.py
=======
uv run --no-sync python mon_script.py
>>>>>>> upstream/main
<<<<<<< HEAD
>>>>>>> 645fad7 (P-work)
=======
=======
uv run --no-sync python mon_script.py
>>>>>>> upstream/main
>>>>>>> 5d1b774ff2e662ba4f7076f9404deb20b7f7a9e1
