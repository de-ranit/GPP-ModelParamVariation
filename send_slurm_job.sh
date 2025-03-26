# All the parameters must be set based on computational platform you are using. Requires a slurm based platform.
#!/bin/bash
#SBATCH --job-name=my_cool_job                             # job name
#SBATCH --partition=Q_007                                  # partition name
#SBATCH --nodes=1                                          # nodes requested (min-max format or single number if min and max are same)
#SBATCH --ntasks=1                                         # tasks requested
##SBATCH --nodelist=(node-agent-[005,007,009,001])         # specific nodes to run the job
#SBATCH --array=1-11                                       # number of jobs %jobs allowed to run at a time
#SBATCH --cpus-per-task=20                                 # cores requested per job
##SBATCH --mem=70G                                         # memory per node
#SBATCH --mem-per-cpu=3G                                   # memory per core/ CPU
#SBATCH --output=./err_out_files/outfile_%x-%A_%a.log      # send stdout to outfile (create a directory named err_out_files)
#SBATCH --error=./err_out_files/errfile_%x-%A_%a.log       # send stderr to errfile (create a directory named err_out_files)
#SBATCH --time=2-2:00:00                                   # max time requested in D-hour:minute:second
#SBATCH --mail-user=bond@secret-service.xx                 # email address of the user
#SBATCH --mail-type=FAIL                                   # when to email the user (in case of job fail)
##SBATCH --dependency=afterany:999999                      # dependency on other job (job ID)

################# activate conda environment ##############################################
# Name of the conda environment desired
ENV_NAME="env_not_found" 

# if the conda environment is already not active, load the conda environment
if [[ $CONDA_DEFAULT_ENV != $ENV_NAME ]]; then

    # get location of conda.sh
    PATH_TO_CONDA=$(conda info | grep -i 'base environment' | cut -d ":" -f 2 | cut -d " " -f 2)
    PATH_TO_CONDA_BASH="${PATH_TO_CONDA}/etc/profile.d/conda.sh"
    
    # Initialize conda
    source $PATH_TO_CONDA_BASH 
    # Activate the environment
    conda activate $ENV_NAME
fi

############## per site optimization & forward run (all cases)#############################
python -u main_opti_and_run_model.py $SLURM_ARRAY_TASK_ID 20
# python -u main_opti_and_run_model.py 1 5 # test run

############## site year optimization #####################################################
# for site year optimization (copying codes to a working directory)
# then the codes/ settings can be modified without affecting the pending jobs
# work_dir=../site_yr_scratch # specify working directory
# mkdir -p $work_dir # create working directory

# current_dir=$(pwd) # get current directory

# # copy the necessary codes, files to the working directory
# rsync -aq --include='main_opti_and_run_model.py' --include='model_settings.xlsx' --include='/site_info/***' --include='/src/***' --exclude='*' $(pwd)/ $work_dir/

# cd $work_dir # change to working directory
# # perform site year optimization in the working directory
# python -u main_opti_and_run_model.py $SLURM_ARRAY_TASK_ID 20
# cd $current_dir # change back to the original directory

# # copy the results back to the original directory
# for outcmaes_dir in $work_dir/outcmaes/*/*; do
#     outcmaes_subpath=${outcmaes_dir#$work_dir}
#     mkdir -p $(pwd)/$outcmaes_subpath
#     rsync -aq $outcmaes_dir/ $(pwd)/$outcmaes_subpath
# done

# for opti_res_dir in $work_dir/opti_results/*/*; do
#     opti_res_subpath=${opti_res_dir#$work_dir}
#     mkdir -p $(pwd)/$opti_res_subpath
#     rsync -aq $opti_res_dir/ $(pwd)/$opti_res_subpath
# done

# # remove the working directory
# # rm -r $work_dir

############## global optimization #####################################################
# python -u main_opti_and_run_model.py
##########################################################################################