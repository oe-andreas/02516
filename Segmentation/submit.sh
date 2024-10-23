#BSUB -J Segmentation
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 15
#BSUB -R "rusage[mem=4GB]"
#BSUB -o Segmentation.out
#BSUB -e Segmentation.err
#BSUB -R "span[hosts=1]"
#BSUB -n 4

# Initialize Python environment
source ../../../../02516_env/bin/activate

# Run Python script
python -m module.main