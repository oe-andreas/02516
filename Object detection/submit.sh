#BSUB -J Object
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 120
#BSUB -R "rusage[mem=4GB]"
#BSUB -o Object.out
#BSUB -e Object.err
#BSUB -R "span[hosts=1]"
#BSUB -n 4

# Initialize Python environment
source ../../../../02516_env/bin/activate

# Run Python script
python3 main.py