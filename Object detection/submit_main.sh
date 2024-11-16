#BSUB -J main
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 120
#BSUB -R "rusage[mem=4GB]"
#BSUB -o main.out
#BSUB -e main.err
#BSUB -R "span[hosts=1]"
#BSUB -n 2

# Initialize Python environment
source ../../../02516_env/bin/activate

# Run Python script
python main.py