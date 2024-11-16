#BSUB -J CalcAP
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 60
#BSUB -R "rusage[mem=12GB]"
#BSUB -o CalcAP.out
#BSUB -e CalcAP.err
#BSUB -R "span[hosts=1]"
#BSUB -n 4

# Initialize Python environment
source ../../../02516_env/bin/activate

# Run Python script
python calculate_AP.py