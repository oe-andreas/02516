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
source /zhome/66/4/156534/venv_1/bin/activate

# Run Python script
python3 test_F.py