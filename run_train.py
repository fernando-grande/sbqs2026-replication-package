"""Train STL or MTL models (see train/run.py for every option).

    python run_train.py --mode both --encoder all
    python run_train.py --mode stl --encoder scibert --task CT-Planning
"""
from train.run import main

if __name__ == "__main__":
    main()
