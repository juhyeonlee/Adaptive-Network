#! /bin/bash

# network size
python main.py --nw_size 6 --coeff 3.0 --beta 0.0
python main.py --nw_size 8 --coeff 3.0 --beta 0.0
python main.py --nw_size 10 --coeff 3.0 --beta 0.0
python main.py --nw_size 20 --coeff 3.0 --beta 0.0
python main.py --nw_size 30 --coeff 3.0 --beta 0.0

#utility coeff 
python main.py --nw_size 8 --coeff 1.0 --beta 0.0
python main.py --nw_size 8 --coeff 3.0 --beta 0.0
python main.py --nw_size 8 --coeff 5.0 --beta 0.0
python main.py --nw_size 8 --coeff 7.0 --beta 0.0

#beta set
python main.py --nw_size 8 --coeff 3.0 --beta 0.0
python main.py --nw_size 8 --coeff 3.0 --beta 0.05
python main.py --nw_size 8 --coeff 3.0 --beta 0.1
python main.py --nw_size 8 --coeff 3.0 --beta 0.15
python main.py --nw_size 8 --coeff 3.0 --beta 0.2
python main.py --nw_size 8 --coeff 3.0 --beta 0.25
python main.py --nw_size 8 --coeff 3.0 --beta 0.3
