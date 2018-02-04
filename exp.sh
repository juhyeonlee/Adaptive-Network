#! /bin/bash

# network size
# network size (area) = (2*(nw_size-2))^2
# averaged number of agents = network size(area) * 4/5
python main.py --nw_size 7 --coeff 3.0 --beta 0.0  #network size(area) = 100, #averaged number of agents = 80
python main.py --nw_size 12 --coeff 3.0 --beta 0.0 #network size(area) = 400, #averaged number of agents = 320
python main.py --nw_size 17 --coeff 3.0 --beta 0.0 #network size(area) = 900, #averaged number of agents = 720
python main.py --nw_size 22 --coeff 3.0 --beta 0.0 #network size(area) = 1600, #averaged number of agents = 1280
python main.py --nw_size 27 --coeff 3.0 --beta 0.0 #network size(area) = 2500, #averaged number of agents = 2000

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
