% Value iteration & TCLE simulator for ICML2018
% 2018-02-06
% Minhae Caslia Kwon

%% 0. parameter setting
clear all;
close all;

%nework paramters
mu = 4/5;
%one_dim = 5;
one_dim_space = [5, 7, 9];

S_tx_r = 2; % tx range of source node
max_tx_r = 3;
beta = 0;

% value iteration parameters
rho = 0.9; % foresighted
R_weight = 2;
u = 0.2;
w=0.53; % weight between cost and reward [0,1], R*w+cost*(1-w)
a_range = -1:0.1:1; % action set
e = 0.01; % epsilon-optimal
epsilon=0.1;% for TCLE


MAX_beta = 0;

beta = 0;
num_ep=1; % max episode
ep_length=100; % number of steps per an episode
MAX_times =num_ep*ep_length;%100;
%%%%%%

len_players=zeros(1,num_ep);

VI_goodput=zeros(length(one_dim_space),MAX_times);
VI_energy=zeros(length(one_dim_space),MAX_times);
VI_con_ratio=zeros(length(one_dim_space),MAX_times);

TCLE_goodput=zeros(length(one_dim_space),MAX_times);
TCLE_energy=zeros(length(one_dim_space),MAX_times);
TCLE_con_ratio=zeros(length(one_dim_space),MAX_times);

for one_dim_idx = 1: length(one_dim_space)
    one_dim = one_dim_space(one_dim_idx);
    MAX_block = one_dim^2;
    times = 1;
    MAX_players = ceil((2*(one_dim-2))^2*4/5);
    len_state = MAX_players;
    %% 1. Value Iteration policy finder
    [final_PI,ini_state] = policy_finder(len_state, a_range, u, R_weight, w, mu, rho, e);
    for episode = 1:num_ep
        %episode
        %% 2. network initialization
        [len_players, players, D1_idx, D2_idx,S1_idx,S2_idx, d,  tx_r, noEvents, block_coords, node_loc] = network_initializer(MAX_block, mu, one_dim, S_tx_r, beta);
        
        % value iteration initial tx_r
        VI_tx_r = tx_r;
        for p = players
            A = sort(d(p,:));
            if length(A)-2<ini_state
                VI_tx_r(p) = A(length(A)-2);
            else
                VI_tx_r(p) = A(ini_state);
            end
        end
        
        for step = 1:ep_length
            %step
            %% 3. Get TCLE network
            [TCLE_adj_matrix, TCLE_tx_r, TCLE_energy(one_dim_idx,times)] = TCLE(max_tx_r, d, players,epsilon,S1_idx,S2_idx, S_tx_r, beta);
            
            
            %% 4. ValueIteration network
            [ VI_adj_matrix,VI_tx_r, VI_energy(one_dim_idx,times)] = ValueIteration(players, d, beta, final_PI, VI_tx_r, a_range, max_tx_r,len_state,S1_idx,S2_idx, S_tx_r);
            
            %% 5. Evaluate NW performance

            [VI_con_ratio(one_dim_idx,times),VI_goodput(one_dim_idx,times)] = Get_NW_Performance (VI_adj_matrix,S1_idx,S2_idx,D1_idx, D2_idx);
            [TCLE_con_ratio(one_dim_idx,times),TCLE_goodput(one_dim_idx,times)] = Get_NW_Performance (TCLE_adj_matrix,S1_idx,S2_idx,D1_idx, D2_idx);
            
            %% 6. NW node location update
            if step < ep_length
                % node location update
                [d, node_loc] = Update_location (MAX_block, noEvents, block_coords, node_loc, D1_idx, D2_idx,S1_idx,S2_idx, one_dim);
                times = times +1;
            end
            
            %VI_con_ratio(one_dim_idx,times-1)
            %TCLE_con_ratio(one_dim_idx,times-1)
           
        end
    end
end

save('TCLE+VI.mat');
 
