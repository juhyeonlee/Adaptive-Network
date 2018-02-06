function [TCLE_adj_matrix, TCLE_tx_r, TCLE_energy] = TCLE(max_tx_r, d, players,epsilon,S1_idx,S2_idx, S_tx_r, beta)

% TCLE

% reference parameters
%Coef=7* 10^(-10)*(4*pi)^2 * (2.5*10^9/3/10^8)^2;
action_d = 0:0.1:max_tx_r;
%action_d = 0:0.1:3.2; %eta:6
action_set = action_d.^2; % tx power set = action set
E0=1; % full energy = 1J
fun = @(x)x.^2;
M = integral(fun,E0,E0+max(action_set));
%epsilon_set = [0 0.03 0.05, 0.2 0.5 0.8 ];% TCLE parameter :NW connection measure

%epsilon= epsilon_set(4);
%tot_tot_NW_U = zeros(length(COST_range),max_N);




% initial distance = max
D_current = max(action_d)*ones(1, length(d)); %distance
idx_current = length(action_set)*ones (1, length(d)); %index
P_current = max(action_set)*ones(1,length(d)); %power
old_idx = idx_current;
flag =0;
NE_it =1;
E = E0*ones(1,length(d)); % left energy after consumption
while (flag ==0 && NE_it<30)
    for player = players% 1:length(d)
        H = zeros(1,length(action_set));
        C = zeros(1,length(action_set));
        for action_idx = 1: length(action_set);% change player i's action
            
            D_current(player) = action_d(action_idx); %distance
            % build A matrix (adjacent matrix)
            
            
            TCLE_adj_matrix=zeros(length(d));
            
            for r=players%1:length(d)
                for c=players%1:length(d)
                    TCLE_adj_matrix(r,c)=(d(r,c)<=D_current(r)); % node r에서 c까지의 거리가 player r의 현재 설정 거리보다 작거나 같으면 1
                end
            end
            
            A_sum = sum(TCLE_adj_matrix,2); % number of outgoing links for each player
            Dia = diag(A_sum); % diagonal matrix
            L = Dia-TCLE_adj_matrix;
            
            [~,eig_mat] = eig(L(players,players));
            eigen_value_set = sort(diag(eig_mat));
            H(action_idx) = (eigen_value_set(2)>epsilon); % benefit term
            C(action_idx) = integral(fun,E0-E(player),E0-E(player)+action_set(action_idx))/M; % cost term
            
            
        end
        U_TCLE = H - C;
        opt_act_idx = find(U_TCLE == max(U_TCLE)); % best response
        idx_current(player) = opt_act_idx; % new action index vector
        D_current(player) = action_d(opt_act_idx); %new distance
        P_current(player) = action_set(opt_act_idx); %new power
        
    end
    if sum(old_idx -idx_current)>0
        flag=0; % action change
        old_idx = idx_current;
        NE_it =NE_it+1;
    else
        flag=1; % no action change and NE found
    end
    
end

%E = E - P_current;% left energy

%determined action index: idx_current
% NE network adjacency matrix
TCLE_adj_matrix=zeros(length(d));
TCLE_tx_r = zeros(1,max(players));
for p = [S1_idx,S2_idx]
    rand_num = rand(length(d),1)';
    TCLE_adj_matrix(p,:) = (((d(p,:)<=S_tx_r).*rand_num)>beta); %with beta link failure rate
end


for p=players%1:length(d)
    %for c=players%1:length(d)
    rand_num = rand(length(d),1)';
    TCLE_adj_matrix(p,:) = (((d(p,:)<=D_current(p)).*rand_num)>beta); %with beta link failure rate
    %TCLE_adj_matrix(r,:)=(d(r,:)<=D_current(r)); % node r에서 c까지의 거리가 player r의 현재 설정 거리보다 작거나 같으면 1
    TCLE_tx_r(p)=D_current(p);
end
    



for r=players%1:length(d)
    for c=players%1:length(d)
        TCLE_adj_matrix(r,c)=(d(r,c)<=D_current(r)); % node r에서 c까지의 거리가 player r의 현재 설정 거리보다 작거나 같으면 1
    end
    TCLE_tx_r(r)=D_current(r);
end
TCLE_energy = sum(TCLE_tx_r.^2)/length(players);
end