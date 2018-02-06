
function [adj_matrix,tx_r, energy] = ValueIteration(players, d, beta, final_PI, tx_r, a_range, max_tx_r, len_state,S1_idx,S2_idx, S_tx_r)

last_tx_r = tx_r;
tx_r = zeros(1,max(players));
% initial state for all players: the same
s_fore = zeros(1,max(players));

%foresighted
a_fore = zeros(1,max(players));
a_idx_fore= zeros(1,max(players));
s_next_fore= zeros(1,max(players));

adj_matrix=zeros(length(d));

for p = [S1_idx,S2_idx]
    rand_num = rand(length(d),1)';
    adj_matrix(p,:) = (((d(p,:)<=S_tx_r).*rand_num)>beta); %with beta link failure rate
end

for p =players
    % foresighted
    rand_num = rand(length(d),1)';
    
    
    s_fore(p) = min(len_state,sum(d(p,:)<= last_tx_r(p))); %s_next: smaller than len_state
    
    a_fore(p) = final_PI(s_fore(p)); %action
    
    a_idx_fore(p) = find(a_range == a_fore(p)); %action index
    % new definiton of action for ICML2018
    tx_r(p) = min(max_tx_r, max(0, last_tx_r(p)+ a_fore(p)));
%    tx_r(p)= sqrt(max(0,last_tx_r(p)^2 + a_fore(p)/pi)); %tx radius after action: bigger than 0
    s_next_fore(p) = min(len_state,sum(d(p,:)<= tx_r(p))); %s_next
    
    adj_matrix(p,:) = (((d(p,:)<=tx_r(p)).*rand_num)>beta); %with beta link failure rate
end



energy = sum(tx_r.^2)/length(players);
end