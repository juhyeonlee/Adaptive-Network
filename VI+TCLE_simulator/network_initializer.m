
function [len_players, players, D1_idx, D2_idx,S1_idx,S2_idx, d,  tx_r, noEvents, block_coords, node_loc] = network_initializer(MAX_block, mu, one_dim, S_tx_r, beta)

%% for function test only
% clear all;
% close all;
% MAX_block =25;
% mu = 0.8;
% one_dim = 5;
% S_tx_r = 2;
% beta = 0;
% %%%

%% default coordination for each block
block_coords = zeros(MAX_block,2);
default = 0:2:2*(sqrt(MAX_block)-1);
temp = 1;
for c2 = 1:length(default)
    for c1 = 1:length(default)
        block_coords(temp,:) = [default(c1), default(c2)];
        temp = temp+1;
    end
end

%% 2.node location

noEvents = zeros(1,MAX_block);
coords = [];

% random generation
for block= 1:MAX_block
    noEvents(block) = poissrnd(mu*4); %?? ?????? ???????? ???? ??, ?????? 2x2?????? 4?? ??????
    coords(1:noEvents(block),1:2,block) = 2*rand(noEvents(block),2); %random location
end

%number of players
players =[];
for t = 1:sqrt(MAX_block)-2 % minus 2?? ????: ?? ?? row???? + ?? ???? row ????
    players = [players,sum(noEvents(1:t*sqrt(MAX_block)+1))+1:sum(noEvents(1:(t+1)*sqrt(MAX_block)-1)) ];
end

len_players = length(players);
% node location
node_num = 1;
node_loc=[];
for b= 1:MAX_block
    for n=1:noEvents(b)
        node_loc(node_num,:) = block_coords(b,:)+coords(n,:,b); % coordination of each node
        node_num = node_num+1;
    end
end

% two source nodes: fixed location
node_loc(sum(noEvents)+1:sum(noEvents)+2,:)=[2,2*(sqrt(MAX_block)-1);2*(sqrt(MAX_block)-1),2*(sqrt(MAX_block)-1)];
% source nodes index
S1_idx = sum(noEvents)+1;
S2_idx = sum(noEvents)+2;


% two destination nodes: fixed location
D1_idx=sum(noEvents)+3;
D2_idx=sum(noEvents)+4;

node_loc(D1_idx:D2_idx,:)=[2,2;  2*(one_dim-1),2];

% % two destination nodes (block 7, 9): random selection
% if noEvents(sqrt(MAX_block)+2) >0
%     D1_idx= sum(noEvents(1:sqrt(MAX_block)+1))+randi(noEvents(sqrt(MAX_block)+2),1,1);
% else D1_idx= sum(noEvents(1:sqrt(MAX_block)+2))+1;
% end
% 
% if noEvents(2*sqrt(MAX_block)-1) > 0
%     D2_idx= sum(noEvents(1:2*sqrt(MAX_block)-2))+randi(noEvents(2*sqrt(MAX_block)-1),1,1);
% else D2_idx= sum(noEvents(1:2*sqrt(MAX_block)-1))-1;
% end

%distance matrix
d=zeros(size(node_loc,1));
for f=1:size(node_loc,1)
    for t=1:size(node_loc,1)
        d(f,t)=norm([node_loc(f,1),node_loc(f,2)]-[node_loc(t,1),node_loc(t,2)]);
    end
end
%d(:,end-1:end)=inf * ones(size(d,1),2); % source node?? ???????? link?? ???? ?????? inf?? ???? ????

%adjacent matrix
adj_matrix = zeros(size(d));
tx_r = zeros(1,max(players));
% for source connection, other nodes: init_txr=0
rand_num = rand(length(d),1)';
for p = [S1_idx,S2_idx]
    adj_matrix(p,:) = (((d(p,:)<=S_tx_r).*rand_num)>beta); %with beta link failure rate
    %tx_r(p)=S_tx_r;
end
end