% adaptive network - network system part
% 2017-11-13
% written by minhae kwon

%%%%%%%%%%%
% 1. parameter setting
% 2. initial network formation
%   - each block size is 2 by 2
%   - consider (5 by 5) blocks for one network
%   - [ x x x x x ]
%   - [ x 7 8 9 x ] 
%   - [ x 4 5 6 x ]
%   - [ x 1 2 3 x ]
%   - [ x x x x x ]
%   - only 3 by 3 blocks are used for data delivery
%   - 2 source nodes (at block 7 & 9) and 2 destination nodes (at block 1 & 3)-fixed location
%   - number of nodes per block: based on poisson point process
%   - location of nodes: random generation inside of each block
% 3. evaluate network performance
%   - connectivity ratio
%   - goodput
%%%%%%%%%%%

%% 1. parameter setting
clear all;
close all;

%paramters
mu = 4/5; % node density
one_dim = 5;
MAX_block = one_dim^2;
beta = 0.1; % link failure rate 
tx_r = 2; % default radius of tx range
%%%%%%


%% 2. initial nework formation

noEvents = zeros(1,MAX_block);
coords = [];

%  default coordination for each block
block_coords = zeros(MAX_block,2);
default = 0:2:2*(one_dim-1);
temp = 1;
for c2 = 1:length(default)
    for c1 = 1:length(default)
        block_coords(temp,:) = [default(c1), default(c2)];
        temp = temp+1;
    end
end


% random generation
for block= 1:MAX_block
    noEvents(block) = poissrnd(mu*4); %number of nodes per block: based on poisson point process
    coords(1:noEvents(block),1:2,block) = 2*rand(noEvents(block),2); %random location
end

%number of players
players =[];
for t = 1:one_dim-2 % the end blocks at each side are not considered
    players = [players,sum(noEvents(1:t*one_dim+1))+1:sum(noEvents(1:(t+1)*one_dim-1)) ];
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
node_loc(sum(noEvents)+1:sum(noEvents)+2,:)=[2, 2*(one_dim-1); 2*(one_dim-1), 2*(one_dim-1) ];

% two destination nodes: fixed location
node_loc(sum(noEvents)+3:sum(noEvents)+4,:)=[2,2;  2*(one_dim-1),2];


%distance matrix
d=zeros(size(node_loc,1));
for f=1:size(node_loc,1)
    for t=1:size(node_loc,1)
        d(f,t)=norm([node_loc(f,1),node_loc(f,2)]-[node_loc(t,1),node_loc(t,2)]);
    end
end
%d(:,end-1:end)=inf * ones(size(d,1),2); 

% adjacent matrix 
adj_matrix = zeros(size(d));
for p = [players, size(node_loc,1)-3:1:size(node_loc,1)] % players + two sources +two destinations
adj_matrix(p,:) = (((d(p,:)<=tx_r).*rand(1))>beta); %with beta link failure rate
end

%% 3. evaluate network performance

G=digraph(adj_matrix);
g = plot(G); % just for visualization
g.XData = node_loc(:,1);
g.YData = node_loc(:,2);
highlight(g,[sum(noEvents)+1:1:sum(noEvents)+4],'NodeColor','r'); % red coloring for source and destinationo nodes
grid on;

it =1;
dist=zeros(4,1);
for s = sum(noEvents)+1:sum(noEvents)+2 % sources
    for t = sum(noEvents)+3:sum(noEvents)+4 % destinations
        [path_temp,dist(it)]= shortestpath(G,s,t);
        %path(it,1:length(path_temp),times)= path_temp;
        it =it+1;
    end
end


connectivity_ratio = sum(dist<Inf)/4  %number of connected path, max 4( source 2 * destination 2)
goodput = sum(1./dist) % goodput

