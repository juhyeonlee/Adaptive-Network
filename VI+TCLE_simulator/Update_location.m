function [d, node_loc] = Update_location (MAX_block, noEvents, block_coords, node_loc, D1_idx, D2_idx,S1_idx,S2_idx, one_dim)

old_node_loc = node_loc;
%noEvents = zeros(1,MAX_block);
coords = [];

% random generation
for block= 1:MAX_block
    %noEvents(block) = poissrnd(mu*4); %?? ?????? ???????? ???? ??, ?????? 2x2?????? 4?? ??????
    coords(1:noEvents(block),1:2,block) = 2*rand(noEvents(block),2); %random location
end

% node location
node_num = 1;
node_loc = zeros(size(old_node_loc));
for b= 1:MAX_block
    for n=1:noEvents(b)
        node_loc(node_num,:) = block_coords(b,:)+coords(n,:,b); % coordination of each node
        node_num = node_num+1;
    end
end

% two source nodes: fixed location
node_loc(S1_idx:S2_idx,:)=[2,2*(sqrt(MAX_block)-1);2*(sqrt(MAX_block)-1),2*(sqrt(MAX_block)-1)];
% source nodes index
% S1_idx = sum(noEvents)+1;
% S2_idx = sum(noEvents)+2;


% two destination nodes: fixed location
% D1_idx=sum(noEvents)+3;
% D2_idx=sum(noEvents)+4;

node_loc(D1_idx:D2_idx,:)=[2,2;  2*(one_dim-1),2];

%distance matrix
d=zeros(size(node_loc,1));
for f=1:size(node_loc,1)
    for t=1:size(node_loc,1)
        d(f,t)=norm([node_loc(f,1),node_loc(f,2)]-[node_loc(t,1),node_loc(t,2)]);
    end
end
%d(:,end-1:end)=inf * ones(size(d,1),2); % source node?? ???????? link?? ???? ?????? inf?? ???? ????

end