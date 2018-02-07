function [connectivity_ratio,goodput] = Get_NW_Performance (adj_matrix,S1_idx,S2_idx,D1_idx, D2_idx)
%foresighted
G=digraph(adj_matrix);

it =1;
dist=zeros(4,1);
for s = [S1_idx,S2_idx]
    for t = [D1_idx,D2_idx]
        [~,dist(it)]= shortestpath(G,s,t);
        %path(it,1:length(path_temp),times)= path_temp;
        it =it+1;
    end
end
connectivity_ratio = sum(dist<Inf)/4;  %number of connected path, max 4( source 2 * destination 2)
goodput = sum(1./dist); % goodput
end