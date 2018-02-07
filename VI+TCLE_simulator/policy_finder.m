

function [final_PI,ini_state] = policy_finder(len_state, a_range, u, R_weight, w, mu, rho, e)

P = zeros(len_state, len_state, length(a_range));
U = zeros(len_state, len_state, length(a_range));

for s = 1:len_state
    for a_idx = 1:length(a_range)
        a = a_range(a_idx);
        for s_next = 1:len_state
            U(s,s_next,a_idx) = u+R_weight *(sqrt(s_next)-sqrt(s))*w - (1-w)*a;
        end
        
        if a<0
            for s_next = 1:s
                K =s; % ceil(s/(1-beta)); %really located number of nodes
                K_next = s_next; % ceil(s_next/(1-beta)); %really located number of nodes
                
                n_C_r=factorial(K)/factorial(K_next)/factorial(K-K_next);
                p = (pi *abs(a))/(s/mu);
                if p<1
                    P(s, s_next, a_idx)=n_C_r*(1-p)^K_next*p^(K-K_next);
                end
            end
            
        elseif a == 0
            s_next = s;
            P(s, s_next, a_idx) =1;
            
        else % a>0
            for s_next = s:len_state
                K = s_next - s; %ceil(abs(s_next-s)/(1-beta)); % with beta link failure rate
                %next state
                lambda = mu * pi * abs(a) ; %increasing area
                P(s, s_next, a_idx) = lambda^K * exp(-lambda)/factorial(K);
            end
        end
        
    end
end


V=zeros(len_state,1); % V matrix
PI=zeros(len_state,1); % Policy Index? matrix

it = 1;
set_e =0;
while (set_e < len_state)
    V_temp=zeros(len_state,length(a_range));%  V matrix
    for s=1:len_state %state
        for a_idx = 1:length(a_range)
            a = a_range(a_idx);
            for s_next = 1: len_state
                %for all action
                V_temp(s,a_idx) =V_temp(s,a_idx)+( P(s,s_next,a_idx)*(U(s,s_next,a_idx) + rho*V(s_next,it)));
            end
        end
        V(s,it+1)=max(V_temp(s,:));
        PI(s,it+1)=min(find(V_temp(s,:)==max(V_temp(s,:))));
    end
    set_e = sum(sum( V(:,it+1) - V(:,it) < (1-rho)./2./rho.*e.*ones(len_state,1,1)));
    it = it +1;
end
final_PI =  a_range(PI(:,end));%-ceil(length(a_range)/2);%foresighted

ini_state = find(final_PI == 0);
end