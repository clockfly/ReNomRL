%%This is main part of DQN algorithm from here we call agent name as actor
%%and we call buffer as memory which is use for data collection and reuse
%%the data for DQN off line learning . and we use epsilon decay for random search
 rng(222) % seed for better performance
actor=dqn_agent;
Memory=buffer;
cart=cart_pole;%cart_pole is instance as cart 
batch_size=32;
global epsilon;
epsilon=1;
global epsilon_min;
epsilon_min=0.01;
global epsilon_decay;
epsilon_decay=0.995;
action=[0,-1,1];% here actions are [no-move left-move right-move]
max_episode=10000; %for maximum iteration
max_it=200; % maxmimum length of cartploe is 200
for se =1:max_episode
    s_t=cart.re_set;
    rewrd=0;
    rewd=0;
    memory=Memory.len;
    if memory>batch_size
        S=Memory.randslc(batch_size); % get data from buffer for offline learning
        for ie=1:batch_size
            reward=S(ie,11);
            done=S(ie,10);
            next_state=S(ie,6:9);
            stte=S(ie,1:4);                
            acton=S(ie,5);
            
            tar=actor.act_on(next_state,0);
            use_tar=actor.t_predict(next_state);
            target=zeros(1,3);
            if done
                target(acton)=reward;
            else
                f=reward+0.95*use_tar(tar);
                target(acton)=f;
            end
        end
        actor.backpropagate(stte,target);
        % here we calculate the epsilon after one episode decay
       if epsilon > epsilon_min
          epsilon= epsilon*epsilon_decay;                
       end 
    end    
    for step =1:max_it %game run for 
        state=s_t;
        a_t = actor.act_on(state,epsilon);% get action index from agent
        at=action(a_t);        
        [N_s,rewd,terminate]=cart.forward(at,se);
        trajectory =[state,a_t,N_s,terminate,rewd];       
        Memory.append(trajectory) 
        rewrd=rewrd+rewd; 
        if terminate==1  % if game is tarminate the reset the game 
            actor.set_target_weight;
            break;
        end
        s_t=N_s; % next state is set to the state
    end
 fprintf('Episode  %d,   reward  %d\n',se,rewrd)
 end
