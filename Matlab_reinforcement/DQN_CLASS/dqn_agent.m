classdef dqn_agent <handle 
    properties(SetAccess=private)
        a_w={rand( 4,20),rand(20,3)};% weight initialization
        grads = {}; % for gradient
        learning_rate=0.001; 
        randstream
        seed224
    end   
   methods 
        function y_pre =predict(self,X) % for action prediction
            W1 = self.a_w{1};
            W2 = self.a_w{2};
            z_1=(X*W1); 
            H_1=tanh(z_1);
            z_2=(H_1*W2);
            y_pre=z_2;
        end
        function act=act_on(self,sta,epsilon)          
            if randn()<= epsilon
                act=randi([1,3],1,1); % random action index 
            else
                inn=self.predict(sta);
               [~,act]=max(inn); % action is max of pridiction
            end
        end
        function backpropagate(self, X, y_tgt) % forback propagation
            w1= self.a_w{1};
            w2= self.a_w{2};
            z{1}= X;           
            z{2}= tanh(z{1}*w1);           
            z{3} =  z{2}*w2; 
            error = (z{3}-y_tgt);
            clip_error=zeros(size(error),'like',error);
            [x,y]=size(error);
            for l=1:x 
                for m=1:y
                    if error(l,m)>0.9
                        clip_error(l,m)=1;
                    elseif error(l,m)<-1
                         clip_error(l,m)=-1;
                     else
                         clip_error(l,m)=error(l,m);
                    
                    end
                end
            end
                        
            grad1= (z{2}'*clip_error);
            e_h12= w2*clip_error';
            dm2= (z{2}-z{2}.^2).*e_h12';
            grad2=(z{1}'*dm2);
            self.a_w{1}=w1-(grad2*self.learning_rate);
            self.a_w{2}=w2-(grad1*self.learning_rate);
            self.a_w={self.a_w{1},self.a_w{2}};
        end
            
    end
    
end

