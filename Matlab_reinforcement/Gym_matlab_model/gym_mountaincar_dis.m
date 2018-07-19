classdef gym_mountaincar_dis <handle
    properties(SetAccess=private)
        mini_action=-1.0;
        max_action= 1.0;
        mini_position=-1.2;
        max_position=0.6;
        max_speed=0.07;
        goal_position=0.45;
        low_state=[-1.2,-0.07];
        high_state=[0.6,0.07];
        defult=[random('Uniform',-0.6,-0.4),0];
        state=[random('Uniform',-0.6,-0.4),0];
        maxite=800;
        count=0;
    end   
    methods
        function [next,done,reward]= forward(self,action)
            if self.count==self.maxite
                done=1;
                self.count=0;
                self.state=self.defult;
            else 
                done=0;
            end
            self.count=self.count+1;
            position=self.state(1);
            velocity=self.state(2);
            N_velocity=velocity+(action)*0.001-0.0025*cos(3*position);
            if N_velocity<-self.max_speed
                N_velocity=-self.max_speed;
            elseif N_velocity>self.max_speed
                N_velocity=self.max_speed;
            else
                N_velocity=N_velocity;
            end
            N_position=position+N_velocity;
            if N_position<self.mini_position
                N_position=self.mini_position;
            elseif N_position>self.max_position
                N_position=self.max_position;
            else
                 N_position= N_position;
            end 
            if (N_position==self.mini_position && N_velocity<=0)
                N_velocity=0;
            end
            if position>=self.goal_position
                reward=1;
            else
                reward=-1;
            end     
           
            next=[N_position,N_velocity]; 
            self.state=next;
            
            self.MountainCarPlot(action)
        end
         function init_state=re_set(self)
            init_state=[random('Uniform',-0.6,-0.4),0];
            self.state=init_state;
        end
        function MountainCarPlot(self,acton)
            plot(1);
            x=self.state;        
            set(gco,'BackingStore','off')  % for realtime inverse kinematics
            set(gco,'Units','data')
            xplot =-1.6:0.05:0.6;
            yplot =sin(3*xplot);
            %Mountain
            h = area(xplot,yplot,-1.1);   
            set(h,'FaceColor',[.1 .8 .25])
            hold on
            % Car  [1 .7 .1]
            plot([x(1)-0.075 x(1)+0.075] ,[sin(3*(x(1)-0.075))+0.1  sin(3*(x(1)+0.075))+0.1 ],'-','LineWidth',10,'Color',[1 .7 .7]);
            % wheels
            plot(x(1)-0.05,sin(3*(x(1)-0.05))+0.06,'ok','markersize',12,'MarkerFaceColor',[.5 .1 .5]);
            plot(x(1)+0.05,sin(3*(x(1)+0.05))+0.06,'ok','markersize',12,'MarkerFaceColor',[.5 .1 .5]);
            %Goal
            plot(0.45,sin(3*0.5)+0.1,'-pk','markersize',15,'MarkerFaceColor',[1 .7 .1]);
            % direction of the force
            if (acton<0)
                  plot(x(1)-0.08-0.05,sin(3*(x(1)-0.05))+0.2,'<k','MarkerFaceColor','g','markersize',10);
            elseif (acton>0)
                  plot(x(1)+0.08+0.05,sin(3*(x(1)+0.05))+0.2,'>k','MarkerFaceColor','g','markersize',10);
            end
            title(strcat ('Step: ',int2str(self.count)));
            %-----------------------
            axis([-1.6 0.6 -1.1 1.5]);
            drawnow
            hold off
        end
    end
end

        
