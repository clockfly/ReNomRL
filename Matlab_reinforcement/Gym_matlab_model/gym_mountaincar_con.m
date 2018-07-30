classdef gym_mountaincar_con <handle
    properties(SetAccess=private)
        mini_action=-1.0;
        max_action= 1.0;
        mini_position=-1.6;
        max_position=0.6;
        max_speed=0.07;
        goal_position=0.5;
        power=0.060;
        low_state=[-1.2,-0.07];
        high_state=[0.6,0.07];
        state=[-0.5,0]
        maxite=200;
        count=0;
    end   
    methods
        function [next,done,reward]= forward(self,st,action)
            if self.count==self.maxite
                self.count=0;
            end
            self.state=st;
            self.count=self.count+1;
            position=self.state(1);
            velocity=self.state(2);
            force=min(max(action(1),-1.0),1.0);
            velocity=velocity+force*self.power-0.075*cos(3*position);
            if velocity>self.max_speed
                velocity=self.max_speed;
            end
            if velocity<-self.max_speed
                velocity=-self.max_speed;
            end
            position=position+velocity;
            if position>self.max_position
                position=self.max_position;
            end
            if position<self.mini_position
                position=self.mini_position;
            end
            
            if position>=self.goal_position
                done=1;
            else
                done=0;
            end     
            reward=(self.state(1)+self.state(2)-0.57);             
           
            next=[position,velocity]; 
            self.state=next;
            
            self.MountainCarPlot(action)
        end
        function MountainCarPlot(self,acton)
            subplot(2,1,2);
            x=self.state;        
            set(gco,'BackingStore','off')  % for realtime inverse kinematics
            set(gco,'Units','data')
            xplot =-1.6:0.05:0.6;
            yplot =sin(3*xplot);
            %Mountain
            h = area(xplot,yplot,-1.1);   
            set(h,'FaceColor',[.1 .7 .1])
            hold on
            % Car  [1 .7 .1]
            plot([x(1)-0.075 x(1)+0.075] ,[sin(3*(x(1)-0.075))+0.2  sin(3*(x(1)+0.075))+0.2 ],'-','LineWidth',10,'Color',[1 .7 .1]);
            % wheels
            plot(x(1)-0.05,sin(3*(x(1)-0.05))+0.06,'ok','markersize',12,'MarkerFaceColor',[.5 .5 .5]);
            plot(x(1)+0.05,sin(3*(x(1)+0.05))+0.06,'ok','markersize',12,'MarkerFaceColor',[.5 .5 .5]);

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

        
