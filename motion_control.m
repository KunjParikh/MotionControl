clear all


SensorN = 4; ShapeIndex = 2;
K1 = 10; K2 = 6000; K3 = 20; K4 = 40; mu = 30; mu_f = 10;  Dt = 0.01; Step = 10000;  % rhombus

hfig=(figure);
[x,y]=meshgrid(-10:.2:10,-40:.2:40);
%f=@(x,y,z)(x.^2+1/4* y.^2+z.^2+ 0.0005 * randn(1));
%f=@(x,y,z)(sqrt((x.^2)+ (y.^2)));   % no field noise
f=@(x,y)(sqrt(x^2+3*y^2+(4*y)-x-(2*x*y)));   % no field noise

z_desired =4;
X_centre = 0;
Y_centre = 0;
ang=0:0.01:2*pi;
%xp=z_desired*cos(ang);
%yp=z_desired*sin(ang);
%plot(X_centre+xp,Y_centre+yp);

axis ([-10 10 -10 10]);
hold on;
ezplot('x^2+3*y^2+(4*y)-x-(2*x*y)=16');
xlabel('x');ylabel('y');
grid on;

%Dz_f = @(x,y)[(x)/sqrt(x.^2+y.^2),(y)/sqrt(x.^2+y.^2)];
Dz_f = @(x,y)[(2*x-1-2*y)/sqrt(x^2+3*y^2+(4*y)-x-(2*x*y)),...
    (6*y-4-2*x)/sqrt(x^2+3*y^2+(4*y)-x-(2*x*y))];

r_c = [-2,-6].';
z_c = f(r_c(1),r_c(2));

r(:,1) =r_c + [1,0].';
r(:,2) =r_c + [-1,0].';
r(:,3) =r_c + [0,-1].';
r(:,4) =r_c + [0,1].';

        Phi =  [1/2 -1/sqrt(2) 0 -1/2 ;
                1/2 1/sqrt(2) 0 -1/2;
                0 0  1/sqrt(2) 1/2 ;
                0 0  -1/sqrt(2) 1/2];
        Phi = Phi.';
        Phi_inv = inv(Phi);
disp(Phi)
disp(Phi(1,:))
disp(r)
for j = 1:SensorN               % initialize Jacobi vectors
    d_q(:,j) = [0;0];
    q(:,j) = Phi(j,:) * r.';
    u_r(:,j) = [0;0];

    Vel_q(:,j) = [0;0];

end
   plot([r(1,1) r(1,3)],[r(2,1) r(2,3)],'y','LineWidth',1.5);
   plot([r(1,2) r(1,4)],[r(2,2) r(2,4)],'y','LineWidth',1.5);

%TODo add phi for jacobi transformaation

r_c_old = r_c;
Dz = Dz_f(r_c(1),r_c(2));
z_c = f(r_c(1),r_c(2));
a = 0.6;
b = 0.6;
Hessian = [2 0; 0 0.5];
Velocity_c = 1;

%%%%%%%%%%%%%%  Main Loop  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = 0.0001 * zeros(3);
x_2 = [1,0].';
y_2 = [0,1].';
%r_obs = [5.8,1.8].';
%r_obs(:,1) = [4.94,-8.69].';
%r_obs(:,1) = [-8.78,-4.78].';
%r_obs(:,2) = [8.628,5.05].';
r_obs(:,1) = [-5.278,-2.011].';
r_obs(:,2) = [4.021,1.95].';
%x_2 = [0,0].';
%y_2 = [0,0].';
for i = 1:2
plot(r_obs(1,i),r_obs(2,i),'o','color','g','MarkerSize',10,'MarkerFaceColor','g'); hold on;
end


for i = 2:Step

    for j = 1:SensorN
    %z_r(j) = f(r(1,j),r(2,j),r(3,j))+0.01*randn(1)*f(r(1,j),r(2,j),r(3,j));   %%%% measurements at each step
          z_r(j) = f(r(1,j),r(2,j));
          %+0.002*randn(1)*f(r(1,j),r(2,j));   %%%% measurements at each step
    end

    %%% Should I update z_c ?

     [z_c,Dz,P] = kalmanFilter(z_c,Dz,r,z_r,r_c,r_c_old,P,Hessian,SensorN);


     %[Hessian] = HessianEstimation(r,z_r,r_c,z_c,Dz,SensorN,Hessian);
     [Hessian] = [2 0;2 0] * rand(1);
     %[Hessian] = [2 -2;-2 6] * rand(1);

     r_c_old = r_c;

     D_z_1 = Dz';
    %%%%%%% New Motion Control %%%%%%%%%%%%%%%%

     y_1 = (D_z_1 / norm(D_z_1));
     x_1 = [cos(pi/2) -sin(pi/2);
             sin(pi/2) cos(pi/2)]*y_1;

     if i == 2
        x_2 = x_1;
        y_2 = y_1;
     end

     %angle = atan2(y_2(2), y_2(1)) - atan2(x_2(2), x_2(1));
     %plot(x_1(1)+r_c(1),x_1(2)+r_c(2),'x','color','r','MarkerSize',4,'MarkerFaceColor','k'); hold on;
     %plot(y_1(1)+r_c(1),y_1(2)+r_c(2),'x','color','g','MarkerSize',4,'MarkerFaceColor','k'); hold on;

     %plot(x_2(1)+r_c(1),x_2(2)+r_c(2),'x','color','b','MarkerSize',4,'MarkerFaceColor','k'); hold on;
     %plot(y_2(1)+r_c(1),y_2(2)+r_c(2),'x','color','y','MarkerSize',4,'MarkerFaceColor','k'); hold on;


     theta = atan2(x_2(2), x_2(1)) - atan2(x_1(2), x_1(1));




     kappa_1 = (x_1.' * Hessian * x_1) / norm(D_z_1);
     kappa_2 = (x_1.' * Hessian * y_1) / norm(D_z_1);

     f_z = mu_f * (1 - (z_desired / z_c).^2);
     norm_val = norm(r_c-r_obs)
     f_rep = 0;
     for k = 1:2
         %disp(i)
         f_rep =  f_rep + 3*log(norm(r_c-r_obs(:,k))/15);
     end

     u_c = kappa_1 * cos(theta) + kappa_2 * sin(theta) - (2 * f_z * ...
     norm(D_z_1) * cos(theta/2).^2) + K4 * sin(theta/2) +f_rep ;
     %v_c = -K1*(Velocity_c-1);;
     if isnan(u_c) == 1
        disp(u_c)
    end
     %y_2_temp = y_2 - Dt .*  u_c *x_2
     %x_2_temp = x_2 + Dt .*  u_c *y_2

     %%x_2 = x_2_temp;
     %y_2 = y_2_temp;
     x_2 = x_2 + Dt .* (u_c * y_2);

     x_2 = x_2/norm(x_2);
     y_2 = [cos(-pi/2) -sin(-pi/2);
             sin(-pi/2) cos(-pi/2)]*x_2;



     %Velocity_c = x_2 + Dt .* (u_c .* y_2);
     %Velocity_c = Velocity_c / norm(Velocity_c);


     r_c = r_c + Dt .* x_2;



    %%%%%% Formation Control %%%%%%%%%%%%
    e_1 = (r(:,2) - r(:,1));
    e_1 = e_1/norm(e_1);
    e_2 = (r(:,3) - r(:,4));
    e_2 = e_2/norm(e_2);


    q_0(:,2) = (a / sqrt(2)) .* e_1;
    q_0(:,3) = (b / sqrt(2)) .* e_2;
    q_0(:,4) = 0;


    for j = 2:SensorN
        d_q(:,j) = d_q(:,j) + Dt * u_r(:,j);
        u_r(:,j) = -K2 * (q(:,j) - q_0(:,j)) - K3 * d_q(:,j);
    end

    for j = 2:SensorN
        Vel_q(:,j) = Vel_q(:,j) + Dt * u_r(:,j);
        q(:,j) = q(:,j) + Dt * Vel_q(:,j);
    end

    q_N = r_c;
    for j = 2:SensorN
        q_N = [q_N q(:,j)];
    end

    for j = 1:SensorN
        r(:,j) = Phi_inv(j,:) * q_N.';  %% inverse Jacobi transform
    end



    %%%%%%%%  plot the formation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        plot(r_c(1),r_c(2),'o','color','k','MarkerSize',2,'MarkerFaceColor','k'); hold on;
     if mod(i,150)==0

        xx23 = r(1,1:4); yy23 = r(2,1:4);
        %xx456 = r(1,4:6); yy456 = r(2,4:6); z
        %plot(r(1,1),r(2,1),'o','color','b','MarkerSize',8,'MarkerFaceColor','b');
        plot(xx23,yy23,'o','color','r','MarkerSize',8,'MarkerFaceColor','r');
        %plot(xx456,yy456,zz456,'o','color','m','MarkerSize',8,'MarkerFaceColor','m');


        plot([r(1,1) r(1,3)],[r(2,1) r(2,3)],'y','LineWidth',1.5);
        plot([r(1,2) r(1,3)],[r(2,2) r(2,3)],'y','LineWidth',1.5);
        plot([r(1,2) r(1,4)],[r(2,2) r(2,4)],'y','LineWidth',1.5);
        plot([r(1,1) r(1,4)],[r(2,1) r(2,4)],'y','LineWidth',1.5);

     end

end



