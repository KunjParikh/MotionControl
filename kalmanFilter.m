function [z_c,Dz,P] = kalmanFilter(z_c,Dz,r,z_r,r_c,r_c_old,P,Hessian,SensorN)
%SensorN = SensorN - 1;%why minus 1?

 s = [z_c,Dz].';
M = 0.001 * eye(3); R = 0.001 * eye(SensorN); 
 U = 0.001 * eye(4);%TODO: change size of U 
A = [1 (r_c - r_c_old).';zeros(2,1) eye(2)]; %A_k_s
h = [0;Hessian * (r_c - r_c_old)]; %h(k-1)
 %P = 0.0001 * zeros(3)
 
Hc=[Hessian(:,1).' Hessian(:,2).'].';

for i = 1:SensorN
    C(i,:) = [1 (r(:,i) - r_c).'];
    D(i,:) = 0.5 * kron((r(:,i)-r_c),(r(:,i)-r_c));
end


    s_e = A * s + h;
    P_e = A * P * A.' + M;
    K = P_e * C.' *  inv(C*P_e*C.'+ D*U*D.' + R);
    
    s = s_e + K * (z_r(1:SensorN).' - C*s_e - D*Hc);

    P = inv(inv(P_e) + C.' * inv(D*U*D.' + R) * C); % why P is needed since we already have s ?
    %P = (eye(3) - K*C) * P_e;
z_c = s(1); Dz = s(2:end)';
z_c = double(z_c);
Dz = double(Dz);

