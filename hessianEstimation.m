function[Hessian] = HessianEstimation(r,z_r,r_c,z_c,Dz,SensorN,Hessian)

%%% Calculate z_E and z_J
s = [z_c,Dz].';
Dz = Dz';
%get the centre r_E using sensors centre r1,r3,r4
co_ordinates = [r(:,1) r(:,3) r(:,4)];
x_axis = co_ordinates(1,:);
y_axis = co_ordinates(2,:);
polyin = polyshape(x_axis,y_axis);
[x,y] = centroid(polyin);
r_E = [x,y].';
Hc=[Hessian(:,1).' Hessian(:,2).'].';

%set inital z_E value
z_E_cap = [z_r(:,1) z_r(:,3) z_r(:,4)].';

%calculate C_E and dz_E


C_E(1,:) = [1 (r(:,1) - r_E).'];
C_E(2,:) = [1 (r(:,3) - r_E).'];
C_E(3,:) = [1 (r(:,4) - r_E).'];

% SHould De shuld be transposed or not transposed..?
D_E(1,:) = 0.5 * kron((r(:,1)-r_E),(r(:,1)-r_E)).';
D_E(2,:) = 0.5 * kron((r(:,3)-r_E),(r(:,3)-r_E)).';
D_E(3,:) = 0.5 * kron((r(:,4)-r_E),(r(:,4)-r_E)).';


s_E = inv(C_E)*(z_E_cap - D_E*Hc);


z_E = s_E(1);
dz_E = s_E(2:end);

z_j = z_c;

r_j = r_E + (z_c -z_E) * (dz_E / norm(dz_E));


%%% find dz_j using step 4 not sure if its right or not can be implemented
%%% differently


%D = (r(:,1)-r_j)
%x= size(D)

%term_2 = D.'* 

%dz_j = (z_r(:,1) - z_j-(0.5 * (r(:,1)-r_j).' * Hessian * (r(:,1)-r_j)))/(r(:,2)-r_j);

%z_j_cap = [z_r(:,1) z_r(:,3) z_r(:,4)].';

C_j(1,:) = [1 (r(:,1) - r_j).'];
C_j(2,:) = [1 (r(:,3) - r_j).'];
C_j(3,:) = [1 (r(:,4) - r_j).'];

Dz_j(1,:) = 0.5 * kron((r(:,1)-r_j),(r(:,1)-r_j)).';
Dz_j(2,:) = 0.5 * kron((r(:,3)-r_j),(r(:,3)-r_j)).';
Dz_j(3,:) = 0.5 * kron((r(:,4)-r_j),(r(:,4)-r_j)).';

s_j = inv(C_j)*(z_j - Dz_j*Hc);

%z_j = s_j(1);
dz_j = s_j(2:end);



%%% Calculate z_F and z_K

%get the centre r_F using sensors centre r2,r3, r4
co_ordinates = [r(:,2) r(:,3) r(:,4)];
x_axis = co_ordinates(1,:);
y_axis = co_ordinates(2,:);
polyin = polyshape(x_axis,y_axis);
[x,y] = centroid(polyin);
r_F = [x,y].';

%set inital z_F value
z_F = [z_r(:,2) z_r(:,3) z_r(:,4)].';

%calculate C_F and dz_F
C_F(1,:) = [1 (r(:,2) - r_F).'];
C_F(2,:) = [1 (r(:,3) - r_F).'];
C_F(3,:) = [1 (r(:,4) - r_F).'];

D_F(1,:) = 0.5 * kron((r(:,2)-r_F),(r(:,2)-r_F)).';
D_F(2,:) = 0.5 * kron((r(:,3)-r_F),(r(:,3)-r_F)).';
D_F(3,:) = 0.5 * kron((r(:,4)-r_F),(r(:,4)-r_F)).';

s_F = inv(C_F)*(z_F - D_F*Hc);

z_F = s_F(1);
dz_F = s_F(2:end);
z_k = z_c;

r_k = r_F + (z_c -z_F) * (dz_F / norm(dz_F));

%%% find dz_k using step 4 not sure if its right or not can be implemented
%%% differently

%dz_k = (z_r(:,2) - z_k-(0.5 * (r(:,2)-r_k).' * Hessian * (r(:,2)-r_k)))*inv(r(:,2)-r_k);


%z_k_cap = [z_r(:,2) z_r(:,3) z_r(:,4)].';

C_k(1,:) = [1 (r(:,2) - r_k).'];
C_k(2,:) = [1 (r(:,3) - r_k).'];
C_k(3,:) = [1 (r(:,4) - r_k).'];

Dz_k(1,:) = 0.5 * kron((r(:,2)-r_k),(r(:,2)-r_k)).';
Dz_k(2,:) = 0.5 * kron((r(:,3)-r_k),(r(:,3)-r_k)).';
Dz_k(3,:) = 0.5 * kron((r(:,4)-r_k),(r(:,4)-r_k)).';

s_k = inv(C_k)*(z_k - Dz_k*Hc);
dz_k = s_k(2:end);

% y_1j is the unit vetor along the direction of gradient dz_j
% y_1k is the unit vetor along the direction of gradient dz_k
% y_1c is the unit vetor along the direction of gradient dz_c

y_1j = dz_j / norm(dz_j);
y_1k = dz_k / norm(dz_k);
y_1c = Dz / norm(Dz);

delta_theta_l = acos(dot(y_1j,y_1c));
delta_s_l = norm(r_j - r_c);

delta_theta_r = acos(dot(y_1k,y_1c));
delta_s_r = norm(r_k - r_c);

kappa_c = 0.5 * ((delta_theta_l / delta_s_l) + (delta_theta_r / delta_s_r));

Hxx = -norm(Dz)* kappa_c;
Hxy = (norm(dz_k) - norm(dz_j))/(delta_s_l + delta_s_r);

%r_x = r(:,4)-r_c;
%Hyy = 2*(z_r(:,4) - z_c - dot(Dz,r_x) - 0.5*((Hxx * r_x(1) * r_x(1)) + (2 * Hxy * r_x(1) * r_x(2))))/( r_x(2) * r_x(2));
r_x = r(:,3)-r_c;
Hyy = 2*(z_r(:,3) - z_c - dot(Dz,r_x) - 0.5*((Hxx * r_x(1) * r_x(1)) + (2 * Hxy * r_x(1) * r_x(2))))/( r_x(2) * r_x(2));

%Hyy = (Hyy1 + Hyy2)/2;

Hessian = [Hxx, Hxy;
            Hxy, Hyy];



