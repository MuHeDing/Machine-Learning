% x=load('F:\\Machine Learning\\exp\\ex5Data\\ex5Linx.dat'); % 7*1
% y=load('F:\\Machine Learning\\exp\\ex5Data\\ex5Liny.dat');
% plot(x,y, 'o', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
% hold on;
% m=size(x,1);
% origin=x;
% x = [ones(m,1),x,x.^2,x.^3,x.^4,x.^5 ] ; % 7*6
% 
% h=@(x,theta) x*theta;
% 
% n=size(x,2);
% I=eye(n);
% I(1,1)=0;
% lamda=10;
% disp(lamda*I)
% u=(x'*x+lamda*I)\x'*y
% 
% y=h(x,u)
% plot(origin,y,'b--');




x = load('F:\\Machine Learning\\exp\\ex5Data\\ex5Logx.dat'); % 117*2
y = load('F:\\Machine Learning\\exp\\ex5Data\\ex5Logy.dat'); %117*1
figure
[m,n]=size(x);  % 117*2

u=size(y);   %117*1
pos=find(y==1);  % 1-58
neg = find(y==0);  % 59-117
 %plot(x(pos,1),x(pos,2),'+')
plot(x(pos, 1), x(pos, 2), 'k+','LineWidth', 2,'MarkerSize', 7);
hold on
plot(x(neg, 1), x(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);

iteration=20;



x=map_feature(x(:,1),x(:,2));
[p,q]=size(x);  % 117*28

lamda=10;
h=@(x,theta) 1.0 ./(1.0+exp(-x*theta)); % 117*1
loss=@(theta,y,x,theta_reg)  (x'*(h(x,theta)-y)+lamda*theta_reg)/m;
I=eye(q);
I(1,1)=0;
J=zeros(iteration,1);
theta=zeros(q,1);

for i=1:iteration
    theta_reg=[0;theta(2:q)];
    hessian=(x'*diag(h(x,theta))*diag(1-h(x,theta))*x+lamda*I)/m;
    % sum(theta_reg .^ 2)
    % J(i)=(-y'*log(h(x,theta))-(1-y')*log(1-h(x,theta)))/m+(lamda*theta'*theta)/(2*m); 
    J(i)=(-y'*log(h(x,theta))-(1-y')*log(1-h(x,theta)))/m+(lamda*sum(theta_reg.^2))/(2*m); 
    gradient=loss(theta,y,x,theta_reg);
    theta=theta-hessian\gradient;    
end
theta
   
u = linspace(-1,1.5,200);
v = linspace(-1,1.5,200);

z = zeros(length(u),length(v));

for i=1:length(u)
for j=1:length(v)
z(j,i)=map_feature(u(i),v(j))*theta;
end
end

contour(u,v,z',[0,0],'LineWidth',2);
legend('y=1', 'y=0','Decision Boundary')
figure;
plot(0:iteration-1, J, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
xlabel('Iteration'); ylabel('J')

function out = map_feature(feat1, feat2)
% MAP_FEATURE    Feature mapping function for Exercise 4
%
%   map_feature(feat1, feat2) maps the two input features
%   to higher-order features as defined in Exercise 4.
%
%   Returns a new feature array with more features
%
%   Inputs feat1, feat2 must be the same size
%
% Note: this function is only valid for Ex 4, since the degree is
% hard-coded in.
    degree = 6;
    out = ones(size(feat1(:,1)));
    for i = 1:degree
        for j = 0:i
            out(:, end+1) = (feat1.^(i-j)).*(feat2.^j);
        end
    end
end








