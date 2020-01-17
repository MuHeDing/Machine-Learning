x=load("F:\\Machine Learning\\exp\\ex4Data\\ex4Data\\ex4x.dat");
y=load("F:\\Machine Learning\\exp\\ex4Data\\ex4Data\\ex4y.dat");
pos = find ( y == 1 );  % 1µΩ40
neg = find ( y == 0 );  % 40µΩ80
m=size(x,1);
x=[ones(m,1),x];
plot ( x ( pos , 2 ) , x ( pos , 3 ),'r+') ; hold on
plot ( x ( neg , 2 ) , x ( neg , 3 ) ,'bo')
xlabel('Exam1'); ylabel('Exam2')
legend('Admitted', 'Not admitted')
iteration=10;


theta=zeros(3,1);
J=zeros(iteration,1);

z=@(x,theta) 1.0 ./(1.0+exp(-x*theta)); % 80*1

loss=@(theta,y,x)  (x'*(z(x,theta)-y))/m;  % 3*1

for i=1:iteration


%hessian=(x'*diag(z(x,theta))*diag(z(-x,theta))*x)/m;  % º∆À„Hessian matrix 3*3
hessian=(x'*diag(z(x,theta))*diag(1-z(x,theta))*x)/m;
gradient=loss(theta,y,x);  % 3*1
J(i)=(-y'*log(z(x,theta))-(1-y')*log(1-z(x,theta)))/m;  
theta=theta - hessian\gradient;  % 3*1

end


plot_x = [min(x(:,2))-5,  max(x(:,2))+5];

plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold on
plot(20,80,'p', 'MarkerFaceColor','g', 'MarkerSize',16)
figure;
plot(0:iteration-1, J, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
xlabel('Iteration'); ylabel('J')




