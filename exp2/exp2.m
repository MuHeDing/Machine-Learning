%加载数据，数据预处理
x=load('F:\Machine Learning\exp\ex2Data\ex2x.dat');
y=load('F:\Machine Learning\exp\ex2Data\ex2y.dat');

m=47;
x=[ones(m,1),x];

sigma=std(x);
mu=mean(x);
x(:,2)=(x(:,2)-mu(2))./sigma(2);
x(:,3)=(x(:,3)-mu(3))./sigma(3);
%数据初始化，定义了四个学习率
theta=zeros(size(x(1,:)))';
theta1=zeros(size(x(1,:)))';
theta2=zeros(size(x(1,:)))';
theta3=zeros(size(x(1,:)))';
alpha=0.01;
alpha1=0.03;
alpha2=0.1;
alpha3=0.3;
J=zeros(50,1);
J1=zeros(50,1);
J2=zeros(50,1);
J3=zeros(50,1);
%函数定义
h=@(x,theta) x*theta;
loss=@(theta,x,y) mean((h(x,theta)-y).^2)/2;
iteration=@(theta,alpha,y,x) theta-alpha*(x'*(h(x,theta)-y))/m;

%开始循环迭代,使用四个学习率上来尝试
for num_iterations=1:50
J(num_iterations)=loss(theta,x,y);
theta=iteration(theta,alpha,y,x);
%第二个
J1(num_iterations)=loss(theta,x,y);
theta1=iteration(theta1,alpha1,y,x);
%第三个
J2(num_iterations)=loss(theta2,x,y);
theta2=iteration(theta2,alpha2,y,x);
%第四个
J3(num_iterations)=loss(theta3,x,y);
theta3=iteration(theta3,alpha3,y,x);
end
figure;
plot(0:49,J(1:50),'b-');
hold on;
plot(0:49,J1(1:50),'r-');
plot(0:49,J2(1:50),'k-');
plot(0:49,J3(1:50),'g-');
xlabel ( 'Number of iterations' );
ylabel ( ' Cost J ' );

theta=zeros(size(x(1,:)))';
alpha=0.4;
J=zeros(50,1);
for num_iterations=1:50
J(num_iterations)=loss(theta,x,y);
theta=iteration(theta,alpha,y,x);
end
plot(0:49,J(1:50),'b-');
xlabel ( 'Number of iterations' );
ylabel ( ' Cost J ' );
disp(theta);
% 数据预处理
t1=(1650-mu(2))./sigma(2);
t2=(3-mu(3))./sigma(3);
disp(h([1,t1,t2],theta));

% 使用
u=(x'*x)\x'*y
%u=x\y
disp(u)
t1=(1650-mu(2))./sigma(2);
t2=(3-mu(3))./sigma(3);
disp(h([1,t1,t2],u));






