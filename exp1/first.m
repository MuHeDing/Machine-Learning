x = load ( 'F:\\Machine Learning\\exp\\ex1Data\\ex1x.dat') ; %50Î¬
y = load ( 'F:\\Machine Learning\\exp\\ex1Data\\ex1y.dat') ; %50Î¬
figure % open a new f i g u r e window
plot (x , y , ' o ' ) ;
ylabel ( 'Height in meters ')
xlabel ( 'Age in years ' )
u=0;
m=length(y); 
x=[ones(m,1),x]; % 50*2Î¬¶È
h=@(x,theta) x*theta;
alpha=0.07;
theta=zeros(2,1); % 2*1Î¬¶È
loss=@(theta,x,y) mean((h(x,theta)-y).^2)/2;
iteration=@(theta,alpha,y,x) theta-alpha*(x'*(h(x,theta)-y))/m;

for j=1:1500
    theta=iteration(theta,alpha,y,x);
end
hold on
plot(x(:,2),x*theta,'-');
legend( ' Training data ' , ' Linear regression' );

J_vals=zeros(100,100);
theta0=linspace(-3,3,100);
theta1=linspace(-1,1,100);
for i=1:length(theta0)
for j=1:length(theta1)
    t=[theta0(i);theta1(j)];
    J_vals(i,j)=loss(t,x,y);
end
end
J_vals =J_vals';
figure ;
surf (theta0, theta1, J_vals);
xlabel ( 'ntheta0' ) ; ylabel ( 'ntheta1' );
      
