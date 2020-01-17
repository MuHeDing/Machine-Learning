
red=load('F:\\Machine Learning\\exp\\ex3Data\\ex3red.dat');%14*2 
blue=load('F:\\Machine Learning\\exp\\ex3Data\\ex3blue.dat');
green=load('F:\\Machine Learning\\exp\\ex3Data\\ex3green.dat');
%two(red,blue)
three(red,blue,green)
function two(red,blue)
% 行长度
m=size(red,1);
n=size(blue,1);
% 均值
rm=mean(red);

bm=mean(blue);
% 画出所有点
plot(red(:,1),red(:,2),'rx')
hold on


plot(blue(:,1),blue(:,2),'b.')

s1=(red-rm)'*(red-rm) ;% 2*2
s2=(blue-bm)'*(blue-bm);
sw=s1+s2;
sb=(rm-bm)'*(rm-bm);

%方法一
[V,D]=eig(inv(sw)*sb);
eigenvalue=diag(D);
lamda=max(eigenvalue);%求最大特征值
for i=1:length(V)%求最大特征值对应的序数
    if lamda==eigenvalue(i)
        break;
    end
end
disp(V)
w=V(:,i); %2,1

k=w(2)/w(1);

plot([0,10],[0,10*k],'k');

%使用方法二来求出 w
% w=inv(sw)*(rm-bm)'  % 2*1
% 
% k=w(2)/w(1)
% 
% k=w(2)/w(1);

plot([0,10],[0,10*k],'k');
rx=[];
ry=[];
for i=1:m
    rx(i)=(red(i,1)+k*red(i,2))/(k^2+1);
    ry(i)=k*rx(i);
end
plot(rx,ry,'r*');
hold on
bx=[];
by=[];
for i=1:n
    bx(i)=(blue(i,1)+k*blue(i,2))/(k^2+1);
    by(i)=k*bx(i);
end
plot(bx,by,'b*');
end

function three(red,blue,green)

% 行长度
m=size(red,1);
n=size(blue,1);
z=size(green,1);
% 均值
rm=mean(red);

bm=mean(blue);

gm=mean(green);
% 画出所有点
plot(red(:,1),red(:,2),'rx')
hold on

plot(blue(:,1),blue(:,2),'b*')

plot(green(:,1),green(:,2),'g*')


miu=(rm+bm+gm)/3; %1*2

sb1=m*(rm-miu)'*(rm-miu);
sb2=n*(bm-miu)'*(bm-miu);
sb3=z*(gm-miu)'*(gm-miu);

sb=sb1+sb2+sb3;

sw1=(red-rm)'*(red-rm);
sw2=(blue-bm)'*(blue-bm);
sw3=(green-gm)'*(green-gm);

sw=sw1+sw2+sw3;


[V,D]=eig(inv(sw)*sb);
eigenvalue=diag(D);
lamda=max(eigenvalue);%求最大特征值
for i=1:length(V)%求最大特征值对应的序数
    if lamda==eigenvalue(i)
        break;
    end
end
w=V(:,i);

k=w(2)/w(1);




plot([0,10],[0,10*k],'k');

rx=[];
ry=[];
for i=1:m
    rx(i)=(red(i,1)+k*red(i,2))/(k^2+1);
    ry(i)=k*rx(i);
end
plot(rx,ry,'ro');
hold on
bx=[];
by=[];
for i=1:n
    bx(i)=(blue(i,1)+k*blue(i,2))/(k^2+1);
    by(i)=k*bx(i);
end
plot(bx,by,'bo');
gx=[];
gy=[];
for i=1:z
    gx(i)=(green(i,1)+k*green(i,2))/(k^2+1);
    gy(i)=k*gx(i);
end
plot(gx,gy,'go');
 end



