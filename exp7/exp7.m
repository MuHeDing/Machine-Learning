[trainlabels,trainfeatures]=libsvmread('F:\Machine Learning\exp\ex7Data\twofeature.txt');

m=size(trainlabels);  % 51 *1
n=size(trainfeatures);  % 51 *2
pos=find(trainlabels==1);  
neg = find(trainlabels==-1);

plot(trainfeatures(pos, 1), trainfeatures(pos, 2),'o','MarkerFaceColor', 'b','MarkerSize', 5);
hold on
plot(trainfeatures(neg, 1), trainfeatures(neg, 2),'o','MarkerFaceColor', 'g','MarkerSize', 5);

model=svmtrain(trainlabels,trainfeatures,'-s 0 -t 0 -c 100');

w=model.SVs'* model.sv_coef;
b=-model.rho;
if(model.Label(1)==-1)
    w=-w;
    b=-b;
end
w
b
x = linspace(min(trainfeatures(:,1)), max(trainfeatures(:,1)), 30);
y = (-1/w(2))*(w(1)*x + b);
plot(x,y,'k-', 'LineWidth', 2);


[train_y,train_x]=libsvmread('F:\Machine Learning\exp\ex7Data\email_train-all.txt');
model=svmtrain(train_y,train_x,'-s 0 -t 0 -c 1');

w=model.SVs'* model.sv_coef;
b=-model.rho;
if(model.Label(1)==-1)
    w=-w;
    b=-b;
end
[test_y, test_x] = libsvmread('F:\Machine Learning\exp\ex7Data\email_test.txt');

[predicted_label, accuracy, decision_values] = svmpredict(test_y, test_x, model);
accuracy
% decision_values
% predicted_label
