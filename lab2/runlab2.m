%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAB 1, Bayesian Decision Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Attribute Information for IRIS data:
%    1. sepal length in cm
%    2. sepal width in cm
%    3. petal length in cm
%    4. petal width in cm

%    class label/numeric label: 
%       -- Iris Setosa / 1 
%       -- Iris Versicolour / 2
%       -- Iris Virginica / 3


%% this script will run lab1 experiments..
clear
load irisdata.mat

%% extract unique labels (class names)
labels = unique(irisdata_labels);

%% generate numeric labels
numericLabels = zeros(size(irisdata_features,1),1);
for i = 1:size(labels,1)
    numericLabels(find(strcmp(labels{i},irisdata_labels)),:)= i;
end

% %% feature distribution of x1 for two classes
% figure
% 
%     
% subplot(1,2,1), hist(irisdata_features(find(numericLabels(:)==1),2),100), title('Iris Setosa, sepal width (cm)');
% subplot(1,2,2), hist(irisdata_features(find(numericLabels(:)==2),2),100); title('Iris Veriscolour, sepal width (cm)');
% 
% figure
% 
% subplot(1,2,1), hist(irisdata_features(find(numericLabels(:)==1),1),100), title('Iris Setosa, sepal length (cm)');
% subplot(1,2,2), hist(irisdata_features(find(numericLabels(:)==2),1),100); title('Iris Veriscolour, sepal length (cm)');
%     
% 
% figure
% 
% plot(irisdata_features(find(numericLabels(:)==1),1),irisdata_features(find(numericLabels(:)==1),2),'rs'); title('x_1 vs x_2');
% hold on;
% plot(irisdata_features(find(numericLabels(:)==2),1),irisdata_features(find(numericLabels(:)==2),2),'k.');
% axis([4 7 1 5]);
% 
    

% %% build training data set for two class comparison
% % merge feature samples with numeric labels for two class comparison (Iris
% % Setosa vs. Iris Veriscolour
% trainingSet = [irisdata_features(1:150,:) numericLabels(1:150,1)];
% D=trainingSet
% A=D(1:50,2:3);
% B=D(51:100,2:3);
% C=D(101:150,2:3);
% x=0.7*length(A);
% z=0.7*length(B)
% Atrain = A(1:(0.7*length(A)),1:2);
% Btrain = B(1:(0.7*length(B)),1:2);
% Aclassify = A(x+1:end,1:2);
% Bclassify = B(z+1:end,1:2);
% 
% AugAtrain = [ones(size(Atrain,1),1),Atrain];
% AugBtrain = [ones(size(Btrain,1),1),Btrain];
% NormBtrain = -AugBtrain
% 
% AugAtrain = AugAtrain';
% AugBtrain = AugBtrain';
% NormBtrain = NormBtrain';
% Yi = [AugAtrain, NormBtrain];
% Y = [AugAtrain, AugBtrain]
% a= [0 0 1]; %a initial
% threshold = 0;
% learn_rate = 0.01;
% 
% for t=1:300
%     M = a*Yi
%     Mlogic = M <= 0;
%     a = a' - (sum(Yi(:,Mlogic),2)*-learn_rate)
%     a = a'
% end    
% 
% Y = [ones(size(Aclassify,1),1), Aclassify]'
% 
% 
% Yb = [ones(size(Bclassify,1),1), Bclassify]'
% 
% gx = a*Yb



%% Lab1 experiments (include here)
%% Question 1
trainingSet = [irisdata_features(1:150,:) numericLabels(1:150,1)];
D=trainingSet
A=D(1:50,2:3);
B=D(51:100,2:3);


Atrain = A(1:(0.3*length(A)),1:2); %30 percent
Btrain = B(1:(0.3*length(B)),1:2); %30 percent

Aclassify = A((0.3*length(A):length(A)),1:2); %70 percent
Bclassify = B((0.3*length(B):length(B)),1:2); % 70 percent

AugAtrain = [ones(size(Atrain,1),1),Atrain];
AugBtrain = [ones(size(Btrain,1),1),Btrain];
NormBtrain = -AugBtrain

AugAtrain = AugAtrain';
AugBtrain = AugBtrain';
NormBtrain = NormBtrain';
Yi = [AugAtrain, NormBtrain];
Y = [AugAtrain, AugBtrain]
a= [0 0 1]; %a initial
threshold = 0;
learn_rate = 0.01;
count = 0;

figure;
scatter(Atrain(:,1),Atrain(:,2),'rs')
hold on;
scatter(Btrain(:,1),Btrain(:,2),'k')
hold on;
syms x2 x3;
eqn = a*[1;x2;x3] == 0;
xSol = solve(eqn, x3);
ezplot(xSol);


for t=1:300
    count = count + 1;
    M = a*Yi;
    Mlogic = M <= 0;
    a = a' - (sum(Yi(:,Mlogic),2)*-learn_rate);
    a = a';
    eqn = a*[1;x2;x3] == 0;
    xSol = solve(eqn, x3);
    ezplot(xSol);
    hold on;
    if (all(Mlogic(:) == threshold))
        break;
    end   
    
end    

Legend=cell(count+3,1)
 for iter=1:count+3
   if iter == 1
       Legend{iter} = ('Class A');
   elseif iter == 2
       Legend{iter} = ('Class B');
   else
   Legend{iter}=strcat('interation', num2str(iter-3));
   end
 end
legend(Legend)
title('x2 vs x3 (Training Data of 30%)');
xlim([-6 6]);
ylim([-6 6]);
xlabel('x2');
ylabel('x3');


eqn = a*[1;x2;x3] == 0;
xSol = solve(eqn, x3) == x3;

figure;
scatter(Atrain(:,1),Atrain(:,2),'rs')
hold on;
scatter(Btrain(:,1),Btrain(:,2),'k')
ezplot(xSol);
title('x2 vs x3 (Training Data of 30%)');
legend('Class A','Class B');
xlim([0 6]);
ylim([0 6]);
xlabel('x2');
ylabel('x3');


%question 2

Ya = [ones(size(Aclassify,1),1), Aclassify]'

gxa = a*Ya

gxAAcc = 1 -((length(gxa)-sum(gxa(:) > 0))/length(gxa)); % g(x) for A

Yb = [ones(size(Bclassify,1),1), Bclassify]'

gxb = a*Yb

gxBAcc = 1 -((length(gxb)-sum(gxb(:) < 0))/length(gxb)); % g(x) for B


%% question 3

trainingSet = [irisdata_features(1:150,:) numericLabels(1:150,1)];
D=trainingSet
A=D(1:50,2:3);
B=D(51:100,2:3);


Atrain = A(1:(0.7*length(A)),1:2); %70
Btrain = B(1:(0.7*length(B)),1:2); %70

Aclassify = A((round(0.7*length(A)+1):length(A)),1:2); %30
Bclassify = B((round(0.7*length(B)+1):length(B)),1:2); %30

AugAtrain = [ones(size(Atrain,1),1),Atrain];
AugBtrain = [ones(size(Btrain,1),1),Btrain];
NormBtrain = -AugBtrain

AugAtrain = AugAtrain';
AugBtrain = AugBtrain';
NormBtrain = NormBtrain';
Yi = [AugAtrain, NormBtrain];
Y = [AugAtrain, AugBtrain]
a= [0 0 1]; %a initial
threshold = 0;
learn_rate = 0.01;
count = 0;

figure;
scatter(Atrain(:,1),Atrain(:,2),'rs')
hold on;
scatter(Btrain(:,1),Btrain(:,2),'k')
hold on;
syms x2 x3;
eqn = a*[1;x2;x3] == 0;
xSol = solve(eqn, x3);
ezplot(xSol);

for t=1:300
    count = count + 1;
    M = a*Yi
    Mlogic = M <= 0;
    a = a' - (sum(Yi(:,Mlogic),2)*-learn_rate)
    a = a'
    eqn = a*[1;x2;x3] == 0;
    xSol = solve(eqn, x3);
    ezplot(xSol);
    if (all(Mlogic(:) == threshold))
        break;
    end     
end 
Legend=cell(count+3,1)
 for iter=1:count+3
   if iter == 1
       Legend{iter} = ('Class A');
   elseif iter == 2
       Legend{iter} = ('Class B');
   else
   Legend{iter}=strcat('interation', num2str(iter-3));
   end
 end
legend(Legend)
title('x2 vs x3 (Training Data of 70%)');
xlim([-6 6]);
ylim([-6 6]);
xlabel('x2');
ylabel('x3');

 
syms x2 x3;
eqn = a*[1;x2;x3] == 0;
xSol = solve(eqn, x3) == x3;

figure;
scatter(Atrain(:,1),Atrain(:,2),'rs')
hold on;
scatter(Btrain(:,1),Btrain(:,2),'k')
ezplot(xSol);
title('x2 vs x3 (Training Data of 70%)');
legend('Class A','Class B');
xlabel('x2');
ylabel('x3');
xlim([0 6]);
ylim([0 6]);



Ya = [ones(size(Aclassify,1),1), Aclassify]'

gxa = a*Ya

gxAAcc = 1 -((length(gxa)-sum(gxa(:) > 0))/length(gxa)); % g(x) for A

Yb = [ones(size(Bclassify,1),1), Bclassify]'

gxb = a*Yb

gxBAcc = 1 -((length(gxb)-sum(gxb(:) < 0))/length(gxb)); % g(x) for B

gxb = a*Yb

%% Question 4.1

trainingSet = [irisdata_features(1:150,:) numericLabels(1:150,1)];
D=trainingSet

B=D(51:100,2:3);
C=D(101:150,2:3);

Btrain = B(1:(0.7*length(B)),1:2); %70 Percent training set
Ctrain = C(1:(0.7*length(C)),1:2); %70 Percent training set

Bclassify = B((0.7*length(B))+1:(length(B)),1:2); % 30 percent classification set
Cclassify = C((0.7*length(C))+1:(length(C)),1:2); % 30 percent classification set

AugBtrain = [ones(size(Btrain,1),1),Btrain];
AugCtrain = [ones(size(Ctrain,1),1),Ctrain];
NormCtrain = -AugCtrain;

Yi = [AugBtrain', NormCtrain'];
Y = [AugBtrain', AugCtrain'];
a= [0 0 1]; %a initial
threshold = 0;
learn_rate = 0.01;
count = 0;

figure;
scatter(Btrain(:,1),Btrain(:,2),'rs')
hold on;
scatter(Ctrain(:,1),Ctrain(:,2),'k')
hold on;
syms x2 x3;
eqn = a*[1;x2;x3] == 0;
xSol = solve(eqn, x3);
ezplot(xSol);
for t=1:300
    count = count + 1;
    M = a*Yi
    Mlogic = M <= 0;
    a = a' - (sum(Yi(:,Mlogic),2)*-learn_rate);
    a = a';
%     eqn = a*[1;x2;x3] == 0;
%     xSol = solve(eqn, x3);
%     ezplot(xSol);
%     Uncomment for graph of linear boundary through iterations
    if (all(Mlogic(:) == threshold))
        break;
    end   
    
end  
Legend=cell(count+3,1)
 for iter=1:count+3
   if iter == 1
       Legend{iter} = ('Class B');
   elseif iter == 2
       Legend{iter} = ('Class C');
   else
   Legend{iter}=strcat('interation', num2str(iter-3));
   end
 end
legend(Legend)
title('x2 vs x3 (Training Data of 70%)');
xlim([-6 6]);
ylim([-6 6]);
xlabel('x2');
ylabel('x3');
 
syms x2 x3;
eqn = a*[1;x2;x3] == 0;
xSol = solve(eqn, x3) == x3;

figure;
scatter(Btrain(:,1),Btrain(:,2),'rs');
hold on;
scatter(Ctrain(:,1),Ctrain(:,2),'k');
ezplot(xSol);
title('x2 vs x3 (Training Data of 70%)');
legend('Class B','Class C');
xlabel('x2');
ylabel('x3');
xlim([0 6]);
ylim([0 6]);

Yb = [ones(size(Bclassify,1),1), Bclassify]'

gxb = a*Yb

gxBAcc = 1 -((length(gxb)-sum(gxb(:) > 0))/length(gxb)); % g(x) for class B

Yc = [ones(size(Cclassify,1),1), Cclassify]'

gxc = a*Yc

gxCAcc = 1 -((length(gxc)-sum(gxc(:) < 0))/length(gxc)); % g(x) for class A

%% Question 4.2

trainingSet = [irisdata_features(1:150,:) numericLabels(1:150,1)];
D=trainingSet

B=D(51:100,2:3);
C=D(101:150,2:3);

Btrain = B(1:(0.3*length(B)),1:2); %30 Percent training set
Ctrain = C(1:(0.3*length(C)),1:2); %30 Percent training set

Bclassify = B((0.3*length(B))+1:(length(B)),1:2); %70 percent classification set
Cclassify = C((0.3*length(C))+1:(length(C)),1:2); %70 percent classification set


AugBtrain = [ones(size(Btrain,1),1),Btrain];
AugCtrain = [ones(size(Ctrain,1),1),Ctrain];
NormCtrain = -AugCtrain;


Yi = [AugBtrain', NormCtrain'];
Y = [AugBtrain', AugCtrain'];
a= [0 0 1]; %a initial
threshold = 0;
learn_rate = 0.01;

count = 0;

figure;
scatter(Btrain(:,1),Btrain(:,2),'rs')
hold on;
scatter(Ctrain(:,1),Ctrain(:,2),'k')
hold on;
syms x2 x3;
eqn = a*[1;x2;x3] == 0;
xSol = solve(eqn, x3);
ezplot(xSol);

for t=1:300
    count = count + 1;
    M = a*Yi;
    Mlogic = M <= 0;
    a = a' - (sum(Yi(:,Mlogic),2)*-learn_rate);
    a = a';
%     eqn = a*[1;x2;x3] == 0;
%     xSol = solve(eqn, x3);
%     ezplot(xSol);
%     Uncomment for graph of linear boundary through iterations
    if (all(Mlogic(:) == threshold))
        break;
    end   
   
end 
Legend=cell(count+3,1)
 for iter=1:count+3
   if iter == 1
       Legend{iter} = ('Class B');
   elseif iter == 2
       Legend{iter} = ('Class C');
   else
   Legend{iter}=strcat('interation', num2str(iter-3));
   end
 end
legend(Legend)
title('x2 vs x3 (Training Data of 30%)');
xlim([-6 6]);
ylim([-6 6]);
xlabel('x2');
ylabel('x3');


syms x2 x3;
eqn = a*[1;x2;x3] == 0;
xSol = solve(eqn, x3) == x3;

figure;
scatter(Btrain(:,1),Btrain(:,2),'rs');
hold on;
scatter(Ctrain(:,1),Ctrain(:,2),'k');
ezplot(xSol);
title('x2 vs x3 (Training Data of 30%)');
legend('Class B','Class C');
xlabel('x2');
ylabel('x3');
xlim([0 6]);
ylim([0 6]);

Yb = [ones(size(Bclassify,1),1), Bclassify]'

gxb = a*Yb

gxBAcc = 1 -((length(gxb)-sum(gxb(:) > 0))/length(gxb)); % g(x) for class B

Yc = [ones(size(Cclassify,1),1), Cclassify]'

gxc = a*Yc

gxCAcc = 1 -((length(gxc)-sum(gxc(:) < 0))/length(gxc)); % g(x) for class C

%% Question 5 reworked

trainingSet = [irisdata_features(1:150,:) numericLabels(1:150,1)];
D=trainingSet
A=D(1:50,2:3);
B=D(51:100,2:3);

Atrain = A(1:(0.7*length(A)),1:2);
Btrain = B(1:(0.7*length(B)),1:2);

Aclassify = A((round(0.7*length(A)+1):length(A)),1:2);
Bclassify = B((round(0.7*length(B)+1):length(B)),1:2);

AugAtrain = [ones(size(Atrain,1),1),Atrain];
AugBtrain = [ones(size(Btrain,1),1),Btrain];
NormBtrain = -AugBtrain

AugAtrain = AugAtrain';
AugBtrain = AugBtrain';
NormBtrain = NormBtrain';
Yi = [AugAtrain, NormBtrain];
Y = [AugAtrain, AugBtrain]
a= [0 0 5]; %a initial
threshold = 0;
learn_rate = 0.04;
count = 0;

figure;
scatter(Atrain(:,1),Atrain(:,2),'rs')
hold on;
scatter(Btrain(:,1),Btrain(:,2),'k')
hold on;
syms x2 x3;
eqn = a*[1;x2;x3] == 0;
xSol = solve(eqn, x3);
ezplot(xSol);

for t=1:300
    count = count + 1;
    M = a*Yi
    Mlogic = M <= 0;
    a = a' - (sum(Yi(:,Mlogic),2)*-learn_rate);
    a = a';
    eqn = a*[1;x2;x3] == 0;
    xSol = solve(eqn, x3);
    ezplot(xSol);
    if (all(Mlogic(:) == threshold))
        break;
    end     
end 
Legend=cell(count+3,1);
 for iter=1:count+3
   if iter == 1
       Legend{iter} = ('Class A');
   elseif iter == 2
       Legend{iter} = ('Class B');
   else
   Legend{iter}=strcat('interation', num2str(iter-3));
   end
 end
legend(Legend);
title('x2 vs x3 (Training Data of 70%)');
xlim([-6 6]);
ylim([-6 6]);
xlabel('x2');
ylabel('x3');

 
syms x2 x3;
eqn = a*[1;x2;x3] == 0;
xSol = solve(eqn, x3) == x3;

figure;
scatter(Atrain(:,1),Atrain(:,2),'rs');
hold on;
scatter(Btrain(:,1),Btrain(:,2),'k');
ezplot(xSol);
title('x2 vs x3 (Training Data of 70%)');
legend('Class A','Class B');
xlabel('x2');
ylabel('x3');
xlim([0 6]);
ylim([0 6]);



Ya = [ones(size(Aclassify,1),1), Aclassify]'

gxa = a*Ya

gxAAccModified = 1 -((length(gxa)-sum(gxa(:) > 0))/length(gxa));

Yb = [ones(size(Bclassify,1),1), Bclassify]'

gxb = a*Yb

gxBAccModfied = 1 -((length(gxb)-sum(gxb(:) < 0))/length(gxb));

gxb = a*Yb
