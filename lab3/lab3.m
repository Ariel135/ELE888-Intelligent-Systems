clear all;
%% Part 1: Classical XOR
disp('Inputs x1 and x2');
x1=[-1 -1 1 1];
x2=[-1 1 -1 1];

disp('Target values [t]');
target=[-1 1 1 -1]

disp('Threshold and learning rate values');
learning_rate=0.1
threshold=0.001


zk=[0 0 0 0];
deltaJ=0;
itr=1;
flag=0;

disp('Randomly Initial Weight Values');
wj1=[0.707 1 1]
wj2=[-1.63 1 1]
wk=[1 0.4 -0.6]

while flag==0

    deltawj1=[0 0 0];
    deltawj2=[0 0 0];
    deltawjk=[0 0 0];
    
    for n=1:length(x2)
        
        xm=[1 x1(n) x2(n)];
        
        netj1=wj1*xm';
        netj2=wj2*xm';
        
        y1_der=1-(tanh(netj1))^2;
        y2_der=1-(tanh(netj2))^2;
        
        y1=tanh(netj1);
        y2=tanh(netj2);
        
        y=[1 y1 y2];
        
        netk=y*wk';
        
        zk(n)=tanh(netk);
        
        zkprime=1-zk(n)^2;
        
        deltak=(target(n)-zk(n))*zkprime;
        deltaj1=y1_der*wk(2)*deltak;
        deltaj2=y2_der*wk(3)*deltak;

        deltawj1=deltawj1+learning_rate*deltaj1*xm;
        deltawj2=deltawj2+learning_rate*deltaj2*xm;
        deltawjk=deltawjk+learning_rate*deltak*y;
    end
    
    %Weight Vectors
    wj1 = wj1 + deltawj1;
    wj2 = wj2 + deltawj2;
    wk  = wk  + deltawjk;
    
    %Error criterion
    J(itr) = 0.5*norm(target-zk)^2;
    
    if (itr==1)
        deltaJ(itr)=J(itr);
    else
        deltaJ(itr) = abs(J(itr-1) - J(itr));
        if ((deltaJ(itr) < threshold) && itr < 100000)
            flag=1;
        end
    end
    
    itr = itr + 1;
end

%Plot learning curve
figure;
n=[0:1:length(J)-1];
plot(n,J)
hold on; grid;
title('Learning Curve For J(r)');
ylabel('J(r)');
xlabel('r');


disp('Number of epoch iterations required:');
itr-1


disp('Final weight vectors');
wj1
wj2
wk

%Plotting the decision boundaries for XOR

figure;

x11=(-3:3);
w01=wj1(1);
w11=wj1(2);
w21=wj1(3);
x21=-(w11/w21)*x11-(w01/w21);
plot(x11,x21,'--');

w01=wj2(1);
w11=wj2(2);
w21=wj2(3);
x22=-(w11/w21)*x11-(w01/w21);

hold on;

boundedline=plot(x11,x22,'-');

for i=1:length(x1)
    if (zk(i)<0)
        false=plot(x1(i),x2(i),'bo');
    else
        true=plot(x1(i),x2(i),'bx');
    end
end


hold on; 
grid;

title('x1 vs. x2 Decision Boundaries');
xlabel('x1');
ylabel('x2');

legend([true,false,boundedline],'XOR True = 1','XOR False = -1 ','Boundary Line');

%Find the accuracy

correct=0;
accuracy=0;


for i=1:length(x1)
    if floor(zk(i))==target(i) || ceil(zk(i))==target(i)
        correct=correct+1;
    end
end

accuracy=correct*100/length(x1);
disp('Accuracy');
accuracy

% figure;

y11=(-3:3);
w01=wk(1);
w11=wk(2);
w21=wk(3);
y21=-(w11/w21)*x11-(w01/w21);

plot(x11,x22,'k');

hold on; grid;

title('Decision Space - y1 vs y2');

xlabel('y1');
ylabel('y2');

%% Part 2: Wine Sorting

WineData=load('wine.data');

TrainingSet=zeros(107,3);

TrainingSet(1:59,:,:)=WineData(1:59,1:3); %Class 1=w1
TrainingSet(60:107,:,:)=WineData(131:178,1:3); %Class 3=w2

% Normalize TrainingSet such that w2=-1
for i=60:length(TrainingSet)
    TrainingSet(i,1)=-1;
end

target=TrainingSet(:,1)'; %Targets are class labels

x1=TrainingSet(:,2)'; %make x1 represent alcohol content

x2=TrainingSet(:,3)'; %make x2 represent malic acid content

data = x1;
actual_minx1 = min(data(:));
actual_maxx1 = max(data(:));
desired_min = -3;
desired_max =  3;

x1 = (data - actual_minx1)*((desired_max - desired_min)/(actual_maxx1 - actual_minx1)) + desired_min;

data = x2;
actual_minx2 = min(data(:));
actual_maxx2 = max(data(:));

x2 = (data - actual_minx2)*((desired_max - desired_min)/(actual_maxx2 - actual_minx2)) + desired_min;


zk=[0 0 0 0];
itr=1;
deltaJ=0;
flag=0;

while flag==0

    deltawj1=[0 0 0];
    deltawj2=[0 0 0];
    deltawjk=[0 0 0];
    
    for n=1:length(x1)
        xm=[1 x1(n) x2(n)];
        
        netj1=wj1*xm';
        netj2=wj2*xm';
        
        y1_der=1-(tanh(netj1))^2;
        y2_der=1-(tanh(netj2))^2;
        y1=tanh(netj1);
        y2=tanh(netj2);
        
        y=[1 y1 y2];
        
        netk=y*wk';
        
        zk(n)=tanh(netk);
        zkprime=1-zk(n)^2;
        
        deltak=(target(n)-zk(n))*zkprime;
        deltaj1=y1_der*wk(2)*deltak;
        deltaj2=y2_der*wk(3)*deltak;

        deltawj1=deltawj1+learning_rate*deltaj1*xm;
        deltawj2=deltawj2+learning_rate*deltaj2*xm;
        deltawjk=deltawjk+learning_rate*deltak*y;
    end
    
    %Weight Vectors
    wj1=wj1+deltawj1;
    wj2=wj2+deltawj2;
    wk=wk+deltawjk;
    
    %Error criterion
    J(itr)=0.5*norm(target-zk)^2;
    
    if (itr==1)
        deltaJ(itr)=J(itr);
    else
        deltaJ(itr)=abs(J(itr-1)-J(itr));
        if ((deltaJ(itr)<threshold) && itr<100000)
            flag=1;
        end
    end
    
    itr=itr+1;
end

%Plot learning curve
figure;
n=[0:1:length(J)-1];
plot(n,J)
hold on; 
grid;
title('Learning Curve J(r)');
xlabel('r');
ylabel('J(r)');

disp('Number of epoch iterations required:');
itr-1
disp('Final weight vectors');
wj1
wj2
wk

%Plot the decision boundaries

figure;
x11=(-3:3);
w01=wj1(1);
w11=wj1(2);
w21=wj1(3);
x21=-(w11/w21)*x11-(w01/w21);
plot(x11,x21,'k');

w01=wj2(1);
w11=wj2(2);
w21=wj2(3);
x22=-(w11/w21)*x11-(w01/w21);

hold on;

boundedline=plot(x11,x22,'k');

for i=1:length(x1)
    if (zk(i)<0)
        false=plot(x1(i),x2(i),'bo');
    else
        true=plot(x1(i),x2(i),'bx');
    end
end

hold on; 
grid;

title('x1 vs. x2 Decision Space');
ylabel('Malic Acid Content');
xlabel('Alcohol Content (%)');

legend([true,false,boundedline],'Class 1','Class 2','Boundary');

%Find the accuracy
correct=0;
accuracy=0;
for i=1:length(x1)
    if floor(zk(i))==target(i) || ceil(zk(i))==target(i)
        correct=correct+1;
    end
end
accuracy=correct*100/length(x1);
disp('Accuracy');
accuracy

figure;
y11=(-3:3);
w01=wk(1);
w11=wk(2);
w21=wk(3);

y21=-(w11/w21)*x11-(w01/w21);

plot(x11,x22,'k');

hold on; 
grid;

title('y1 vs. y2 Decision Space');
xlabel('y1');
ylabel('y2');
