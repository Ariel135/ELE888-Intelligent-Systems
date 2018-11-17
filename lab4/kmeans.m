function kmeans(c)
%Algorithm takes k (c=k) means and classifies data into clusters around
%different mean points
I=imread('house.tiff');
imshow(I);
[M,N,D]=size(I);
X=reshape(I,M*N,3);
x=double(X);
figure;
plot3(x(:,1),x(:,2),x(:,3),'.')
xlim([0 255]);
ylim([0 255]);
zlim([0 255]);
hold on; grid;
title('All pixels in RGB');
xlabel('Red');
ylabel('Green');
zlabel('Blue');

mu=zeros(2,3); %Data has 3 axes: R,G,B
Mus=zeros(2,3);
Mucount=1;


classifyMatrix = zeros(length(X),1);


disp('Initial Mu values');
mu=rand(c,3)*255 %Initialize random numbers between 0 and 255

for i=1:c
    Mus(i,:)=mu(i,:);
    Mucount=Mucount+1;
end

iteration=0;
delta_mu=1;
jpt_count=1;
f=@(a,b) (a-b).^2;
jpt=0;
jps=0;



while(delta_mu>0)
    v=zeros(length(x),2);
    jp=zeros(length(x),2);
    index=0;
    for j=1:c
        v(:,j)=squeeze(sqrt(sum(bsxfun(f,x,mu(j,:)),2)));
        jp(:,j)  = sum( bsxfun(f,x,mu(j,:)) ,2);
        jps(j) = sum(jp(:,j));
    end
    
    jpt(jpt_count) = sum(jps);
    jpt_count = jpt_count+ 1;
    [minv,index]=min(v,[],2);
    [cluster,count]=MinIndex(x,index);
    
    if(iteration==0) %Get cluster mean at first stage
        cluster1=cluster;
        count1=count; %Get cluster mean at second stage
    end
    if(iteration==1)
        cluster2=cluster;
        count2=count;
    end
    meanV=zeros(1,3);
    delta_mu=0;
    
    for i=1:c
        meanV=mean(cluster(1:count(i)-1,(i*3)-2:(i*3)));
        Mus(Mucount,:)=meanV;
        Mucount=Mucount+1;
        delta_mu=delta_mu+abs(mu(i,:)-meanV);
        %delta_mu will remain zero if no mu change from previous iteration
        mu(i,:)=meanV;
    end
    iteration=iteration+1;
end
figure;
plot(1:iteration,jpt); %Plot the error criterion
grid;
xlabel('Number of Iterations');
ylabel('Error Criterion J');
title('Error Criterion');

%% Find the Xie-Beni Index
XieBeni=0;
for k=1:size(x,1)
    for j=1:c
        if index(k,1)==j
            XieBeni=XieBeni+v(k,j)/min(norm(mu-repmat(mu(c,:),c,1)));
        end
    end
end

disp('Xie-Beni Index');
XieBeni=XieBeni/size(x,1)

%% Plot first stage clusters
figure;
subplot(2,2,1);
colour=zeros(1,3);
for i=1:c
    colour(i,:)= Mus(length(Mus)-i,:)/255;
    plot3(cluster1(1:count1(i)-1,(i*3)-2),cluster1(1:count1(i)-1,(i*3)-1),cluster1(1:count1(i)-1,(i*3)),'.','Color',colour(i,:));
    hold on;
end
xlabel('Red');
ylabel('Green');
zlabel('Blue');
title('First Stage');
grid; 
%% Plot second stage clusters
subplot(2,2,2);
colour=zeros(1,3);
for i=1:c
    colour(i,:)= Mus(length(Mus)-i,:)/255;
    plot3(cluster2(1:count2(i)-1,(i*3)-2),cluster2(1:count2(i)-1,(i*3)-1),cluster2(1:count2(i)-1,(i*3)),'.','Color',colour(i,:));
    hold on;
end
xlabel('Red');
ylabel('Green');
zlabel('Blue');
title('Second Stage');
grid; 
%% Plot final stage clusters
subplot(2,2,3);
colour=zeros(1,3);
for i=1:c
    colour(i,:)= Mus(length(Mus)-i,:)/255;
    plot3(cluster(1:count(i)-1,(i*3)-2),cluster(1:count(i)-1,(i*3)-1),cluster(1:count(i)-1,(i*3)),'.','Color',colour(i,:));
    hold on;
end
xlabel('Red');
ylabel('Green');
zlabel('Blue');
title('Final Stage');
grid;

%% Display the image w/ dominant colours
for i=1:length(index)
    image(i,:)=mu(index(i),:);
end

Ilabeled=reshape(image,M,N,3);
subplot(2,2,4);
imshow(uint8(Ilabeled));
title('Image');

%% Plot the cluster means

figure;
subplot(2,2,3)
plot3(Mus(length(Mus)-c:length(Mus),1),Mus(length(Mus)-c:length(Mus),2),Mus(length(Mus)-c:length(Mus),3),'*','Color',[ 0 0 1 ])
title('Last Cluster Mean');
grid;

subplot(2,2,1)
plot3(Mus(1:c,1),Mus(1:c,2),Mus(1:c,3),'*','Color',[ 0 0 1 ])
title('First Cluster Mean');
grid;

subplot(2,2,2)
plot3(Mus(c+1:c*2,1),Mus(c+1:c*2,2),Mus(c+1:c*2,3),'*','Color',[ 0 0 1 ])
title('Second Cluster Mean');
grid;

%% Display any data

disp('Final Mean values');
mu
end