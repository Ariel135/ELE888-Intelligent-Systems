Z = imread('house.tiff');
figure; imshow(Z);
[M,N,D]=size(Z);
X = reshape(Z,M*N, D);
X = double(X);
figure;
plot3(X(:,1),X(:,2),X(:,3),'.')
xlim([0 255]);
ylim([0 255]);
zlim([0 255]);
hold on; grid;
title('All pixels in RGB');
xlabel('Red');
ylabel('Green');
zlabel('Blue');



%% Part A

c=2
disp('Initial Mu values');
mu=rand(c,3)*255 %Initialize random numbers between 0 and 255

delta_mu = ones(c,3);

iteration = 1;

error_criterion = [];

while(~all(delta_mu(:) == 0))
    
   if iteration == 1
        figure;
        plot3(mu(:,1),mu(:,2),mu(:,3),'*'); % 1st iteration
        title('First Iteration');
        grid;
   elseif iteration == 2
       figure;
       plot3(mu(:,1),mu(:,2),mu(:,3),'*'); % 2nd iteration
       title('Second Iteration');
       grid;
   end  
    
    
   for i = 1:length(X)
       distanceTo1(i,:) = X(i,:) - mu(1,:);
       distanceTo2(i,:) = X(i,:) - mu(2,:); 
       
   end    

    Classification = sum(distanceTo1.^2,2) > sum(distanceTo2.^2,2);
    
    I = [X Classification];
    
    Cluster2=I(I(:,4) == 1, 1:3);
    Cluster1=I(I(:,4) ~= 1, 1:3);
    
    error_criterion_cluster1 = sum(sum((Cluster1 - repmat(mu(1,:),length(Cluster1),1)).^2,2));
    error_criterion_cluster2 = sum(sum((Cluster2 - repmat(mu(2,:),length(Cluster2),1)).^2,2));
    error_criterion(iteration) = error_criterion_cluster1 + error_criterion_cluster2;
    
    

    Centroid2 = mean(Cluster2);
    Centroid1 = mean(Cluster1);
    
    Centroids_New = [Centroid1;Centroid2];
    
    delta_mu = abs(mu - Centroids_New);
    
    mu = Centroids_New;
    
    iteration = iteration + 1;
    
end    
    
     figure;
     plot3(mu(:,1),mu(:,2),mu(:,3),'*'); % Last iteration
     title('Last Iteration');
     grid;
     
     figure;
     plot(error_criterion) % Last iteration
     title('Error Criterion')
     xlabel('Iteration')
     ylabel('J')
     grid;

    figure;
    plot3(Cluster1(:,1),Cluster1(:,2),Cluster1(:,3),'.','Color', mu(1,:)/255); % Cluster 1
    hold on;
    plot3(Cluster2(:,1),Cluster2(:,2),Cluster2(:,3),'.','Color', mu(2,:)/255); % Cluster 2
    xlim([0 255]);
    ylim([0 255]);
    zlim([0 255]);
    grid;
    title('All pixels in RGB');
    xlabel('Red');
    ylabel('Green');
    zlabel('Blue');

    J = ones(length(X),3);
    
    for i = 1 : length(X)
        if Classification(i) == 1
           J(i,:) = mu(2,:);
        else
           J(i,:) = mu(1,:);
        end
    end   

    Ilabeled = reshape(J,M,N,3)
    figure;
    imshow(uint8(Ilabeled));

%% Part B

c=5
disp('Initial Mu values');
mu=rand(c,3)*255 %Initialize random numbers between 0 and 255


delta_mu = ones(c,3);

iteration = 1;

classification = zeros(length(X),1);

Error_Criterion = 0;

while(~all(delta_mu(:) == 0))
     
    
   for i = 1:length(X)
       distanceTo1 = X(i,:) - mu(1,:);
       distanceTo2 = X(i,:) - mu(2,:); 
       distanceTo3 = X(i,:) - mu(3,:); 
       distanceTo4 = X(i,:) - mu(4,:); 
       distanceTo5 = X(i,:) - mu(5,:); 
       distances = [ sum(distanceTo1.^2,2) sum(distanceTo2.^2,2) sum(distanceTo3.^2,2) sum(distanceTo4.^2,2) sum(distanceTo5.^2,2)];
       [row,column] = min(distances);
       classification(i,1) = column;
       
   end    
   
    I = [X classification];
    
    Cluster1=I(I(:,4) == 1, 1:3);
    Cluster2=I(I(:,4) == 2, 1:3);
    Cluster3=I(I(:,4) == 3, 1:3);
    Cluster4=I(I(:,4) == 4, 1:3);
    Cluster5=I(I(:,4) == 5, 1:3);
    
    

    Centroid1 = mean(Cluster1);
    Centroid2 = mean(Cluster2);
    Centroid3 = mean(Cluster3);
    Centroid4 = mean(Cluster4);
    Centroid5 = mean(Cluster5);
    
    Centroids_New = [Centroid1;Centroid2;Centroid3;Centroid4;Centroid5];
    
    delta_mu = abs(mu - Centroids_New);
    
    mu = Centroids_New;
    
    iteration = iteration + 1;
    
    
end    
    
  
     
    figure;
    plot3(Cluster1(:,1),Cluster1(:,2),Cluster1(:,3),'.','Color', mu(1,:)/255); % Cluster 1
    hold on;
    plot3(Cluster2(:,1),Cluster2(:,2),Cluster2(:,3),'.','Color', mu(2,:)/255); % Cluster 2
    hold on;
    plot3(Cluster3(:,1),Cluster3(:,2),Cluster3(:,3),'.','Color', mu(3,:)/255); % Cluster 3
    hold on;
    plot3(Cluster4(:,1),Cluster4(:,2),Cluster4(:,3),'.','Color', mu(4,:)/255); % Cluster 4
    hold on;
    plot3(Cluster5(:,1),Cluster5(:,2),Cluster5(:,3),'.','Color', mu(5,:)/255); % Cluster 2
    xlim([0 255]);
    ylim([0 255]);
    zlim([0 255]);
    grid;
    title('All pixels in RGB');
    xlabel('Red');
    ylabel('Green');
    zlabel('Blue');
     
     
    
    J = ones(length(X),3);
    
    for i = 1 : length(X)
        if classification(i) == 1
           J(i,:) = mu(1,:);
        elseif classification(i) == 2
           J(i,:) = mu(2,:);
        elseif classification(i) == 3
           J(i,:) = mu(3,:);
        elseif classification(i) == 4
           J(i,:) = mu(4,:);
        elseif classification(i) == 5
           J(i,:) = mu(5,:);
        end
    end   
    


    %Find the Xie-Beni Index
    XieBeni=0;
    for i=1:size(X)
        for j=1:c
            if I(i,4)==j
                Denom = sqrt(sum(((mu-repmat(mu(c,:),c,1)).^2),2));
                Denom = sort(Denom);
                Denom = Denom(2);
                XieBeni = XieBeni + ( (norm(I(i,1:3) - mu(j,:)))/Denom);
            end
        end
    end

    disp('Xie-Beni Index Is');
    XieBeni=XieBeni/length(X)
    
    
    Ilabeled = reshape(J,M,N,3);
    figure;
    imshow(uint8(Ilabeled));
    
    

