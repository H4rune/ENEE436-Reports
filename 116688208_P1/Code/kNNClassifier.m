clear all
close all



% Dataset
data_folder = 'C:\Users\harun\Desktop\Main Folder\Work\College\Spring 2022\ENEE436\Projects\Project 1\Project 1/Data/';

%Test Ratio
test_ratio = 0.3;


load([data_folder,'data.mat'])
Ns = 200;
face_n = face(:,:,1:3:3*Ns);
face_x = face(:,:,2:3:3*Ns);
face_il = face(:,:,3:3:3*Ns);

i = randi([1,Ns],1);

data = [];
labels = [];
[m,n] = size(face_n(:,:,i));
for subject=1:Ns
    %neutral face: label 0
    face_n_vector = reshape(face_n(:,:,subject),1,m*n);
    data = [data ; face_n_vector];
    labels = [labels 0];
    %face with expression: label 1
    face_x_vector = reshape(face_x(:,:,subject),1,m*n);
    data = [data ; face_x_vector];
    labels = [labels 1];  
end

% Split to train and test data
[data_len,data_size] = size(data);
N = round((1-test_ratio)* data_len);
idx = randperm(data_len);
train_data = data(idx(1:N),:);
train_labels = labels(idx(1:N));
test_data = data(idx(N+1:2*Ns),:);
test_labels = labels(idx(N+1:2*Ns));



%[trash,train_data] = my_lda(train_data,train_labels);
 
%Preprocessing

vectors = my_pca(train_data);
train_data = train_data * vectors(:,1:100);
[trash,train_data] = my_lda(train_data,train_labels);
train_labels;


%[trash,test_data] = my_lda(test_data,test_labels);

vectors = my_pca(test_data);
test_data = test_data * vectors(:,1:100);
[trash,test_data] = my_lda(test_data,test_labels);
test_labels;

%set k value
k = 5;




Classes = unique(train_labels);


correct = 0;

for i = 1:size(test_data)
    sample = test_data(i,:);
    truelabel = test_labels(i);
    
    dist = zeros(size(train_data,1),1);
    
    for j = 1:size(train_data,1)
        dist(j) = norm(sample - train_data(j,:));
        
    end
    %create distance array for all distances to 1 test sample
    dist;
    
    %find the index of the k closest points
    [distance, index] = mink(dist,k);
    train_labels(index);
    
    %loop through each point and record which label is associated
    %labelCounter = [];
    labelCounter(1:size(Classes,2)) = 0;
    
    
    
    for j = 1:size(index,1)
        
        %labelCounter(label(index(i))) = labelCounter(label(index(i))) +1;
        
        %train_labels(index(j))
        
        
        %update label counter at index according to training point label
        for t = 1:size(Classes,2)
            if(Classes(1,t) == train_labels(1,index(j)))
                labelCounter(t) = labelCounter(t) +1;
            end
            
        end
        
        
    end
    
    
    
    [val,ind] = max(labelCounter);
    
    predictedLabel = Classes(ind);
    truelabel;
    
    if(predictedLabel == truelabel)
        correct = correct + 1;
    end
end

accuracy = correct / size(test_labels,2);



