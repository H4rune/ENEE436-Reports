%% ENEE 436: Project 1 -- Loading the dataset in MATLAB
clear all
close all
%% Problem Parameters

% Dataset
data_folder = 'C:\Users\harun\Desktop\Main Folder\Work\College\Spring 2022\ENEE436\Projects\Project 1\Project 1/Data/';

%Test Ratio
test_ratio = 0.2;

%% Load Face Data
load([data_folder,'data.mat'])
Ns = 200;
face_n = face(:,:,1:3:3*Ns);
face_x = face(:,:,2:3:3*Ns);
face_il = face(:,:,3:3:3*Ns);

figure,
sgtitle('DATA: Example of the 3 images of a random subject')
if true
    i = randi([1,Ns],1);
    subplot(1,3,1),imshow(face_n(:,:,i))
    subplot(1,3,2),imshow(face_x(:,:,i))
    subplot(1,3,3),imshow(face_il(:,:,i))   
end

% Convert the dataset in data vectors and labels for
% netutral vs facil expression classification

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


%% Load Pose Data

load([data_folder,'pose.mat'])
[rows,columns,images,subjects]= size(pose);

% Show some examples of dataset
if true
    figure,
    sgtitle('POSE: Example of 3 images of a random subject')
    s = randi([1,subjects],1);
    subplot(1,3,1),imshow(uint8(pose(:,:,1,s)))
    subplot(1,3,2),imshow(uint8(pose(:,:,2,s)))
    subplot(1,3,3),imshow(uint8(pose(:,:,3,s)))   
end

% Convert the datase in data vectors and labels for subject identification
data = [];
labels = [];
for s=1:subjects
    for i=1:images
        pose_vector = reshape(pose(:,:,i,s),1,rows*columns);
        data = [data;pose_vector];
        labels = [labels s];        
    end
end

% Split to train and test data
[data_len,data_size] = size(data);
N = round((1-test_ratio)* data_len);
idx = randperm(data_len);
train_data = data(idx(1:N),:);
train_labels = labels(idx(1:N));
test_data = data(idx(N+1:data_len),:);
test_labels = labels(idx(N+1:data_len));

%% Load Illumination Data

load([data_folder,'illumination.mat'])
[data_size,images,subjects] = size(illum);

% Show some examples of the dataset
if true
    figure,
    sgtitle('ILLUMINATION: Example of 3 images of a random subject')
    s = randi([1,subjects],1);
    subplot(1,3,1),imshow(uint8(reshape(illum(:,1,s),[48 40])))
    subplot(1,3,2),imshow(uint8(reshape(illum(:,2,s),[48 40])))
    subplot(1,3,3),imshow(uint8(reshape(illum(:,3,s),[48 40])))   
end

% Convert the datase in data vectors and labels for subject identification
data = [];
labels = [];
for s=1:subjects
    for i=1:images
        data = [data;illum(:,i,s)'];
        labels = [labels s];        
    end
end

% Split to train and test data
[data_len,data_size] = size(data);
N = round((1-test_ratio)* data_len);
idx = randperm(data_len);
train_data = data(idx(1:N),:);
train_labels = labels(idx(1:N));
test_data = data(idx(N+1:data_len),:);
test_labels = labels(idx(N+1:data_len));


