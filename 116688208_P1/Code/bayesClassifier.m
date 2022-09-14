% Dataset
data_folder = 'C:\Users\harun\Desktop\Main Folder\Work\College\Spring 2022\ENEE436\Projects\Project 1\Project 1/Data/';

%Test Ratio
test_ratio = 0.1;


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


%Preprocessing

vectors1 = my_pca(train_data);
vectors2 = my_pca(test_data);

pcadimensions = 300;

pcatrain = train_data * vectors1(:,1:pcadimensions);
pcatest = test_data * vectors2(:,1:pcadimensions);

[~,ldatrain]=my_lda(pcatrain,train_labels);
[~,ldatest]=my_lda(pcatest,test_labels);



%Set Data to be input into the Bayes' Classifier
Data = ldatrain;
Label = train_labels;
testData = ldatrain;
testLabel = train_labels;


%seperate into two classes of data
    Classes = unique(Label);
    ClassSampleCounter = zeros(1,size(Classes,2));
    ClassSampleCounter(size(Classes,2)) = 0;
    ClassData = [];
    
    for i = 1:size(Data,1)
         for j = 1:size(Classes,2)
            if(labels(i) == Classes(:,j))
                ClassSampleCounter(1,j) = ClassSampleCounter(1,j) + 1;
                ClassData(ClassSampleCounter(1,j),:,j) = Data(i,:);
                
            end
        end
    end
    
    %Due to the incorrect size of ClassData
    %We have to set non-values to NaN
    for j = 1:size(ClassSampleCounter,2)
        ClassData(ClassSampleCounter(1,j)+1:size(ClassData,1),:,j) = nan;
    end
    

    
    
    %Calculate the mean of each class data
    %Each row corresponds to a class
    ClassMean = [];

    for i = 1: size(ClassData,3)
        ClassMean(i,:) = nanmean(ClassData(:,:,i),1);
    end
    
    ClassMean;
    


    %now calculate gaussian covariance by estimation
    %{
    cov1 = (class_one(1,:)-GausMean(2,:)).'*((class_one(1,:)-GausMean(2,:)).') 
    for i = 2:size(class_one,1)
        cov1 = cov1 + (class_one(i,:)-GausMean(2,:))*(class_one(i,:)-GausMean(2,:)).' ;
    end
    %}
    GausCov = [[[]]];
    for i = 1:size(Classes,2) %i = class
        GausCov(:,:,i) = zeros(size(ClassData,2));
        for j = 1:size(ClassData,1) %j = row of single sample Possible error bc taking size of class data
            GausCov(:,:,i) = GausCov(:,:,i) + ((ClassData(j,:,i) - ClassMean(i)).' * (ClassData(j,:,i) - ClassMean(i)));
        end
        GausCov(:,:,i) = (GausCov(:,:,i) / size(ClassData,1)) + eye(size(GausCov(:,:,1))) * .0001;
    end
    
    

ClassData(1,:,1);
ClassMean(2);
testData(1,:) ;


   %Calculate Posteriors
   %GausPost = [size(class_zero,1)/size(Data,1) ; size(class_one,1)/size(Data,1) ]
   
   GausPost = [];
   for i = 1 : size(Classes,2)
       GausPost(i) = ClassSampleCounter(1,i) / size(Data,1);
   end
   
   GausPost;
   
   

%Now that we have the ML estimates of all the variables we need, we can
%calculate the discriminant function

%x = test data point

g = zeros(size(test_data,1),size(Classes,2));
for i = 1 : size(Classes,2)
    
    for j = 1:size(testData,1)
        g(j,i) = -(1/2)*det(GausCov(:,:,i)) - (1/2) * (testData(j,:) - ClassMean(i,:)) * inv(GausCov(:,:,i)) * (testData(j,:) - ClassMean(i,:)).' + log(GausPost(i));
    end
    size(g);
end
   

%now test what the accuracy of g is
correct = 0;
for i = 1:size(g,1)
    [val,ind] = max(g(i,:));
    predictedClass = Classes(ind);
    
    if(predictedClass == testLabel(i))
        correct = correct +1;
    end
end
correct;
accuracy = correct / size(testLabel,2);
