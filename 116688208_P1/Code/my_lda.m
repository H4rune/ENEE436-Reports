function [vectors, newData] = my_lda(data,labels)
    %LDA function for dimensionality reduction
    %Vectors will return the coefficients
    %newData will return data * vectors

    %split the data into sets based on class
    %in my code, this data will be held in Class Data
    %The third dimension (:,:,i) will determine the class
    [samples,variables] = size(data);
    Classes = unique(labels);
    ClassData = zeros(size(data,2));
    
    %ClassSampleCounter is used to keep track of how mnay samples
    %are in each class
    ClassSampleCounter = [];
    ClassSampleCounter(size(Classes,2)) = 0;
    
    
    for i = 1:samples
         for j = 1:size(Classes,2)
            if(labels(i) == Classes(:,j))
                ClassSampleCounter(1,j) = ClassSampleCounter(1,j) + 1;
                ClassData(ClassSampleCounter(1,j),:,j) = data(i,:);
                
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
    
    %Calculate S0, S1, etc for each of the classes
    %This data will be held in the Sx variable with the
    %3rd dimention (:,:,i) corresponding to class
    
    Sx = [];
    
    for i = 1:size(ClassData,3)
        %This New Matrix will eliminate all NaN from the data making it
        %useable
        newMatrix = ClassData(:,:,i);
        newMatrix = newMatrix(any(~isnan(newMatrix),2),:);
        
        %Calculate Scatter here
        Sx(:,:,i) = (size(newMatrix,1) - 1) * cov(newMatrix);
    end 
    
    %With the individual Scatters calculated, find Sw
    %The Sw is the scatter within the classes
    
    Sw = Sx(:,:,1);

    for i = 2:size(Sx,3)
        Sw = Sw + Sx(:,:,i);
    end
    Sw = Sw + eye(size(Sw,1))*.0001;
    
    %Next, we Calculate the Scatter between classes Sb
    
    Sb = zeros(size(Sw));

    for i = 1:size(Classes,2)
        %This New Matrix will eliminate all NaN from the data making it
        %useable
        newMatrix = ClassData(:,:,i);
        newMatrix = newMatrix(any(~isnan(newMatrix),2),:);
        
        %Here is the summation for Sb
        Sb = Sb +  ((ClassMean(i,:) - mean(data,1)).' * (ClassMean(i,:) - mean(data,1)));
    end
   
    %Construct the Covariance from Sb and Sw
    Cov = Sw \ Sb;
    
    %Find the eigen vectors and order them from high to low eigenvalues
    [vectors,values] = eig(Cov);
    [d,ind] = sort(diag(values),'descend');
    vectors = vectors(:,ind);
    
    %Lastly, return only the c-1 colums of the eigenvectors
    %and compute the new data to be returned too
    vectors = vectors(:,1:min([size(Classes,2)-1,size(vectors,2)]));
    newData = data * vectors;
    
end