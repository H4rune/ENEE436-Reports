function [vectors, newData] = my_pca(data)
    %PCA function for dimensionality reduction
    %Vectors will return the coefficients
    %newData will return data * vectors
    
    %remove the mean from the data
    data_mean_removed = data - mean(data,1);
    
    %next find the covariance of new data
    Cov = cov(data_mean_removed);
    
    %find eigenvectors of covariance matrix
    [vectors,values] = eig(Cov);
    
    %then sort the vectors based on eigenvalues
    [d,ind] = sort(diag(values),'descend');
    vectors = vectors(:,ind);
    
    %calculate newData
    newData = data * vectors;
    
end