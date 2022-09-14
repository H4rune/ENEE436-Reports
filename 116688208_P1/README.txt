Hello!

To run this code it is advised that you have all the files in one place. my_lda.m and my_pca.m are functions
that the other files call so they have to be within the same folder for that to work.
These files correspond to LDA/MDA and PCA respectively. They both are well commented if you are
interested in how they function. 

There are a total of 4 files corresponding to either the Bayes Classifer or k-NN Classifier. The .m 
and .mlx files contain the same code just in different file formats.

To run either of these codes, you should replace the data_folder variable at the top of the script.
I don't have any of the scripts set to output anything. Rather, I simply unsuppressed (removed semicolon)
certain variables and had them print out. I did this for the testing accuracies(found near the 
bottom of the script). In order to find the accuracies or any other data the same will need to be done.
I also left a section in each script called preprocessing. This is where I conducted the preproccessing
before feeding the data into the classifier. In order to get the same results as me, you may need to adjust
this section.

Lastly, all the codes currently load neutral vs facial expression data. If you want to run a different
data set on my code you can. You just have to load the data from load_data.m and then run the classification
code after that.