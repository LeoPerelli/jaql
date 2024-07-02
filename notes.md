it seems there is no reason to use an a-symmetric interval for the quantised numbers, as this does not prevent "squishing."
at the same time, to save computation you can choose a symmetric interval for the non-quantised numbers. however, if their
distribution is highly imbalanced, you will loose a lot of precision. so in these cases it is either better to use chunking,
or to sacrifice performance gains and just use the asymmetric range (min, max)


the above was called scalar quantization. a more powerful tool when dealing with vectors is product quantization. here, 
you use k means to find centroids and subsitute the vectors with these centroids. normally, you do this on a chunk basis, 
rather than on the whole vector. so you take your dataset of vectors, chunk them up, and for each chunk index you collect all 
chunks from the different vectors. this produces a distribution-aware quantization, that can be more effective than a global 
clustering. at the same time, i am not sure what the advantage of clustering them separately is, when you coould carry out 
the clustering with all the chunks. you could use the same number of centroids as in the disjoint case, and the complexity of
k-means is linear both in n and in k. maybe the advantage is in the compression for the storage, as a higher k means using a 
higher dimensional int. so to save on memory, if they actually are similar, it is more convenient to do many small clusterings, 
rather than one single big one. this is assuming that k*n is constant, but actually its bullshit because in the single 
case i have both bigger n and bigger k. 