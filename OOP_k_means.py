
"""
Implementation of K-means algorithm
"""
import pandas
import numpy as np
import matplotlib.pyplot as plt

"""
initializing data point set
"""
def initialize_datapoint_set(dataset):
    datapointset = [];
    class datapoint:
        def __init__(self,attributes,cluster_ref):
            self.attr =  attributes
            self.ref = cluster_ref

        def getAttributes(self) :
            return self.attr;

        def getClusterReferrence(self) :
            return self.ref;

        def setORupateClusterReferrence(self, cluster_ref):
            self.ref = cluster_ref;
   
    for indrow in dataset[:len(dataset)]:
        cluster_referrence = 0;
        point = datapoint(indrow, cluster_referrence);
        datapointset.append(point);
    
    return datapointset;

"""
initializing clusters
"""
def initialize_cluster_set(centroids):
    clusterset = [];
    class cluster :
        def __init__(self, centroid) :
            self.center = centroid

        def getCentroid(self) :
            return self.center;

        def setORupdateCentroid(self, centroid) :
            self.center = centroid;
            
    for center in centroids :
        clusterset.append(cluster(center));
     
    return clusterset;

"""
calculating distance between datapoints and center
"""
def calculate_distance(point_feature_vector, center_vector):
    return np.sqrt((np.sum((point_feature_vector - center_vector)**2)));

"""
assigning each points to a particular cluster
"""
def assign_points_to_cluster(datapointset, clusterset, performance_vector):
    
    dist_of_points = [];
    for points in datapointset[:len(datapointset)]:
        min_distance = i = 0;
        distances_vec = [];
        for cluster in clusterset:
            distance = calculate_distance(points.getAttributes(), cluster.getCentroid());
            distances_vec.append(distance);
            if (i == 0) :
                min_distance = distance;
                points.setORupateClusterReferrence(i);
            elif (distance < min_distance) :
                min_distance = distance;
                points.setORupateClusterReferrence(i);          
            i +=1
        dist_of_points.append(min_distance);
    # np.min(np.sum(distances_vec))
    
    performance_vector.append(np.sum(dist_of_points));

"""
update centroids each cluster
"""
def update_centroids(datapointset, clusterset, k):
    close_to_stop = 0;
    to_continue = True;
    clusterpoints = {};
    centroids_arr = [];
    for i in range(k):
        clusterpoints[i] = [];

    """  
    for each point moving a particular point to a cluster
    """
    for points in datapointset[:len(datapointset)]:
        clusterref = points.getClusterReferrence();
        for key in clusterpoints.keys() :
            if clusterref == key :
                clusterpoints[key].append(points.getAttributes())
                break;
    
    """      
    improvising mean of each cluster
    """
    for key in clusterpoints.keys() :
        clastervalueset  = np.array(clusterpoints.get(key));
        if len(clastervalueset) != 0 :
            new_mean = np.mean(clastervalueset, axis = 0)
            prev_centroid = (clusterset[key]).getCentroid();
            if np.allclose(prev_centroid,new_mean) :
                close_to_stop+=1;
            (clusterset[key]).setORupdateCentroid(new_mean)
    
    if close_to_stop == k :
        to_continue = False;
        for indx in clusterpoints.keys() :
            centroids_arr.append((clusterset[indx]).getCentroid())
        
    return to_continue, centroids_arr;

"""
k - means algorithm
"""
def k_means_algo(dataset, k):
    continue_iterration = True;
    centroids = dataset[np.random.choice(range(dataset.shape[0]), k, replace=False),:]
    performance_vector = [];
    
    """
    randomly initializing centroids depending on number of clusters (i.e. k)
    """
    datapointset = initialize_datapoint_set(dataset);
    clusterset = initialize_cluster_set(centroids);
 
 
    while continue_iterration :
        assign_points_to_cluster(datapointset, clusterset, performance_vector);
        continue_iterration,center_arr= update_centroids(datapointset, clusterset, k);
    
    return np.array(center_arr), np.array(performance_vector);

#d = pandas.read_csv(open('/s/chopin/k/grad/amchakra/Documents/Machine Learning/Machine Learning Sample Data/iris.data'))

#final_data = np.array(d.iloc[:(len(d) - 1),:2].values)

f = open("**** Input File Path ****","r")
data = np.loadtxt(f ,delimiter=' ', usecols=range(0,9))

final_data = data[:,1:5]

print('final ', final_data)
performance_vectr = [];


# [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38] # , 36, 40, 42
sample_arr = range(10); 
for i in sample_arr:
    vectr = [];
    for j in range(4):
        centers, performance_vec = k_means_algo(final_data, (i+2))
        vectr.append(np.min(performance_vec));
    performance_vectr.append((np.mean(vectr)))
    # i+=1;


# centers, performance_vec = k_means_algo(final_data, 3)

# print('performance vec ', performance_vec)
# plotting performance vector only 
plt.plot(sample_arr, performance_vectr)
plt.xlabel('K');
plt.ylabel('Performance Function (J)')
# plt.bar(sample_arr, np.array(performance_vectr))
# plt.plot(performance_vec);
# plt.xlabel('Iterations');
# plt.ylabel('Performance Function (J)');
"""
i = 2;
while i < 8 :
    lable_nm = 'for k = ',i
    print('label ', i)
    centers, performance_vec = k_means_algo(final_data, i)
    plt.plot(performance_vec,label = lable_nm);
    i+=1;
"""
# plotting scatter diagram on points and final set of centers
# plt.scatter(final_data[:,0], final_data[:,1], s=80,c="green",alpha=0.5, label = 'Data Points');
# plt.scatter(centers[:,0], centers[:,1], s=80,c="red",alpha=0.5, label = 'Centroids');
# plt.xlabel('sepal length');
# plt.ylabel('sepal width');
# plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
plt.savefig("k_means_result.png")
plt.show();

