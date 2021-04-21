// PCL lib Functions for processing point clouds 

#include "processPointClouds.h"
#include <unordered_set>
#include "kdtree3D.h"



//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint, Eigen::Vector4f RoofminPoint, Eigen::Vector4f RoofmaxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // TODO:: Fill in the function to do voxel grid point reduction and region based filtering
    // Create the filtering object
    typename pcl::VoxelGrid<PointT> sor;
    typename pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
    sor.setInputCloud (cloud);
    sor.setLeafSize (filterRes, filterRes, filterRes);
    sor.filter (*cloud_filtered);

    typename pcl::PointCloud<PointT>::Ptr cloud_cropped (new pcl::PointCloud<PointT>);
    typename pcl::CropBox<PointT> crop(true );
    crop.setInputCloud(cloud_filtered);
    crop.setMin(minPoint);
    crop.setMax(maxPoint);
    crop.filter(*cloud_cropped); 

    typename pcl::PointCloud<PointT>::Ptr cloud_cropped_noRoof (new pcl::PointCloud<PointT>);
    typename pcl::CropBox<PointT> noRoof(false );
    noRoof.setInputCloud(cloud_cropped);
    noRoof.setMin(RoofminPoint);
    noRoof.setMax(RoofmaxPoint);
    noRoof.setNegative(true);
    noRoof.filter(*cloud_cropped_noRoof); 



    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloud_cropped_noRoof;

}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud) 
{
  // TODO: Create two new point clouds, one cloud with obstacles and other with segmented plane
    typename pcl::ExtractIndices<PointT> extract;
    typename pcl::PointCloud<PointT>::Ptr plane(new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr obstacles(new pcl::PointCloud<PointT>());
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*plane);
    extract.setNegative (true);
    extract.filter (*obstacles);
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstacles, plane);
    return segResult;
}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();
    // segmentPlane using PCL Library function 
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    // Create the segmentation object
    typename pcl::SACSegmentation<PointT> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (maxIterations);
    seg.setDistanceThreshold (distanceThreshold);
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
    }
    // seperate the points into two pointclouds
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers,cloud);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;
    return segResult;
}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane_RANSC(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // SegmentPlane using self-built RANSC Code
    std::unordered_set<int> inliersResult;
    srand(time(NULL));
    
    // find the indices for the biggest plane
    while(maxIterations--){
    // For max iterations, Randomly sample subset and fit plane
        std::unordered_set<int> inliers;
        while(inliers.size()<3)
          inliers.insert(rand()%(cloud->points.size()));
        float x1, y1, z1, x2, y2, z2, x3, y3, z3;
        auto itr = inliers.begin();
        x1 = cloud->points[*itr].x;
        y1 = cloud->points[*itr].y;
        z1 = cloud->points[*itr].z;
        itr++;
        x2 = cloud->points[*itr].x;
        y2 = cloud->points[*itr].y;
        z2 = cloud->points[*itr].z;
        itr++;
        x3 = cloud->points[*itr].x;
        y3 = cloud->points[*itr].y;
        z3 = cloud->points[*itr].z;

        float i = (y2-y1)*(z3-z1)-(z2-z1)*(y3-y1);
        float j = (z2-z1)*(x3-x1)-(x2-x1)*(z3-z1);
        float k = (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);

        float d = -(i*x1 + j*y1 + k*z1);
        // Measure distance between every point and fitted line
        for (int index=0; index<(cloud->points.size());index++){
          if(inliers.count(index)>0)
            continue;

          float x4 = cloud->points[index].x;
          float y4 = cloud->points[index].y;
          float z4 = cloud->points[index].z;
          float dist = fabs(i*x4 + j*y4 + k*z4 + d)/sqrt(i*i + j*j + k*k);
        // If distance is smaller than threshold count it as inlier
          if(dist<=distanceThreshold)
            inliers.insert(index);
        }
        if(inliers.size()>inliersResult.size())
          inliersResult=inliers;
    }
    // seperate the points into two pointclouds
    typename pcl::PointCloud<PointT>::Ptr  cloudInliers(new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr cloudOutliers(new pcl::PointCloud<PointT>());

    for(int index = 0; index < cloud->points.size(); index++)
    {
        PointT point = cloud->points[index];
        if(inliersResult.count(index))
            cloudInliers->points.push_back(point);
        else
            cloudOutliers->points.push_back(point);
    }
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    //std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers,cloud);
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(cloudOutliers,cloudInliers);
    return segResult;
}

template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    //Clustering using PCL Libraty function
    // Creating the KdTree object for the search method of the extraction
    typename pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud (cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    typename pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (clusterTolerance); 
    ec.setMinClusterSize (minSize);
    ec.setMaxClusterSize (maxSize);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
      typename pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);
      for (const auto& idx : it->indices){
        //cloud_cluster->points.push_back (cloud->points[idx]); 
        cloud_cluster->push_back ((*cloud)[idx]);
      }
      cloud_cluster->width = cloud_cluster->size ();
      cloud_cluster->height = 1;
      cloud_cluster->is_dense = true;
      clusters.push_back(cloud_cluster);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;
    return clusters;
}

template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering_euclidean(typename pcl::PointCloud<PointT>::Ptr & cloud, float clusterTolerance, int minSize, int maxSize)
{
    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();
    // Clusterung using self-built Code
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    //Create KdTree
    KdTree* tree = new KdTree;
    for (int i=0; i<cloud->points.size(); i++) {
        std::vector<float> point_to_insert = {cloud->points[i].x, cloud->points[i].y, cloud->points[i].z};
        tree->insert(point_to_insert, i); 
    }
    //Search clusters using the KdTree
    //std::vector<std::vector<float>> points;
    std::vector<std::vector<int>> clusters_indices = euclideanCluster(cloud, tree, clusterTolerance, minSize, maxSize);

    // using ExtractIndices
    // for(std::vector<int> cluster_indices : clusters_indices)
    // {
    //     pcl::PointIndices::Ptr inliers (new pcl::PointIndices());
    //     for (int point_indice : cluster_indices){
    //         inliers->indices.push_back(point_indice);
    //     }
    //     typename pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>());
    //     pcl::ExtractIndices<PointT> extract;
    //     extract.setInputCloud(cloud);
    //     extract.setIndices(inliers);
    //     extract.setNegative(false);
    //     extract.filter(*cluster);
    //     clusters.push_back(cluster);
    // }

    for(std::vector<int> cluster_indices : clusters_indices){
        typename pcl::PointCloud<PointT>::Ptr clusterCloud(new pcl::PointCloud<PointT>());
        for (int point_indice : cluster_indices){
            clusterCloud->points.push_back(cloud->points[point_indice]);
        }
        clusters.push_back(clusterCloud);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}

template<typename PointT>
std::vector<std::vector<int>> ProcessPointClouds<PointT>::euclideanCluster(typename pcl::PointCloud<PointT>::Ptr & cloud, KdTree* tree, float distanceTol, int minSize, int maxSize)
{

    // eturn list of indices for each cluster
    std::vector<std::vector<int>> clusters;
    std::vector<bool> processed(cloud->points.size(),false);
    int i=0;
    while(i<cloud->points.size()){
        if(processed[i]){
        i++;
        continue;
        }

        std::vector<int> cluster;
        clusterHelper(i,cloud,cluster,processed,tree,distanceTol);
        if (cluster.size()>=minSize && cluster.size()<=maxSize)
            clusters.push_back(cluster);
        i++;
    }
    return clusters;
}

template<typename PointT>
void ProcessPointClouds<PointT>::clusterHelper(int i, const typename pcl::PointCloud<PointT>::Ptr & cloud, std::vector<int>& cluster, std::vector<bool>& processed, KdTree* tree, float distanceTol){
    processed[i]=true;
    cluster.push_back(i);
    std::vector<int> nearby = tree->search(std::vector<float>{cloud->points[i].x, cloud->points[i].y, cloud->points[i].z}, distanceTol);
    for(int j:nearby){
        if(!processed[j])   
        clusterHelper(j,cloud,cluster,processed,tree,distanceTol);
    }
}


template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}


template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}


template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}