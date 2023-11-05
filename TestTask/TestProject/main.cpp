#include <iostream>
#include <math.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/progressive_morphological_filter.h>

//#include <pcl/ModelCoefficients.h>
//#include <pcl/sample_consensus/model_types.h>
//#include <pcl/sample_consensus/method_types.h>
//#include <pcl/segmentation/sac_segmentation.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/hal/interface.h>

using namespace pcl;
//using namespace visualization;
using namespace std;
using namespace cv;

void visualizePC(PointCloud<PointXYZ>::Ptr cloud) {
    /*
        Just a single cloud visualization
    */
    visualization::CloudViewer viewer("Simple Cloud Viewer");
    viewer.CloudViewer::showCloud(cloud);
    while (!viewer.wasStopped()) {}
}


void printPointsInfo(PointCloud<PointXYZ>::Ptr cloud) {
    cout << "Points number: " << size(cloud->points) << endl;
    PointXYZ minCoords, maxCoords;
    getMinMax3D(*cloud, minCoords, maxCoords);
    cout << "Points coordinate ranges:\n"
        << "X: " << minCoords.x << " " << maxCoords.x << "\n"
        << "Y: " << minCoords.y << " " << maxCoords.y << "\n"
        << "Z: " << minCoords.z << " " << maxCoords.z << endl;
}


void trimPC_ROI(PointCloud<PointXYZ>::Ptr cloud, float xHalfROI, float yHalfROI) {
    /*
        x and y parameters are used to trim cloud between (-x, x) and (-y, y) - horizontal ROI trimming
    Maybe z trimming also makes sense, but it's not required according to the assignment
    The source cloud is trimmed directly without new PC creation
    */

    PassThrough<PointXYZ> pass;
    // X trimming
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-xHalfROI, xHalfROI);
    pass.filter(*cloud);
    // Y trimming
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-yHalfROI, yHalfROI);
    pass.filter(*cloud);
    //// Z trimming (the simplest but far from complete groung filtration)
    //pass.setInputCloud(cloud);
    //pass.setFilterFieldName("z");
    //pass.setFilterLimits(-1.0, 10.0);
    //pass.filter(*cloud);

    printPointsInfo(cloud);
    visualizePC(cloud);
}


PointCloud<PointXYZ>::Ptr getGroundFilteredPC(const PointCloud<PointXYZ>::Ptr cloud) {
    /*
        Filtration of ground in the PC provided. Progressive morphlogical filter is used.
    */
    PointCloud<PointXYZ>::Ptr groundFilteredCloud(new PointCloud<PointXYZ>);
    PointIndicesPtr groundIds(new PointIndices);

    // PMF ground filtration
    ProgressiveMorphologicalFilter<PointXYZ> pmf;
    pmf.setInputCloud(cloud);
    //pmf.setSlope();
    pmf.setInitialDistance(0.1f);
    //pmf.setMaxDistance();
    pmf.setCellSize(0.5f);
    pmf.extract(groundIds->indices);

    // Extraction via indices
    ExtractIndices<PointXYZ> extractor;
    extractor.setInputCloud(cloud);
    extractor.setIndices(groundIds);
    extractor.setNegative(true);
    extractor.filter(*groundFilteredCloud);

    printPointsInfo(groundFilteredCloud);
    visualizePC(groundFilteredCloud);

    io::savePCDFileBinary("../../clouds/ground_filtered_cloud.pcd", *groundFilteredCloud);
    std::cerr << "Saved " << size(groundFilteredCloud->points) << " data points to ground_filtered_cloud.pcd" << endl;

    return groundFilteredCloud;
}

void generateBEV(PointCloud<PointXYZ>::Ptr cloud, int height, int width, float cellSize, float xHalfROI, float yHalfROI) {
    /*
        Generates BEV image for provided PC cloud. 
    height and width are pixel size for an output image.
    sellSize is a side length of a square area which is matched to a pixel.
    xROI and yROI are x and y coordinate ranges of the provided PC cloud.
    TODO: Actually, the necessity of heigth/width and xHalfROI/yHalfROI with cellSize proportional match check is required,
    but for now it is assumed that ROI is square-shaped as long as the output BEV-image.
    */
    Mat imgMat = Mat::zeros(height, width, CV_8U); // BEV-image matrix
    Mat imgPtsPerPixel = Mat::zeros(height, width, CV_BIG_UINT(32)); // Density matrix

    float x = 0.0, y = 0.0;
    unsigned int idLine = 0, idCol = 0;
    unsigned int maxDensity = 0;


    // Pixel-wise density calculation
    for (size_t i = 0; i < size(cloud->points); i++) {
        // Coordinates with half-ROI offsets for each point to be positive
        if (cloud->points[i].x != 0.0 && cloud->points[i].y != 0.0) {
            x = cloud->points[i].x + xHalfROI;
            y = cloud->points[i].y + yHalfROI;
            idLine = (unsigned int)(height - round(y / cellSize));
            idCol = (unsigned int)round(x / cellSize) - 1;
            imgPtsPerPixel.at<uint>(idLine, idCol)++;
            if (imgPtsPerPixel.at<uint>(idLine, idCol) > maxDensity) {
                maxDensity++;
            }
        }
    }

    // Filling the output BEV-image pixel-by-pixel proportionally with normalization 0-maxDensity -> 0-255
    for (int ln = 0; ln < height; ln++) {
        for (int cl = 0; cl < width; cl++) {
            imgMat.at<uchar>(ln, cl) = (unsigned char)round(imgPtsPerPixel.at<uint>(ln, cl) * 255.0 / maxDensity);
        }
    }

    circle(imgMat, Point((unsigned int)round(width / 2), (unsigned int)round(height / 2)), (unsigned int)round(height / 2), Scalar(127));
    drawMarker(imgMat, Point((unsigned int)round(width / 2), (unsigned int)round(height / 2)), Scalar(127), MARKER_CROSS, 2);

    imshow("BEV-image", imgMat);
    waitKey(0);

    // Save output BEV-img
    /*bool result = false;
    try
    {
        result = imwrite("../../img/outBEV.png", imgMat);
    }
    catch (const cv::Exception &ex)
    {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    }
    if (result)
        printf("Saved PNG file.\n");
    else
        printf("ERROR: Can't save PNG file.\n");*/
}


int main(int argc, char** argv) {
    PointCloud<PointXYZ>::Ptr sourceCloud(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr groundFilteredCloud(new PointCloud<PointXYZ>);

    if (io::loadPCDFile<PointXYZ>("../../clouds/test_cloud.pcd", *sourceCloud) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read file test_cloud.pcd \n");
        return (-1);
    }

    std::cout << "Loaded "
        << sourceCloud->width * sourceCloud->height
        << " data points from test_cloud.pcd"
        << std::endl;
    printPointsInfo(sourceCloud);
    visualizePC(sourceCloud);
    trimPC_ROI(sourceCloud, 100.0, 100.0);
    groundFilteredCloud = sourceCloud;
    groundFilteredCloud = getGroundFilteredPC(sourceCloud);
    generateBEV(groundFilteredCloud, 200, 200, 1.0, 100.0, 100.0);
    generateBEV(groundFilteredCloud, 400, 400, 0.5, 100.0, 100.0);
    generateBEV(groundFilteredCloud, 800, 800, 0.25, 100.0, 100.0);

    return 0;
}