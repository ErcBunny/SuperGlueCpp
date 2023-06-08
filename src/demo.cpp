#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <SuperGlue.h>

using namespace std;
using namespace cv;

void imshow_superglue(Mat &img_1, Mat &img_2, Mat &img_3)
{
    cout << "===== imshow_superglue =====" << endl;

    Mat img_1_gray, img_2_gray, img_3_gray;
    cvtColor(img_1, img_1_gray, COLOR_BGR2GRAY);
    cvtColor(img_2, img_2_gray, COLOR_BGR2GRAY);
    cvtColor(img_3, img_3_gray, COLOR_BGR2GRAY);

    SuperGlue superglue(
        4,
        0.005,
        -1,
        true,
        20,
        0.67,
        true);

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    vector<float> conf;

    // test forward_full, 2 images
    superglue.forward_full(img_1_gray, img_2_gray, keypoints_1, keypoints_2, matches, conf);
    Mat img1_keypoints;
    drawKeypoints(img_1, keypoints_1, img1_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    // imshow("img1_superpoint", img1_keypoints);

    Mat img_match;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    imshow("img_superglue_full", img_match);

    // test forward_append, get keypoints of the first, then append new frames; reduce redundant keypoint computation
    superglue.get_keypoints(img_1_gray);
    superglue.forward_append(img_2_gray, keypoints_1, keypoints_2, matches, conf);

    Mat img_match_1;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match_1);
    imshow("img_superglue_append_1", img_match_1);

    superglue.forward_append(img_3_gray, keypoints_1, keypoints_2, matches, conf);
    Mat img_match_2;
    drawMatches(img_2, keypoints_1, img_3, keypoints_2, matches, img_match_2);
    imshow("img_superglue_append_2", img_match_2);
}

/*
 * adapted from
 * https://github.com/sunzuolei/orb/blob/master/feature_extration.cpp
 */
void imshow_orb(Mat &img_1, Mat &img_2)
{
    cout << "===== imshow_orb =====" << endl;

    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    std::vector<KeyPoint> keypoints_1, keypoints_2;
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    Mat descriptors_1, descriptors_2;
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    Mat img1_keypoints;
    drawKeypoints(img_1, keypoints_1, img1_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    // imshow("img1_orb", img1_keypoints);

    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);

    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }
    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (matches[i].distance <= max(1.5 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    // imshow("img_match", img_match);
    imshow("img_goodmatch_orb", img_goodmatch);
}

int main(int argc, char **argv)
{
    Mat img_1 = imread("../assets/freiburg_sequence/1341847980.722988.png", IMREAD_COLOR);
    Mat img_2 = imread("../assets/freiburg_sequence/1341847989.802890.png", IMREAD_COLOR);
    Mat img_3 = imread("../assets/freiburg_sequence/1341847995.870641.png", IMREAD_COLOR);

    imshow_superglue(img_1, img_2, img_3);
    imshow_orb(img_1, img_2);

    waitKey(0);
    return 0;
}