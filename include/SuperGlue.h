//
// Created by ryan on 6/6/23.
//

#ifndef SUPERGLUE_SUPERGLUE_H
#define SUPERGLUE_SUPERGLUE_H

#include <string>
#include <Python.h>
#include <opencv2/core/core.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

class SuperGlue
{
public:
    SuperGlue(
        int nms_radius,
        float keypoint_threshold,
        int max_keypoints,
        bool weights_indoor,
        int sinkhorn_iterations,
        float match_threshold,
        bool use_cuda,
        const std::string &module_path);

    ~SuperGlue();

    void set_config(
        int nms_radius,
        float keypoint_threshold,
        int max_keypoints,
        int sinkhorn_iterations,
        float match_threshold);

    void get_init_keypoints(cv::InputArray image);

    void forward_full(
        cv::InputArray image_0,
        cv::InputArray image_1,
        std::vector<cv::KeyPoint> &keypoints_0,
        std::vector<cv::KeyPoint> &keypoints_1,
        std::vector<cv::DMatch> &matches,
        std::vector<float> &confidence);

    void forward_append(
        cv::InputArray image,
        std::vector<cv::KeyPoint> &keypoints_0,
        std::vector<cv::KeyPoint> &keypoints_1,
        std::vector<cv::DMatch> &matches,
        std::vector<float> &confidence);

    void get_keypoints(
        cv::InputArray image,
        std::vector<cv::KeyPoint> &keypoints,
        std::vector<float> &keypoint_scores,
        cv::Mat &descriptors,
        cv::Mat &frame_tensor);

    void match_keypoints(
        std::vector<cv::KeyPoint> &keypoints0,
        std::vector<float> &keypoint_scores0,
        cv::InputArray descriptors0,
        cv::InputArray frame_tensor0,
        std::vector<cv::KeyPoint> &keypoints1,
        std::vector<float> &keypoint_scores1,
        cv::InputArray descriptors1,
        cv::InputArray frame_tensor1,
        std::vector<cv::DMatch> &matches,
        std::vector<float> &confidence);


private:
    PyObject *py_module_SuperGlueWrapper{};

    PyObject *py_class_SuperGlueWrapper{};
    PyObject *py_obj_SuperGlueWrapper{};

    PyObject *py_func_set_config{};
    PyObject *py_func_get_init_keypoints{};
    PyObject *py_func_forward_full{};
    PyObject *py_func_forward_append{};
    PyObject *py_func_get_keypoints{};
    PyObject *py_func_match{};

    PyObject *py_args_set_config{};
    PyObject *py_args_get_init_keypoints{};
    PyObject *py_args_forward_full{};
    PyObject *py_args_forward_append{};
    PyObject *py_arg_get_keypoint{};
    PyObject *py_arg_match{};

    static PyObject *grayim_to_py_array(cv::InputArray image);

    static void unpack_results(
        PyObject *ret,
        std::vector<cv::KeyPoint> &keypoints_0,
        std::vector<cv::KeyPoint> &keypoints_1,
        std::vector<cv::DMatch> &matches,
        std::vector<float> &confidence);

};

#endif // SUPERGLUE_SUPERGLUE_H
