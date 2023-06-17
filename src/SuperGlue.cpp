//
// Created by ryan on 6/6/23.
//

#include <string>
#include <SuperGlue.h>
#include <Python.h>
#include <opencv2/core/core.hpp>
#include <numpy/arrayobject.h>

SuperGlue::SuperGlue(int nms_radius, float keypoint_threshold, int max_keypoints, bool weights_indoor,
                     int sinkhorn_iterations, float match_threshold, bool use_cuda)
{
    Py_Initialize();
    _import_array();

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../models')");
    py_module_SuperGlueWrapper = PyImport_ImportModule("SuperGlueWrapper");
    py_class_SuperGlueWrapper = PyObject_GetAttrString(py_module_SuperGlueWrapper, "SuperGlueWrapper");

    PyObject *py_args_init = PyTuple_New(7);
    PyTuple_SetItem(py_args_init, 0, Py_BuildValue("i", nms_radius));
    PyTuple_SetItem(py_args_init, 1, Py_BuildValue("f", keypoint_threshold));
    PyTuple_SetItem(py_args_init, 2, Py_BuildValue("i", max_keypoints));
    if (weights_indoor)
        PyTuple_SetItem(py_args_init, 3, Py_BuildValue("s", "indoor"));
    else
        PyTuple_SetItem(py_args_init, 3, Py_BuildValue("s", "outdoor"));
    PyTuple_SetItem(py_args_init, 4, Py_BuildValue("i", sinkhorn_iterations));
    PyTuple_SetItem(py_args_init, 5, Py_BuildValue("f", match_threshold));
    if (use_cuda)
        PyTuple_SetItem(py_args_init, 6, Py_BuildValue("s", "cuda"));
    else
        PyTuple_SetItem(py_args_init, 6, Py_BuildValue("s", "cpu"));
    py_obj_SuperGlueWrapper = PyEval_CallObject(py_class_SuperGlueWrapper, py_args_init);

    py_func_set_config = PyObject_GetAttrString(py_obj_SuperGlueWrapper, "set_config");
    py_func_get_init_keypoints = PyObject_GetAttrString(py_obj_SuperGlueWrapper, "get_init_keypoints");
    py_func_forward_full = PyObject_GetAttrString(py_obj_SuperGlueWrapper, "forward_full");
    py_func_forward_append = PyObject_GetAttrString(py_obj_SuperGlueWrapper, "forward_append");
    py_func_get_keypoints = PyObject_GetAttrString(py_obj_SuperGlueWrapper, "get_keypoints");
    py_func_match = PyObject_GetAttrString(py_obj_SuperGlueWrapper, "match");

    py_args_set_config = PyTuple_New(5);
    py_args_forward_full = PyTuple_New(2);
    py_arg_match = PyTuple_New(8);
}

void SuperGlue::set_config(int nms_radius, float keypoint_threshold, int max_keypoints, int sinkhorn_iterations,
                           float match_threshold)
{
    PyTuple_SetItem(py_args_set_config, 0, Py_BuildValue("i", nms_radius));
    PyTuple_SetItem(py_args_set_config, 1, Py_BuildValue("f", keypoint_threshold));
    PyTuple_SetItem(py_args_set_config, 2, Py_BuildValue("i", max_keypoints));
    PyTuple_SetItem(py_args_set_config, 3, Py_BuildValue("i", sinkhorn_iterations));
    PyTuple_SetItem(py_args_set_config, 4, Py_BuildValue("f", match_threshold));
    PyObject_CallObject(py_func_set_config, py_args_set_config);
}

void SuperGlue::get_init_keypoints(cv::InputArray image)
{
    py_args_get_init_keypoints = Py_BuildValue("(O)", grayim_to_py_array(image));
    PyObject_CallObject(py_func_get_init_keypoints, py_args_get_init_keypoints);
}

void SuperGlue::forward_full(cv::InputArray image_0, cv::InputArray image_1, std::vector<cv::KeyPoint> &keypoints_0,
                             std::vector<cv::KeyPoint> &keypoints_1, std::vector<cv::DMatch> &matches,
                             std::vector<float> &confidence)
{
    PyTuple_SetItem(py_args_forward_full, 0, Py_BuildValue("O", grayim_to_py_array(image_0)));
    PyTuple_SetItem(py_args_forward_full, 1, Py_BuildValue("O", grayim_to_py_array(image_1)));
    PyObject *ret = PyObject_CallObject(py_func_forward_full, py_args_forward_full);

    unpack_results(ret, keypoints_0, keypoints_1, matches, confidence);
}

void SuperGlue::forward_append(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints_0,
                               std::vector<cv::KeyPoint> &keypoints_1, std::vector<cv::DMatch> &matches,
                               std::vector<float> &confidence)
{
    py_args_forward_append = Py_BuildValue("(O)", grayim_to_py_array(image));
    PyObject *ret = PyObject_CallObject(py_func_forward_append, py_args_forward_append);
    unpack_results(ret, keypoints_0, keypoints_1, matches, confidence);
}

PyObject *SuperGlue::grayim_to_py_array(cv::InputArray image)
{
    const int ndims = image.channels() > 1 ? 3 : 2;
    npy_intp dims[3];
    dims[0] = image.rows();
    dims[1] = image.cols();
    dims[2] = image.channels();
    PyObject *array = PyArray_SimpleNewFromData(ndims, dims, NPY_UINT8, image.getMat().data);
    return array;
}

void SuperGlue::unpack_results(PyObject *ret, std::vector<cv::KeyPoint> &keypoints_0,
                               std::vector<cv::KeyPoint> &keypoints_1, std::vector<cv::DMatch> &matches,
                               std::vector<float> &confidence)
{
    keypoints_0.clear();
    keypoints_1.clear();
    matches.clear();
    confidence.clear();

    PyObject *ret_kpts0;
    PyObject *ret_kpts1;
    PyObject *ret_matches;
    PyObject *ret_confidence;
    PyArg_ParseTuple(ret, "OOOO", &ret_kpts0, &ret_matches, &ret_confidence, &ret_kpts1);

    auto *kpts0_array = reinterpret_cast<PyArrayObject *>(ret_kpts0);
    auto *kpts1_array = reinterpret_cast<PyArrayObject *>(ret_kpts1);
    auto *matches_array = reinterpret_cast<PyArrayObject *>(ret_matches);
    auto *confidence_array = reinterpret_cast<PyArrayObject *>(ret_confidence);

    auto *kpts0_data = static_cast<float *>(PyArray_DATA(kpts0_array));
    auto *kpts1_data = static_cast<float *>(PyArray_DATA(kpts1_array));
    auto *matches_data = static_cast<long *>(PyArray_DATA(matches_array));
    auto *confidence_data = static_cast<float *>(PyArray_DATA(confidence_array));

    npy_intp *kpts0_shape = PyArray_SHAPE(kpts0_array);
    npy_intp *kpts1_shape = PyArray_SHAPE(kpts1_array);

    for (int i = 0; i < kpts0_shape[0]; i++)
    {
        keypoints_0.emplace_back(kpts0_data[2 * i], kpts0_data[2 * i + 1], 1);
        confidence.emplace_back(confidence_data[i]);
        if (matches_data[i] != -1)
        {
            matches.emplace_back(i, matches_data[i], 1.0 - confidence_data[i]);
        }
    }

    for (int i = 0; i < kpts1_shape[0]; i++)
    {
        keypoints_1.emplace_back(kpts1_data[2 * i], kpts1_data[2 * i + 1], 1);
    }
}

void SuperGlue::get_keypoints(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints,
                              std::vector<float> &keypoint_scores, cv::Mat &descriptors, cv::Mat &frame_tensor) {
    keypoints.clear();
    keypoint_scores.clear();

    py_arg_get_keypoint = Py_BuildValue("(O)", grayim_to_py_array(image));
    PyObject *ret = PyObject_CallObject(py_func_get_keypoints, py_arg_get_keypoint);

    PyObject *ret_kpts;
    PyObject *ret_scores;
    PyObject *ret_des;
    PyObject *ret_frametensor;
    PyArg_ParseTuple(ret, "OOOO", &ret_kpts, &ret_scores, &ret_des, &ret_frametensor);

    auto *kpts_array = reinterpret_cast<PyArrayObject *>(ret_kpts);
    auto *scores_array = reinterpret_cast<PyArrayObject *>(ret_scores);
    auto *des_array = reinterpret_cast<PyArrayObject *>(ret_des);
    auto *frametensor_array = reinterpret_cast<PyArrayObject *>(ret_frametensor);

    auto *kpts_data = static_cast<float *>(PyArray_DATA(kpts_array));
    auto *scores_data = static_cast<float *>(PyArray_DATA(scores_array));
    auto *des_data = static_cast<float *>(PyArray_DATA(des_array));
    auto *frametensor_data = static_cast<float *>(PyArray_DATA(frametensor_array));

    npy_intp *kpts_shape = PyArray_SHAPE(kpts_array);
    npy_intp *des_shape = PyArray_SHAPE(des_array);
    npy_intp *frametensor_shape = PyArray_SHAPE(frametensor_array);

    cv::Mat des((int)(kpts_shape[0]), (int)(des_shape[0]), CV_32F);
    cv::Mat ten(image.rows(), image.cols(), CV_32F);
    for (int i = 0; i < kpts_shape[0]; i++)
    {
        keypoints.emplace_back(kpts_data[2 * i], kpts_data[2 * i + 1], 1);
        keypoint_scores.emplace_back(scores_data[i]);
        for(int j = 0; j < des_shape[0]; j++)
        {
            des.at<float>(i, j) = des_data[j * kpts_shape[0] + i];
        }
    }
    int cnt = 0;
    for (int i = 0; i < image.rows(); i++)
    {
        for (int j = 0; j < image.cols(); j++)
        {
            ten.at<float>(i, j) = frametensor_data[i * image.cols() + j];
        }
    }
    descriptors = des;
    frame_tensor = ten;
}

void SuperGlue::match_keypoints(std::vector<cv::KeyPoint> &keypoints0, std::vector<float> &keypoint_scores0,
                                cv::InputArray descriptors0, cv::InputArray frame_tensor0,
                                std::vector<cv::KeyPoint> &keypoints1, std::vector<float> &keypoint_scores1,
                                cv::InputArray descriptors1, cv::InputArray frame_tensor1,
                                std::vector<cv::DMatch> &matches, std::vector<float> &confidence) {
    cv::Mat kpts0((int)(keypoints0.size()), 2, CV_32S);
    cv::Mat kpts1((int)(keypoints1.size()), 2, CV_32S);
    cv::Mat des0, des1;
    cv::transpose(descriptors0, des0);
    cv::transpose(descriptors1, des1);

    for (int i = 0; i < keypoints0.size(); i++)
    {
        kpts0.at<int>(i, 0) = (int)(keypoints0[i].pt.x);
        kpts0.at<int>(i, 1) = (int)(keypoints0[i].pt.y);
    }

    for (int i = 0; i < keypoints1.size(); i++)
    {
        kpts1.at<int>(i, 0) = (int)(keypoints1[i].pt.x);
        kpts1.at<int>(i, 1) = (int)(keypoints1[i].pt.y);
    }

    npy_intp dims_kpts0[2]{(int)(keypoints0.size()), 2};
    PyObject *array_kpts0 = PyArray_SimpleNewFromData(2, dims_kpts0, NPY_INT, kpts0.data);

    npy_intp dims_kpts1[2]{(int)(keypoints1.size()), 2};
    PyObject *array_kpts1 = PyArray_SimpleNewFromData(2, dims_kpts1, NPY_INT, kpts1.data);

    npy_intp dims_scores0[2]{1, (int)(keypoints0.size())};
    PyObject *array_scores0 = PyArray_SimpleNewFromData(2, dims_scores0, NPY_FLOAT, keypoint_scores0.data());

    npy_intp dims_scores1[2]{1, (int)(keypoints1.size())};
    PyObject *array_scores1 = PyArray_SimpleNewFromData(2, dims_scores1, NPY_FLOAT, keypoint_scores1.data());

    npy_intp dims_des0[2]{des0.rows, des0.cols};
    PyObject *array_des0 = PyArray_SimpleNewFromData(2, dims_des0, NPY_FLOAT, des0.data);

    npy_intp dims_des1[2]{des1.rows, des1.cols};
    PyObject *array_des1 = PyArray_SimpleNewFromData(2, dims_des1, NPY_FLOAT, des1.data);

    npy_intp dims_tensor0[4]{1, 1, frame_tensor0.rows(), frame_tensor0.cols()};
    PyObject *array_tensor0 = PyArray_SimpleNewFromData(4, dims_tensor0, NPY_FLOAT, frame_tensor0.getMat().data);

    npy_intp dims_tensor1[4]{1, 1, frame_tensor1.rows(), frame_tensor1.cols()};
    PyObject *array_tensor1 = PyArray_SimpleNewFromData(4, dims_tensor1, NPY_FLOAT, frame_tensor1.getMat().data);

    PyTuple_SetItem(py_arg_match, 0, Py_BuildValue("O", array_kpts0));
    PyTuple_SetItem(py_arg_match, 1, Py_BuildValue("O", array_scores0));
    PyTuple_SetItem(py_arg_match, 2, Py_BuildValue("O", array_des0));
    PyTuple_SetItem(py_arg_match, 3, Py_BuildValue("O", array_tensor0));
    PyTuple_SetItem(py_arg_match, 4, Py_BuildValue("O", array_kpts1));
    PyTuple_SetItem(py_arg_match, 5, Py_BuildValue("O", array_scores1));
    PyTuple_SetItem(py_arg_match, 6, Py_BuildValue("O", array_des1));
    PyTuple_SetItem(py_arg_match, 7, Py_BuildValue("O", array_tensor1));
    PyObject *ret = PyObject_CallObject(py_func_match, py_arg_match);

    std::vector<cv::KeyPoint> _0, _1;
    unpack_results(ret, _0, _1, matches, confidence);
}

SuperGlue::~SuperGlue()
{
    Py_Finalize();
}