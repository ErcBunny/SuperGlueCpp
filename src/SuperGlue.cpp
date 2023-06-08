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
    py_func_get_keypoints = PyObject_GetAttrString(py_obj_SuperGlueWrapper, "get_keypoints");
    py_func_forward_full = PyObject_GetAttrString(py_obj_SuperGlueWrapper, "forward_full");
    py_func_forward_append = PyObject_GetAttrString(py_obj_SuperGlueWrapper, "forward_append");

    py_args_set_config = PyTuple_New(5);
    py_args_forward_full = PyTuple_New(2);
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

void SuperGlue::get_keypoints(cv::InputArray image)
{
    py_args_get_keypoints = Py_BuildValue("(O)", to_py_array(image));
    PyObject_CallObject(py_func_get_keypoints, py_args_get_keypoints);
}

void SuperGlue::forward_full(cv::InputArray image_0, cv::InputArray image_1, std::vector<cv::KeyPoint> &keypoints_0,
                             std::vector<cv::KeyPoint> &keypoints_1, std::vector<cv::DMatch> &matches,
                             std::vector<float> &confidence)
{
    PyTuple_SetItem(py_args_forward_full, 0, Py_BuildValue("O", to_py_array(image_0)));
    PyTuple_SetItem(py_args_forward_full, 1, Py_BuildValue("O", to_py_array(image_1)));
    PyObject *ret = PyObject_CallObject(py_func_forward_full, py_args_forward_full);

    unpack_results(ret, keypoints_0, keypoints_1, matches, confidence);
}

void SuperGlue::forward_append(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints_0,
                               std::vector<cv::KeyPoint> &keypoints_1, std::vector<cv::DMatch> &matches,
                               std::vector<float> &confidence)
{
    py_args_forward_append = Py_BuildValue("(O)", to_py_array(image));
    PyObject *ret = PyObject_CallObject(py_func_forward_append, py_args_forward_append);
    unpack_results(ret, keypoints_0, keypoints_1, matches, confidence);
}

PyObject *SuperGlue::to_py_array(cv::InputArray image)
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

SuperGlue::~SuperGlue()
{
    Py_Finalize();
}