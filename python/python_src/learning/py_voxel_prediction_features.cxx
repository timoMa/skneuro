
#define PY_ARRAY_UNIQUE_SYMBOL skneuro_learning_PyArray_API
#define NO_IMPORT_ARRAY

// boost python related
#include <boost/python/detail/wrap_python.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/exception_translator.hpp>
#include <numpy/arrayobject.h>
#include <numpy/noprefix.h>

// vigra numpy array converters
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

// standart c++ headers (whatever you need (string and vector are just examples))
#include <string>
#include <vector>
#include <list>

// my headers  ( in my_module/include )
#include <skneuro/skneuro.hxx>
#include <skneuro/learning/feature_extraction.hxx>


#include <boost/python/object.hpp>
#include <boost/python/stl_iterator.hpp>

#include <skneuro/learning/voxel_prediction_feature_operator.hxx>

namespace bp = boost::python;






template<class T_IN, class T_OUT>
vigra::NumpyAnyArray
computeFeaturesTrain(
    const skneuro::IlastikFeatureOperator & op,
    const vigra::NumpyArray<3, T_IN> & data,
    const vigra::TinyVector<vigra::Int32,3> roiBegin,
    const vigra::TinyVector<vigra::Int32,3> roiEnd,
    const vigra::NumpyArray<1, vigra::TinyVector<vigra::UInt32,3 > > & whereGt,
    vigra::NumpyArray<2, T_OUT> features
){
    typedef typename vigra::NumpyArray<2, T_OUT>::difference_type Shape2;
    features.reshapeIfEmpty(Shape2(op.nFeatures(),whereGt.shape(0)));
    {
        vigra::TinyVector<vigra::UInt32,3> roiBegin_(roiBegin);
        vigra::TinyVector<vigra::UInt32,3> roiEnd_(roiEnd);
        vigra::PyAllowThreads _pythread;
        op.computeFeaturesTrain(data,roiBegin_,roiEnd_, whereGt, features);
    }

    return features;
}


template<class T_IN, class T_OUT>
vigra::NumpyAnyArray
computeFeaturesTest(
    const skneuro::IlastikFeatureOperator & op,
    const vigra::NumpyArray<3, T_IN> & data,
    const vigra::TinyVector<vigra::Int32,3> roiBegin,
    const vigra::TinyVector<vigra::Int32,3> roiEnd,
    vigra::NumpyArray<4, T_OUT> features
){
    typedef typename vigra::NumpyArray<4, T_OUT>::difference_type Shape4;
    Shape4 shape4;
    shape4[0] = op.nFeatures();
    for(size_t d=0; d<3;++d){
        shape4[d+1] = roiEnd[d] - roiBegin[d];
    }
    features.reshapeIfEmpty(shape4);
    {
        vigra::TinyVector<vigra::UInt32,3> roiBegin_(roiBegin);
        vigra::TinyVector<vigra::UInt32,3> roiEnd_(roiEnd);
        vigra::PyAllowThreads _pythread;
        op.computeFeaturesTest(data,roiBegin_,roiEnd_,features);
    }

    return features;
}



void export_voxel_prediction_features(){

    typedef skneuro::IlastikFeatureOperator FeatOp;

    bp::class_<FeatOp>("IlastikFeatureOperator",bp::init<>())
        .def("margin", &FeatOp::margin)
        .def("nFeatures", &FeatOp::nFeatures)
        ///
        .def("trainFeatures",vigra::registerConverters(&computeFeaturesTrain<float,float> ),
            (
                bp::arg("array"),
                bp::arg("roiBegin"),
                bp::arg("roiEnd"),
                bp::arg("whereGt"),
                bp::arg("out") = bp::object()
            )
        )
        .def("testFeatures",vigra::registerConverters(&computeFeaturesTest<float,float> ),
            (
                bp::arg("array"),
                bp::arg("roiBegin"),
                bp::arg("roiEnd"),
                bp::arg("out") = bp::object()
            )
        )
    ;

}
