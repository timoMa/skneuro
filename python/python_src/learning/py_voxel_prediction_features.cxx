
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
#include <boost/python/stl_iterator.hpp> 

namespace bp = boost::python;






template<class OP,class T_IN, class T_OUT>
vigra::NumpyAnyArray
computeFeaturesTrain(
    const OP & op,
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


template<class OP, class T_IN, class T_OUT>
vigra::NumpyAnyArray
computeFeaturesTest(
    const OP & op,
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
        vigra::PyAllowThreads _pythread;
        vigra::TinyVector<vigra::UInt32,3> roiBegin_(roiBegin);
        vigra::TinyVector<vigra::UInt32,3> roiEnd_(roiEnd);
        op.computeFeaturesTest(data,roiBegin_,roiEnd_,features);
    }

    return features;
}




skneuro::IlastikFeatureOperator * ilastikFeatOpConstructor(
    bp::object sigmasBp,
    vigra::NumpyArray<2, bool> featureSelection 
){
    typedef skneuro::IlastikFeatureOperator FeatOp;
    typedef typename FeatOp::UseSigma UseSigma;

    std::vector<float> sigma;
    {
        bp::stl_input_iterator<float> begin(sigmasBp), end;
        sigma.assign(begin,end); 
    }
    std::vector<UseSigma> useSigmaVec(FeatOp::NFeatFunc);

    FeatOp * op;
    {
        vigra::PyAllowThreads _pythread;
        op = new FeatOp(sigma,featureSelection);
    }
    return op;
}

skneuro::SlicFeatureOp * slicFeatOpConstructor(
    const vigra::NumpyArray<1, vigra::UInt32> & seedDistances,
    const vigra::NumpyArray<1, float> & intensityScalings 
){
    typedef skneuro::SlicFeatureOp FeatOp;


    std::vector<unsigned int> seedDistances_(seedDistances.begin(),seedDistances.end());  
    std::vector<double> intensityScalings_(intensityScalings.begin(),intensityScalings.end());

    FeatOp * op;
    {
        vigra::PyAllowThreads _pythread;
        op = new FeatOp(seedDistances_,intensityScalings_);
    }
    return op;
}



void export_voxel_prediction_features(){


    {
        typedef skneuro::IlastikFeatureOperator FeatOp;
        bp::class_<FeatOp>("RawIlastikFeatureOperator",bp::no_init)
            .def("margin", &FeatOp::margin)
            .def("nFeatures", &FeatOp::nFeatures)
            .def("__init__", 
                bp::make_constructor(vigra::registerConverters(&ilastikFeatOpConstructor),bp::default_call_policies(),
                    (
                        bp::arg("sigmas"),
                        bp::arg("featureSelection")
                    )
                )
            )
            .def("trainFeatures",vigra::registerConverters(&computeFeaturesTrain<FeatOp,float,float> ),
                (
                    bp::arg("array"),
                    bp::arg("roiBegin"),
                    bp::arg("roiEnd"),
                    bp::arg("whereGt"),
                    bp::arg("out") = bp::object()
                )
            )
            .def("testFeatures",vigra::registerConverters(&computeFeaturesTest<FeatOp,float,float> ),
                (
                    bp::arg("array"),
                    bp::arg("roiBegin"),
                    bp::arg("roiEnd"),
                    bp::arg("out") = bp::object()
                )
            )
        ;
    }
    {
        typedef skneuro::SlicFeatureOp FeatOp;
        bp::class_<FeatOp>("RawSlicFeatureOp",bp::no_init)
            .def("margin", &FeatOp::margin)
            .def("nFeatures", &FeatOp::nFeatures)
            .def("__init__", 
                bp::make_constructor(vigra::registerConverters(&slicFeatOpConstructor),bp::default_call_policies(),
                    (
                        bp::arg("seedDistances"),
                        bp::arg("intensityScalings")
                    )
                )
            )
            .def("trainFeatures",vigra::registerConverters(&computeFeaturesTrain<FeatOp,float,float> ),
                (
                    bp::arg("array"),
                    bp::arg("roiBegin"),
                    bp::arg("roiEnd"),
                    bp::arg("whereGt"),
                    bp::arg("out") = bp::object()
                )
            )
            .def("testFeatures",vigra::registerConverters(&computeFeaturesTest<FeatOp,float,float> ),
                (
                    bp::arg("array"),
                    bp::arg("roiBegin"),
                    bp::arg("roiEnd"),
                    bp::arg("out") = bp::object()
                )
            )
        ;
    }

}
