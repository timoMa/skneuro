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

// my headers  ( in my_module/include )
#include <skneuro/skneuro.hxx>
#include <skneuro/learning/feature_extraction.hxx>


namespace bp = boost::python;

namespace skneuro{

    template<class PIXEL_TYPE>
    vigra::NumpyAnyArray 
    accumulateFeatures(
        const GridGraph3d & gridGraph,
        const Rag & rag,
        const GridGraph3dAffiliatedEdges & affiliatedEdges,
        const vigra::NumpyArray< 3 , PIXEL_TYPE>  & volume,
        const float histMin,
        const float histMax,
        const size_t nBins,
        const float histSigma,
        vigra::NumpyArray< 2 , float> features
    ){
        SKNEURO_CHECK_OP(rag.edgeNum(), >, 0, "no edges");
        SKNEURO_CHECK_OP(rag.edgeNum(),==,rag.maxEdgeId()+1, "malformed graph");
        typedef typename vigra::NumpyArray< 2 , float>::difference_type Shape2;
        const size_t numberOfFeatures = nBins+2;
        Shape2 featuresShape(rag.edgeNum(),numberOfFeatures);
        features.reshapeIfEmpty(featuresShape);
        accumulateFeatures<PIXEL_TYPE, float>(gridGraph, rag, affiliatedEdges,
                                              volume, histMin, histMax, nBins,
                                              histSigma, features);
        return features;
    }

} // end namespace skneuro



template<class PIXEL_TYPE>
void export_accumulate_features_T(){

    bp::def("_accumulateFeatures", vigra::registerConverters(&skneuro::accumulateFeatures<PIXEL_TYPE>),
        (
            bp::arg("gridGraph"),
            bp::arg("rag"),
            bp::arg("affiliatedEdges"),
            bp::arg("volume"),
            bp::arg("histMin"),
            bp::arg("histMax"),
            bp::arg("nBins"),
            bp::arg("histSigma"),
            bp::arg("out") = bp::object()
        )
    );
}

void export_accumulate_features(){

    export_accumulate_features_T<float>();
    export_accumulate_features_T<vigra::UInt8>();
}



void export_feature_extraction(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    bp::numeric::array::set_module_and_type("numpy", "ndarray");
    bp::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above

    export_accumulate_features();

}
