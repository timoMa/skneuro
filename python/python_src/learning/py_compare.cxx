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

#include <skneuro/learning/partition-comparison.hxx>

namespace bp = boost::python;



template<class T>
double pyRandIndex(
    vigra::NumpyArray<1, UInt32> gt,
    vigra::NumpyArray<1, UInt32> seg,
    const bool ignoreDefaultLabel 
){
    return andres::randIndex(gt.begin(),gt.end(),seg.begin(),ignoreDefaultLabel);
}

template<class T>
double pyVariationOfInformation(
    vigra::NumpyArray<1, UInt32> gt,
    vigra::NumpyArray<1, UInt32> seg,
    const bool ignoreDefaultLabel 
){
    return andres::variationOfInformation(gt.begin(),gt.end(),seg.begin(),ignoreDefaultLabel);
}


template<class T>
void export_compare_t(){

    bp::def("randIndex",vigra::registerConverters(&pyRandIndex<T>),
        (
            bp::arg("a"),
            bp::arg("b"),
            bp::arg("ignoreDefaultLabel")
        )
    );

    bp::def("variationOfInformation",vigra::registerConverters(&pyVariationOfInformation<T>),
        (
            bp::arg("a"),
            bp::arg("b"),
            bp::arg("ignoreDefaultLabel")
        )
    );
}



void export_compare(){

  export_compare_t<vigra::UInt32>();
  export_compare_t<vigra::UInt16>();
}
