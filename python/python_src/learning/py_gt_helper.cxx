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
#include <skneuro/learning/gt_helper.hxx>

namespace bp = boost::python;




vigra::NumpyAnyArray pyRegionToEdgeGt(
    vigra::NumpyArray<3, UInt32> rgt,
    vigra::NumpyArray<3, UInt32> pgt
){
    pgt.reshapeIfEmpty(rgt.shape());
    skneuro::regionToEdgeGt(rgt,pgt);
    return pgt;
}





void export_gt_helper(){

    bp::def("regionToEdgeGt",vigra::registerConverters(&pyRegionToEdgeGt),
        (
            bp::arg("rgt"),
            bp::arg("out")=bp::object()
        )
    );

}


