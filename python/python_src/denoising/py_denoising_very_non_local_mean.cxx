#define PY_ARRAY_UNIQUE_SYMBOL skneuro_denoising_PyArray_API
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
#include <skneuro/denoising/very_non_local_mean.hxx>




namespace bp = boost::python;


vigra::NumpyAnyArray pyVeryNonLocalMean(
    vigra::NumpyArray<3,float> input,
    vigra::NumpyArray<3,float> out
){
    out.reshapeIfEmpty(input.taggedShape());
}




void export_very_non_local_mean(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    bp::numeric::array::set_module_and_type("numpy", "ndarray");
    bp::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above



    bp::def("veryNonLocalMean",vigra::registerConveters(pyVeryNonLocalMean),
        (
            bp::arg("image"),
            bp::arg("out")=bp::object()
        )
    );

    
}
