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


#include <boost/python/object.hpp>
#include <boost/python/stl_iterator.hpp>

#include <skneuro/learning/patch_rf.hxx>

namespace bp = boost::python;

template<class T, class L>
void py_learn(
    skneuro::PatchRf<T, L> & rf,
    vigra::NumpyArray<4, T> features,
    vigra::NumpyArray<3, L> labels
){
    rf.train(features, labels);
}





void export_patch_rf(){
    typedef skneuro::PatchRf<float, vigra::UInt32> PyPatchRf;
    typedef typename  PyPatchRf::Param PyParam;

    bp::class_<PyParam>("PatchRfParam", bp::init<>())
    ;

    bp::class_<PyPatchRf>("PatchRf",bp::init<PyParam>( (bp::arg("param") = PyParam() ) ))
    .def("train",vigra::registerConverters(&py_learn<float, vigra::UInt32> ))
    ;
}
