#define PY_ARRAY_UNIQUE_SYMBOL skneuro_clustering_PyArray_API
//#define NO_IMPORT_ARRAY


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
#include <boost/python/exception_translator.hpp>

// standart c++ headers (whatever you need (string and vector are just examples))
#include <string>
#include <vector>

namespace skneuro{
namespace clustering{
	
    void export_mini_batch_em();
}
}

void translateStdRuntimeError(std::runtime_error const& e){
PyErr_SetString(PyExc_RuntimeError, e.what());
}



// export my module
BOOST_PYTHON_MODULE_INIT(_clustering) {

    // Do not change next 4 lines
    import_array(); 
    vigra::import_vigranumpy();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    boost::python::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above



	boost::python::register_exception_translator<std::runtime_error>(&translateStdRuntimeError);
    skneuro::clustering::export_mini_batch_em();

}