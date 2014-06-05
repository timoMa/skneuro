#define PY_ARRAY_UNIQUE_SYMBOL skneuro_utilities_PyArray_API
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
#include <skneuro/utilities/blocking.hxx>




namespace bp = boost::python;

template<class BLOCKING>
struct BlockingHelper{
    typedef BLOCKING BlockingType;
    typedef typename BlockingType::CoordType CoordType;


};

void export_blocking(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    boost::python::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above





    typedef Block<int, 3>  BlockType;
    typedef Blocking<int, 3>  BlockingType;

    typedef BlockType::Vector CoordType;
    typedef BlockingHelper<BlockingType> BlockingHelperType;


    bp::class_<BlockingType>("Blocking3d",bp::init<>())
        .def(bp::init<const CoordType &, const CoordType & >())
        .def("__len__", &BlockingType::size)
        .def("blockWithBorder", &BlockingType::blockWithBorder,
            (
                bp::arg("index"),
                bp::arg("width")
            )
        )
    ;
}