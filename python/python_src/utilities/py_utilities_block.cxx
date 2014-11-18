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


template<class BLOCK>
struct BlockHelper{
    typedef BLOCK BlockType;
    typedef typename BlockType::CoordType CoordType;

    static CoordType begin(const BlockType & block){
        return block.begin();
    }
    static CoordType end(const BlockType & block){
        return block.end();
    }

};




void export_block(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    boost::python::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above


    typedef skneuro::Block<int, 3>  BlockType;
    typedef skneuro::BlockWithBorder<int, 3> BlockWithBorderType;
    typedef BlockType::Vector CoordType;
    typedef BlockHelper<BlockType> BlockHelperType;



    bp::class_<BlockType>("Block3d",bp::init<>())
        .def(bp::init<const CoordType &, const CoordType & >())
        .def("begin",&BlockHelperType::begin)
        .def("end",&BlockHelperType::end)
        .def("addBorder",&BlockType::addBorder)
        .def("__str__",&BlockType::str)
    ;

    bp::class_<BlockWithBorderType>("BlockWithBorder",bp::init<const BlockType &, const BlockType & >())
        .def(bp::init<>())
        .def("__str__",&BlockWithBorderType::str)
        .def("blockShape",&BlockWithBorderType::blockShape )
        .def("blockWithBorderShape",&BlockWithBorderType::blockWithBorderShape )
    ;



}