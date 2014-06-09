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


template<class BLOCK_WITH_BORDER>
struct BlockDataHelper{
    typedef BLOCK_WITH_BORDER BlockWithBorderType;


    template<class T>
    static vigra::NumpyAnyArray extractBlock_3(
        const BlockWithBorderType & blockWithBorder,
        vigra::NumpyArray<3,T> totalData,
        vigra::NumpyArray<3,T> out 
    ){
        out.reshapeIfEmpty(blockWithBorder.blockWithBorderShape());
        {
            vigra::PyAllowThreads _pythread;
            skneuro::extractBlock(blockWithBorder,totalData,out);
        }
        return out;
    }

    template<class T>
    static void writeFromBlock_3(
        const BlockWithBorderType & blockWithBorder,
        const vigra::NumpyArray<3,T> & blockData,
        vigra::NumpyArray<3,T> totalData
    ){
        {
            vigra::PyAllowThreads _pythread;
            skneuro::writeFromBlock(blockWithBorder, blockData, totalData);
        }
    }

};



template<class T>
void export_block_data_t(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    boost::python::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above

    


    typedef skneuro::Block<int, 3>  BlockType;
    typedef skneuro::BlockWithBorder<int, 3> BlockWithBorderType;
    typedef BlockDataHelper<BlockWithBorderType> Helper;


    bp::def("extractBlock",
        vigra::registerConverters(&Helper::extractBlock_3<T>),
        (
            bp::arg("blockWithBorder"),
            bp::arg("totalData"),
            bp::arg("out") = bp::object()
        )
    );

    bp::def("writeFromBlock",
        vigra::registerConverters(&Helper::writeFromBlock_3<T>),
        (
            bp::arg("blockWithBorder"),
            bp::arg("blockData"),
            bp::arg("totalData")
        )
    );
}


void export_block_data(){
    export_block_data_t<float>();
    export_block_data_t< vigra::Singleband<float> >();
    export_block_data_t< vigra::TinyVector<float, 3> >();
}