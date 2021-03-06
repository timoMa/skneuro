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

// vigra 
#include <vigra/multi_blocking.hxx>
#include <vigra/box.hxx>

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


    template<class T_BLOCK, class T_TOTAL>
    static vigra::NumpyAnyArray extractBlock_3(
        const BlockWithBorderType & blockWithBorder,
        vigra::NumpyArray<3,T_TOTAL> totalData,
        vigra::NumpyArray<3,T_BLOCK> out 
    ){
        out.reshapeIfEmpty(blockWithBorder.border().size());
        {
            vigra::PyAllowThreads _pythread;
            skneuro::extractBlock(blockWithBorder,totalData,out);
        }
        return out;
    }

    template<class T_BLOCK, class T_TOTAL>
    static void writeFromBlock_3(
        const BlockWithBorderType & blockWithBorder,
        const vigra::NumpyArray<3,T_BLOCK> & blockData,
        vigra::NumpyArray<3,T_TOTAL> totalData
    ){
        {
            vigra::PyAllowThreads _pythread;
            skneuro::writeFromBlock(blockWithBorder, blockData, totalData);
        }
    }

};



template<class T_BLOCK, class T_TOTAL>
void export_block_data_t(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    boost::python::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above

        
    typedef vigra::MultiBlocking<3,int> BlockingType;

    typedef typename BlockingType::Block  BlockType;
    typedef typename BlockingType::BlockWithBorder  BlockWithBorderType;
    typedef BlockDataHelper<BlockWithBorderType> Helper;


    bp::def("extractBlock",
        vigra::registerConverters(&Helper::extractBlock_3<T_BLOCK, T_TOTAL>),
        (
            bp::arg("blockWithBorder"),
            bp::arg("totalData"),
            bp::arg("out") = bp::object()
        )
    );

    bp::def("writeFromBlock",
        vigra::registerConverters(&Helper::writeFromBlock_3<T_BLOCK, T_TOTAL>),
        (
            bp::arg("blockWithBorder"),
            bp::arg("blockData"),
            bp::arg("totalData")
        )
    );
}

template<class T_BLOCK, class T_TOTAL>
void export_block_data_tt(){
    export_block_data_t< T_BLOCK , T_TOTAL>();
    export_block_data_t< vigra::Singleband<T_BLOCK> ,vigra::Singleband<T_TOTAL> >();
    export_block_data_t< vigra::TinyVector<T_BLOCK, 3> , vigra::TinyVector<T_TOTAL, 3> >();


}


void export_block_data(){
    export_block_data_tt<float, float>();
    export_block_data_tt<float, vigra::UInt8>();
}
