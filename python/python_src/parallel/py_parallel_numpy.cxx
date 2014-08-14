#define PY_ARRAY_UNIQUE_SYMBOL skneuro_parallel_PyArray_API
#define NO_IMPORT_ARRAY

// boost python related
#include <boost/python/detail/wrap_python.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/object.hpp>
#include <boost/python/stl_iterator.hpp>
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
#include <skneuro/parallel/parallel_multi_array.hxx>

namespace bp = boost::python;

namespace skneuro{

    template<unsigned int DIM, class T>
    bp::tuple pyArrayMinMax(const vigra::NumpyArray<DIM, T> & array){
        T minVal, maxVal;
        parallel::arrayMinMax(array, minVal, maxVal);
        return bp::make_tuple(minVal, maxVal);
    }


} // end namespace skneuro

template<unsigned int DIM, class T>
void export_array_min_max_d_t(){
    bp::def("arrayMinMax", vigra::registerConverters(&skneuro::pyArrayMinMax<DIM, T> ),
        (
            bp::arg("array")
        )
    );
}

template<class T>
void export_array_min_max_t(){
    export_array_min_max_d_t<1, T>();
    export_array_min_max_d_t<2, T>();
    export_array_min_max_d_t<3, T>();
    export_array_min_max_d_t<4, T>();
}


void export_parallel_numpy(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    bp::numeric::array::set_module_and_type("numpy", "ndarray");
    bp::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above

    export_array_min_max_t<float>();
    export_array_min_max_t<double>();
    export_array_min_max_t<vigra::UInt8>();
    export_array_min_max_t<vigra::UInt32>();

}
