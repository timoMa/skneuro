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
#include <skneuro/denoising/ball_rank_order.hxx>




namespace bp = boost::python;




template<class PIXEL_TYPE>
vigra::NumpyAnyArray  pyBallRankOrder(
    vigra::NumpyArray<3,PIXEL_TYPE> image,
    const int radius,
    const float rank,
    vigra::NumpyArray<3,PIXEL_TYPE> out = vigra::NumpyArray<3,PIXEL_TYPE>()
){
    out.reshapeIfEmpty(image.shape());
    {
        vigra::PyAllowThreads _pythread;
        skneuro::ballRankOrderFilter(image, radius, rank, out);
    }
    return out;
}


template<class PIXEL_TYPE>
vigra::NumpyAnyArray  pyBallRankOrderNew(
    vigra::NumpyArray<3,PIXEL_TYPE> image,
    const int radius,
    const float rank,
    const float minVal,
    const float maxVal,
    const float nBins,
    vigra::NumpyArray<3,PIXEL_TYPE> out = vigra::NumpyArray<3,PIXEL_TYPE>()
){
    out.reshapeIfEmpty(image.shape());
    {
        vigra::PyAllowThreads _pythread;
        skneuro::ballRankOrderFilterNew(image, radius, rank,
                                        minVal, maxVal, nBins,
                                        out);
    }
    return out;
}







void exportBallRankOrder(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    bp::numeric::array::set_module_and_type("numpy", "ndarray");
    bp::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above


    bp::def("ballRankOrderFilter",vigra::registerConverters(&pyBallRankOrder<float>),
        (
            bp::arg("image"),
            bp::arg("radius"),
            bp::arg("rank"),
            bp::arg("out")=bp::object()
        )
    );

    bp::def("ballRankOrderFilterNew",vigra::registerConverters(&pyBallRankOrderNew<float>),
        (
            bp::arg("image"),
            bp::arg("radius"),
            bp::arg("rank"),
            bp::arg("minVal"),
            bp::arg("maxVal"),
            bp::arg("out")=bp::object()
        )
    );

}
