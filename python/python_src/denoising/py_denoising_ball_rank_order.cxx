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




template<class T_IN, unsigned int N_RANKS, class T_OUT>
vigra::NumpyAnyArray  pyBallRankOrderNew(
    vigra::NumpyArray<3,T_IN> image,
    const int radius,
    const int takeNth,
    const vigra::TinyVector<float,N_RANKS> & ranks,
    const bool useHistogram,
    const float minVal,
    const float maxVal,
    const float nBins,
    vigra::NumpyArray<3, vigra::TinyVector<T_OUT, N_RANKS> > out
){
    out.reshapeIfEmpty(image.shape());
    {
        vigra::PyAllowThreads _pythread;
        skneuro::ballRankOrderFilterNew<T_IN,N_RANKS,T_OUT>
        (image, radius, takeNth, ranks,useHistogram,minVal, 
        maxVal, nBins, out);
    }
    return out;
}


template<class T_IN, unsigned int N_RANKS, class T_OUT>
void exportRankOrderT(){

    bp::def("ballRankOrder",vigra::registerConverters
        (
            &pyBallRankOrderNew<T_IN, N_RANKS, T_OUT>
        ),
        (
            bp::arg("image"),
            bp::arg("radius"),
            bp::arg("takeNth"),
            bp::arg("ranks"),
            bp::arg("useHistogram"),
            bp::arg("minVal"),
            bp::arg("maxVal"),
            bp::arg("nBins"),
            bp::arg("out")=bp::object()
        )
    );
}






void exportBallRankOrder(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    bp::numeric::array::set_module_and_type("numpy", "ndarray");
    bp::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above


    exportRankOrderT<float,1,float>();
    exportRankOrderT<float,2,float>();
    exportRankOrderT<float,3,float>();
    exportRankOrderT<float,4,float>();
    exportRankOrderT<float,5,float>();
    exportRankOrderT<float,5,float>();
    exportRankOrderT<float,6,float>();
    exportRankOrderT<float,7,float>();
    exportRankOrderT<float,8,float>();
    exportRankOrderT<float,9,float>();


    exportRankOrderT<vigra::UInt8,1,float>();
    exportRankOrderT<vigra::UInt8,2,float>();
    exportRankOrderT<vigra::UInt8,3,float>();
    exportRankOrderT<vigra::UInt8,4,float>();
    exportRankOrderT<vigra::UInt8,5,float>();
    exportRankOrderT<vigra::UInt8,6,float>();
    exportRankOrderT<vigra::UInt8,7,float>();
    exportRankOrderT<vigra::UInt8,8,float>();
    exportRankOrderT<vigra::UInt8,9,float>();


}


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






void exportBallRankOrderOld(){
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

}
