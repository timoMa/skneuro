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
#include <skneuro/denoising/iir_gaussian.hxx>



namespace bp = boost::python;




template<class PIXEL_TYPE>
vigra::NumpyAnyArray  pyGaussianIIR(
    vigra::NumpyArray<3,PIXEL_TYPE> image,
    const float sigma,
    const int numsteps
){
    //out.reshapeIfEmpty(image.shape());
    {
        vigra::PyAllowThreads _pythread;
        skneuro::gaussianIIR(image, sigma, numsteps);
    }
    return image;
}






void exportIIR(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    bp::numeric::array::set_module_and_type("numpy", "ndarray");
    bp::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above


    bp::def("gaussianIIR",vigra::registerConverters(&pyGaussianIIR<float>),
        (
            bp::arg("image"),
            bp::arg("sigma"),
            bp::arg("numsteps")
            //bp::arg("out")=bp::object()
        )
    );

}
