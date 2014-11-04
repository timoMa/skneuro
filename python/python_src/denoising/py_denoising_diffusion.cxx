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
#include <skneuro/denoising/diffusion.hxx>




namespace bp = boost::python;





template<unsigned int DIM, class PIXEL_TYPE>
vigra::NumpyAnyArray pyDiffusion(
    vigra::NumpyArray<DIM, PIXEL_TYPE> input,
    const skneuro::DiffusionParam & param
){
    skneuro::blockwiseDiffusion(input, param);
    return input;
}




void exportDiffusion(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    bp::numeric::array::set_module_and_type("numpy", "ndarray");
    bp::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above


    bp::class_<skneuro::DiffusionParam>("DiffusionParam",bp::init<>())
    
        .def_readwrite("strength", &skneuro::DiffusionParam::strength_)
        .def_readwrite("dt", &skneuro::DiffusionParam::dt_)
        .def_readwrite("maxT", &skneuro::DiffusionParam::maxT_)
        .def_readwrite("sigmaStep", &skneuro::DiffusionParam::sigmaStep_)
        .def_readwrite("C", &skneuro::DiffusionParam::C_)
        .def_readwrite("m", &skneuro::DiffusionParam::m_)
        .def_readwrite("eps", &skneuro::DiffusionParam::eps_)
        .def_readwrite("alpha", &skneuro::DiffusionParam::alpha_)
        .def_readwrite("useSt", &skneuro::DiffusionParam::useSt_)
        .def_readwrite("sigmaTensor1", &skneuro::DiffusionParam::sigmaTensor1_)
        .def_readwrite("sigmaTensor2", &skneuro::DiffusionParam::sigmaTensor2_)
    ;

    
    bp::def("diffusion3d",vigra::registerConverters(&pyDiffusion<3, float>),
        (
            bp::arg("image"),
            bp::arg("param")
        )
    );

    bp::def("diffusion2d",vigra::registerConverters(&pyDiffusion<2,float>),
        (
            bp::arg("image"),
            bp::arg("param")
        )
    );

    bp::def("diffusion2dc",vigra::registerConverters(&pyDiffusion<2,vigra::TinyVector<float, 3> >),
        (
            bp::arg("image"),
            bp::arg("param")
        )
    );

    
}
