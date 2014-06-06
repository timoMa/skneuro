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
#include <skneuro/denoising/non_local_mean.hxx>




namespace bp = boost::python;




template<int DIM,class PIXEL_TYPE,class SMOOTH_POLICY>
vigra::NumpyAnyArray  pyNonLocalMean(
    vigra::NumpyArray<DIM,PIXEL_TYPE> image,
    const typename SMOOTH_POLICY::ParameterType & policyParam,
    const double sigmaSpatial,
    const int searchRadius,
    const int patchRadius,
    const double sigmaMean,
    const int stepSize,
    const int iterations,
    const int nThreads,
    const bool verbose,
    vigra::NumpyArray<DIM,PIXEL_TYPE> out = vigra::NumpyArray<DIM,PIXEL_TYPE>()
){

    SMOOTH_POLICY smoothPolicy(policyParam);
    skneuro::NonLocalMeanParameter param;
    param.sigmaSpatial_=sigmaSpatial;
    param.searchRadius_=searchRadius;
    param.patchRadius_=patchRadius;
    param.sigmaMean_=sigmaMean;
    param.stepSize_=stepSize;
    param.iterations_=iterations;
    param.nThreads_ = nThreads;
    param.verbose_=verbose;
    out.reshapeIfEmpty(image.shape());
    skneuro::nonLocalMean<DIM,PIXEL_TYPE>(image,smoothPolicy,param,out);
    return out;
}

void exportNonLocalMeanPolicyParameterObjects(){

    {
        typedef skneuro::RatioPolicyParameter ParamType;

        bp::class_<ParamType>(
            "RatioPolicy",
            bp::init<const double,const double,const double,const double>(
                (
                    bp::arg("sigma"),
                    bp::arg("meanRatio")=0.95,
                    bp::arg("varRatio")=0.5,
                    bp::arg("epsilon")=0.00001
                )
            )
        )
        .def_readwrite("sigma", &ParamType::sigma_)
        .def_readwrite("meanRatio", &ParamType::meanRatio_)
        .def_readwrite("varRatio", &ParamType::varRatio_)
        .def_readwrite("epsilon", &ParamType::epsilon_)
        ;
            
    }

    {
        typedef skneuro::NormPolicyParameter ParamType;

        bp::class_<ParamType>(
            "NormPolicy",
            bp::init<const double,const double,const double>(
                (
                    bp::arg("sigma"),
                    bp::arg("meanDist"),
                    bp::arg("varRatio")
                )
            )
        )
        .def_readwrite("sigma", &ParamType::sigma_)
        .def_readwrite("meanDist", &ParamType::meanDist_)
        .def_readwrite("varRatio", &ParamType::varRatio_)
        ;
            
    }
}


template<int DIM,class PIXEL_TYPE, class POLICY>
void exportNonLocalMean_template(const std::string name){

    typedef POLICY SmoothPolicyType;
    typedef typename SmoothPolicyType::ParameterType SmoothPolicyParameterType;
    // export the function to python
    bp::def(name.c_str(), registerConverters(&pyNonLocalMean<DIM,PIXEL_TYPE,SmoothPolicyType>) ,
        (
            bp::arg("image"),
            bp::arg("policy"),
            bp::arg("sigmaSpatial")=2.0,
            bp::arg("searchRadius")=3,
            bp::arg("patchRadius")=1,
            bp::arg("sigmaMean")=1.0,
            bp::arg("stepSize")=2,
            bp::arg("iterations")=1,
            bp::arg("nThreads")=8,
            bp::arg("verbose")=true,
            bp::arg("out") = bp::object()
        ),
        "loop over an image and do something with each pixels\n\n"
        "Args:\n\n"
        "   image : input image\n\n"
        "returns an an image with the same shape as the input image"
    );
}




void exportNonLocalMean(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    bp::numeric::array::set_module_and_type("numpy", "ndarray");
    bp::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above


    // export different parameter objects
    exportNonLocalMeanPolicyParameterObjects();

    exportNonLocalMean_template<3,float, skneuro::RatioPolicy<float> >("_nonLocalMean3d");
    exportNonLocalMean_template<3,float, skneuro::NormPolicy<float> >("_nonLocalMean3d");

}
