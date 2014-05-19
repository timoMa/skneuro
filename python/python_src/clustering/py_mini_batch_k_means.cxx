#define PY_ARRAY_UNIQUE_SYMBOL skneuro_clustering_PyArray_API
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
#include <skneuro/clustering/mini_batch_k_means.hxx>


namespace skneuro{
namespace clustering{

namespace mini_batch_k_means{

template<class CLS>
void run(
    CLS & self,
    vigra::NumpyArray<2,float> & features
){
    self.run(features);
}


template<class CLS>
vigra::NumpyAnyArray clusterCenters(
    CLS & self,
    vigra::NumpyArray<2,float> & centers
){
    centers.reshapeIfEmpty(self.clusterCenters().shape());
    centers = self.clusterCenters();
    return centers;
}

} // end namespace mini_batch_k_means 


void export_mini_batch_k_means(){

    namespace python = boost::python ;

    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    python::numeric::array::set_module_and_type("numpy", "ndarray");
    python::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above

    typedef vigra::metrics::Metric<float> PyMetric;
    typedef MiniBatchKMeans<float, PyMetric > PyMiniBatchKMeans; 
    python::class_< PyMiniBatchKMeans > (
        "MiniBatchKMeans",
        python::init<const size_t ,const size_t , const size_t ,const size_t, const  PyMetric & >(
            (
                python::arg("nFeatures"),
                python::arg("nClusters"),
                python::arg("miniBatchSize"),
                python::arg("nIter"),
                python::arg("metric")
            )
        ) 
    )
    .def("run", vigra::registerConverters(&mini_batch_k_means::run<PyMiniBatchKMeans>),
        (
            python::arg("features")
        )
    )
    .def("clusterCenters", vigra::registerConverters(&mini_batch_k_means::clusterCenters<PyMiniBatchKMeans>),
        (
            python::arg("centers")=python::object()
        )
    )
    ;


}

} // end namespace clustering
} // end namespace skneuro
