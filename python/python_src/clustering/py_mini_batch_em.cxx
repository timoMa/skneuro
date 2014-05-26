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
#include <skneuro/clustering/mini_batch_em.hxx>


namespace skneuro{
namespace clustering{


template<class CLS>
struct PyMiniBatchEmHelper{


    static void run(
        CLS & self,
        vigra::NumpyArray<2,double> features
    ){
        self.run(features);
    }

    static vigra::NumpyAnyArray clusterCenters(
        CLS & self,
        vigra::NumpyArray<2,double> & centers
    ){
        centers.reshapeIfEmpty(self.clusterCenters().shape());
        centers = self.clusterCenters();
        return centers;
    }

    static vigra::NumpyAnyArray predict(
        const CLS & self,
        const vigra::NumpyArray<2,double> & features,
        vigra::NumpyArray<2,double>  probabilities
    ){
        typename  vigra::NumpyArray<2,double>::difference_type resShape(self.nClusters(),features.shape(1));
        probabilities.reshapeIfEmpty(resShape);

        self.predict(features,probabilities);
        return probabilities;
    }


    static void initalizeCenters(
        CLS & self,
        const vigra::NumpyArray<2,double> & centers
    ){
        self.initalizeCenters(centers);
    }

    static void initalizeCenters2(
        CLS & self,
        const vigra::NumpyArray<2,double> & features,
        const vigra::NumpyArray<1,vigra::UInt32> & labels
    ){
        self.initalizeCenters2(features,labels);
    }

};


void export_mini_batch_em(){

    namespace python = boost::python ;

    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    python::numeric::array::set_module_and_type("numpy", "ndarray");
    python::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above

    typedef MiniBatchEm<double > PyMiniBatchEm; 
    python::class_< PyMiniBatchEm > (
        "MiniBatchEm",
        python::init<const size_t ,const size_t , const size_t ,const size_t,const double >(
            (
                python::arg("nFeatures"),
                python::arg("nClusters"),
                python::arg("miniBatchSize"),
                python::arg("nIter"),
                python::arg("varScale")
            )
        ) 
    )
    .def("run", vigra::registerConverters(&PyMiniBatchEmHelper<PyMiniBatchEm>::run),
        (
            python::arg("features")
        )
    )
   //.def("clusterCenters", vigra::registerConverters(&PyMiniBatchEmHelper<PyMiniBatchEm>::clusterCenters),
   //    (
   //        python::arg("centers")=python::object()
   //    )
   //)
    .def("predict", vigra::registerConverters(&PyMiniBatchEmHelper<PyMiniBatchEm>::predict),
        (
            python::arg("features"),
            python::arg("out")=python::object()
        )
    )
    .def("initalizeCenters", vigra::registerConverters(&PyMiniBatchEmHelper<PyMiniBatchEm>::initalizeCenters),
        (
            python::arg("centers")
        )
    )
    .def("initalizeCenters", vigra::registerConverters(&PyMiniBatchEmHelper<PyMiniBatchEm>::initalizeCenters2),
        (
            python::arg("features"),
            python::arg("labels")
        )
    )
    ;


}

} // end namespace clustering
} // end namespace skneuro
