#define PY_ARRAY_UNIQUE_SYMBOL skneuro_learning_PyArray_API
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
#include <list>

// my headers  ( in my_module/include )
#include <skneuro/skneuro.hxx>
#include <skneuro/learning/feature_extraction.hxx>


#include <boost/python/object.hpp>
#include <boost/python/stl_iterator.hpp>

namespace bp = boost::python;

namespace skneuro{

    template<class PIXEL_TYPE>
    bp::tuple
    accumulateFeatures(
        const GridGraph3d & gridGraph,
        const Rag & rag,
        const vigra::NumpyArray<3, vigra::UInt32> & labels,
        const GridGraph3dAffiliatedEdges & affiliatedEdges,
        const vigra::NumpyArray< 3 , PIXEL_TYPE>  & volume,
        const AccumulatorOptions & options,
        vigra::NumpyArray< 2 , float> edgeFeatures,
        vigra::NumpyArray< 2 , float> nodeFeatures
    ){
        SKNEURO_CHECK_OP(rag.edgeNum(), >, 0, "no edges");
        SKNEURO_CHECK_OP(rag.edgeNum(),==,rag.maxEdgeId()+1, "malformed graph");
        typedef typename vigra::NumpyArray< 2 , float>::difference_type Shape2;
        const size_t numberOfFeatures = options.featuresPerChannel();
        std::cout<<"numberOfFeatures per channel "<<numberOfFeatures<<"\n";
        if(options.edgeFeatures){
            Shape2 featuresShape(rag.edgeNum(),numberOfFeatures);
            edgeFeatures.reshapeIfEmpty(featuresShape);
        }
        if(options.nodeFeatures){
            Shape2 featuresShape(rag.maxNodeId()+1,numberOfFeatures);
            nodeFeatures.reshapeIfEmpty(featuresShape);
        }
        accumulateFeatures<PIXEL_TYPE, float>(gridGraph, rag, labels, 
                                              affiliatedEdges, volume, 
                                              options, edgeFeatures,
                                              nodeFeatures);

        return bp::make_tuple(edgeFeatures, nodeFeatures);
    }

} // end namespace skneuro



template<class PIXEL_TYPE>
void export_accumulate_features_T(){

    bp::def("_accumulateFeatures", vigra::registerConverters(&skneuro::accumulateFeatures<PIXEL_TYPE>),
        (
            bp::arg("gridGraph"),
            bp::arg("rag"),
            bp::arg("labels"),
            bp::arg("affiliatedEdges"),
            bp::arg("volume"),
            bp::arg("options"),
            bp::arg("edgeFeaturs") = bp::object(),
            bp::arg("nodeFeaturs") = bp::object()
        )
    );
}

vigra::NumpyAnyArray getHistMin(const skneuro::AccumulatorOptions & options){
    vigra::NumpyArray<1, double > histMin(options.histMin);
    return histMin;
}

void setHistMin(skneuro::AccumulatorOptions & options,
                const vigra::NumpyArray<1, double> histMin
){
    options.histMin = histMin;
}

vigra::NumpyAnyArray getHistMax(const skneuro::AccumulatorOptions & options){
    vigra::NumpyArray<1, double > histMax(options.histMax);
    return histMax;
}

void setHistMax(skneuro::AccumulatorOptions & options,
                const vigra::NumpyArray<1, double> histMax
){
    options.histMax = histMax;
}

void setSelect(skneuro::AccumulatorOptions & options,
               const bp::object & obj){

    bp::stl_input_iterator<std::string> begin(obj), end;
    options.select.assign(begin, end);
}

bp::list getSelect(const skneuro::AccumulatorOptions & options){
    bp::list ret;
    for(size_t i=0; i<options.select.size(); ++i)
        ret.append(options.select[i]);
    return ret;
}



void export_accumulate_features(){  

    typedef skneuro::AccumulatorOptions AccOpts;

    bp::class_<AccOpts>("AccumulatorOptions",bp::init<>())
    .add_property("select", &getSelect, &setSelect)
    .def_readwrite("edgeFeaturs", &AccOpts::edgeFeatures)
    .def_readwrite("nodeFeaturs", &AccOpts::nodeFeatures)
    .def_readwrite("nBins", &AccOpts::nBins)
    .def_readwrite("sigmaHist", &AccOpts::sigmaHist)
    .add_property("histMin", 
                  vigra::registerConverters(&getHistMin), 
                  vigra::registerConverters(&setHistMin))
    .add_property("histMax", 
                  vigra::registerConverters(&getHistMax), 
                  vigra::registerConverters(&setHistMax))
    ;

    export_accumulate_features_T<float>();
    export_accumulate_features_T<vigra::UInt8>();
}



void export_feature_extraction(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    bp::numeric::array::set_module_and_type("numpy", "ndarray");
    bp::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above

    export_accumulate_features();

}
