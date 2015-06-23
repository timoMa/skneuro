
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

#include <skneuro/learning/voxel_prediction_tools.hxx>

namespace bp = boost::python;



//template<class T>
//double pyRandIndex(
//    vigra::NumpyArray<1, UInt32> gt,
//    vigra::NumpyArray<1, UInt32> seg,
//    const bool ignoreDefaultLabel 
//){
//    return andres::randIndex(gt.begin(),gt.end(),seg.begin(),ignoreDefaultLabel);
//}
//


template<class T>
bool hasLabels(
    const vigra::NumpyArray<3, T> & labelVolume,
    const vigra::NumpyArray<1, vigra::UInt32> & labels
){
    std::set<vigra::UInt32> lset(labels.begin(), labels.end());

    bool hasLabels = false;
    {
        vigra::PyAllowThreads _pythread;
        hasLabels = skneuro::hasLabels(labelVolume,lset);
    }
    return hasLabels;
}

template<class T>
vigra::UInt32 countLabels(
    const vigra::NumpyArray<3, T> & labelVolume,
    const vigra::NumpyArray<1, vigra::UInt32> & labels
){
    std::set<vigra::UInt32> lset(labels.begin(), labels.end());

    vigra::UInt32 c = 0;
    {
        vigra::PyAllowThreads _pythread;
        c = skneuro::countLabels(labelVolume,lset);
    }
    return c;
}


template<class T>
bp::tuple countLabelsAndFindRoi(
    const vigra::NumpyArray<3, T> & labelVolume,
    const vigra::NumpyArray<1, vigra::UInt32> & labels
){
    std::set<vigra::UInt32> lset(labels.begin(), labels.end());

    vigra::TinyVector<Int32, 3 >  roiBegin;
    vigra::TinyVector<Int32, 3 >  roiEnd;

    vigra::UInt32 c = 0;
    {
        vigra::PyAllowThreads _pythread;
        c = skneuro::countLabelsAndFindRoi(labelVolume,lset,roiBegin,roiEnd);
    }
    return bp::make_tuple(c,roiBegin,roiEnd);
}



template<class T>
void remapLabels(
    vigra::NumpyArray<3, T> labelVolume,
    const vigra::NumpyArray<1, vigra::UInt32> & rLabels
){
    vigra::PyAllowThreads _pythread;
    skneuro::remapLabels(labelVolume,rLabels);
}



template<class T>
bp::tuple getLabelsAndLocation(
    const vigra::NumpyArray<3, T> labelVolume,
    const vigra::NumpyArray<1, vigra::UInt32> & rLabels,
    const UInt32 labelsCount,
    vigra::NumpyArray<1, vigra::UInt32>  actualLabels,
    vigra::NumpyArray<1, vigra::TinyVector<UInt32,3> > whereLabels
){
    typedef typename vigra::NumpyArray<1, vigra::UInt32>::difference_type Shape;
    actualLabels.reshapeIfEmpty(Shape(labelsCount));
    whereLabels.reshapeIfEmpty(Shape(labelsCount));
    {
        vigra::PyAllowThreads _pythread;
        skneuro::getLabelsAndLocation(labelVolume,rLabels,actualLabels,whereLabels);
    }
    return bp::make_tuple(actualLabels, whereLabels);
}


template<class T>
void export_voxel_prediction_tools_t(){

    bp::def("hasLabels",vigra::registerConverters(&hasLabels<T>),
        (
            bp::arg("array"),
            bp::arg("labels")
        )
    );
    bp::def("countLabels",vigra::registerConverters(&countLabels<T>),
        (
            bp::arg("array"),
            bp::arg("labels")
        )
    );
    bp::def("countLabelsAndFindRoi",vigra::registerConverters(&countLabelsAndFindRoi<T>),
        (
            bp::arg("array"),
            bp::arg("labels")
        )
    );
    bp::def("remapLabels",vigra::registerConverters(&remapLabels<T>),
        (
            bp::arg("array"),
            bp::arg("rLabels")
        )
    );
    bp::def("getLabelsAndLocation",vigra::registerConverters(&getLabelsAndLocation<T>),
        (
            bp::arg("array"),
            bp::arg("rLabels"),
            bp::arg("labelsCount"),
            bp::arg("actualLabels") = bp::object(),
            bp::arg("whereLabels") = bp::object()
        )
    );
}

void export_voxel_prediction_tools(){
    export_voxel_prediction_tools_t<vigra::UInt32>();
    export_voxel_prediction_tools_t<vigra::UInt8>();
}
