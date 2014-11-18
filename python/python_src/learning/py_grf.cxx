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

#include <skneuro/learning/grf/rf_topology.hxx>

namespace bp = boost::python;




//template<class GRF>
//void export_grf_t(const std::string & clsName){
//
//    bp::class_<GRF>(clsName.c_str(), bp::init<>())
//    ;
//  
//}



void export_rf_topology(){
    typedef skneuro::RfTopology PyRfTopology;
    bp::class_<PyRfTopology>("RfTopology",bp::init<>())
    ;
}



void export_grf(){


    //typedef skneuro::GeneralizedRandomForest Grf;

    //
    //export_grf_t<Grf>("Patch2dRf");
    export_rf_topology();
}
