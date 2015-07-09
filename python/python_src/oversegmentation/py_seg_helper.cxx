#define PY_ARRAY_UNIQUE_SYMBOL skneuro_oversegmentation_PyArray_API
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

namespace bp = boost::python;





template<class T>
void sizeFilterSegInplace(vigra::NumpyArray<3, T>  seg, const vigra::UInt32 maxVal, const vigra::UInt32 sizeLimit){
    

    std::vector<bool > atBorder(maxVal+1, false);

    for(size_t z=0;z<seg.shape(2); ++z)
    for(size_t y=0;y<seg.shape(1); ++y){
        atBorder[seg(0,y,z)] = true;
        atBorder[seg(seg.shape(0)-1,y,z)] = true;
    }

    for(size_t z=0;z<seg.shape(2); ++z)
    for(size_t x=0;x<seg.shape(0); ++x){
        atBorder[seg(x,0,z)] = true;
        atBorder[seg(x,seg.shape(1)-1,z)] = true;
    }

    for(size_t y=0;y<seg.shape(1); ++y)
    for(size_t x=0;x<seg.shape(0); ++x){
        atBorder[seg(x,y,0)] = true;
        atBorder[seg(x,y,seg.shape(2)-1)] = true;
    }



    std::vector<size_t > counts(maxVal+1,0);

    for(auto iter = seg.begin(); iter!=seg.end(); ++iter){
        counts[*iter] += 1;
    }



    for(auto iter = seg.begin(); iter!=seg.end(); ++iter){
        const auto l = *iter;
        const auto c = counts[l];
        if(c<sizeLimit && atBorder[l] == false){
            *iter = 0;
        }
    }
}



void export_seg_helper(){

    bp::def("sizeFilterSegInplace",vigra::registerConverters(&sizeFilterSegInplace<vigra::UInt32>),
        (
            bp::arg("seg"),
            bp::arg("maxVal"),
            bp::arg("sizeLimit")
        )
    );

}


