#define PY_ARRAY_UNIQUE_SYMBOL skneuro_utilities_PyArray_API
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

// vigra 
#include <vigra/multi_blocking.hxx>
#include <vigra/box.hxx>

// vigra numpy array converters
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

// standart c++ headers (whatever you need (string and vector are just examples))
#include <string>
#include <vector>

// my headers  ( in my_module/include )
#include <skneuro/skneuro.hxx>
#include <skneuro/utilities/blocking.hxx>



namespace bp = boost::python;





class BwbIterHolder{
    typedef vigra::MultiBlocking<3, int> MultiBlocking;
    typedef MultiBlocking::BlockWithBorderIter BlockWithBorderIter;

public:
    typedef BlockWithBorderIter const_iterator;
    BwbIterHolder(
        const vigra::MultiBlocking<3,int> & mb, 
        const vigra::TinyVector<int, 3> & w
    )
    : begin_(mb.blockWithBorderBegin(w)),
      end_(mb.blockWithBorderEnd(w))
    {
    }
    //BlockWithBorderIter begin()const{
    //    std::cout<<"get block with b begin\n";
    //    return begin_;
    //}
    //BlockWithBorderIter end()const{
    //    std::cout<<"get block with b begin\n";
    //    return end_;
    //}
    MultiBlocking::BlockWithBorder getBlock(const size_t i)const{
        return begin_[i];
    }
private:
    BlockWithBorderIter begin_;
    BlockWithBorderIter end_;
};



BwbIterHolder getIter(const vigra::MultiBlocking<3, int>  & mb, 
                      const vigra::TinyVector<int, 3> & w){
    return BwbIterHolder(mb, w);
}

size_t nBlock(const vigra::MultiBlocking<3, int>  & mb){
    return std::distance(mb.blockBegin(), mb.blockEnd());
}


vigra::TinyVector<int,3> getShape(const vigra::MultiBlocking<3, int>  & mb){
    return mb.shape();
}

vigra::TinyVector<int,3> getBSlockShape(const vigra::MultiBlocking<3, int>  & mb){
    return mb.blockShape();
}

vigra::TinyVector<int,3> getBlocksPerAxis(const vigra::MultiBlocking<3, int>  & mb){
    return mb.blocksPerAxis();
}

void export_blocking(){
    // Do not change next 4 lines
    //import_array(); 
    //vigra::import_vigranumpy();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    boost::python::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above

    typedef vigra::MultiBlocking<3, int> MultiBlocking;
    typedef MultiBlocking::BlockWithBorderIter BlockWithBorderIter;





    typedef vigra::Box<int, 3>  BlockType;
    typedef vigra::MultiBlocking<3,int>  BlockingType;
    typedef BlockType::Vector CoordType;




    bp::class_<BwbIterHolder>("BwbIterHolder",bp::no_init)
    //.def("__iter__",bp::range<bp::return_value_policy<bp::return_by_value> >(&BwbIterHolder::begin,&BwbIterHolder::end))
    //.def("__iter__",bp::iterator<const BwbIterHolder >())
    .def("getBlock",&BwbIterHolder::getBlock)
    ;



    bp::class_<BlockingType>("Blocking3d",bp::init<const CoordType , const CoordType  >())
        .def("__len__", &nBlock)
        .def("blockWithBorderIter",&getIter)
        .def("shape",&getShape)
        .def("blockShape",&getBSlockShape)
        .def("blocksPerAxis",&getBlocksPerAxis)
    ;
}