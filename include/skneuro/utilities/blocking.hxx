#ifndef SKNEURO_UTILITIES_BLOCKING 
#define SKNEURO_UTILITIES_BLOCKING 

#include <vigra/multi_blocking.hxx>
#include <vigra/box.hxx>

#include <iostream>


namespace skneuro{






template<class COORDINATE,class T_BLOCK, class T_TOTAL>
void extractBlock(
    const vigra::detail_multi_blocking::BlockWithBorder<3, COORDINATE> & blockWithBorder,
    const vigra::MultiArrayView<3,T_TOTAL> & totalData,
    vigra::MultiArrayView<3,T_BLOCK> & blockData
){
    typedef typename vigra::detail_multi_blocking::BlockWithBorder<3, COORDINATE>::Point Point;
    Point begin = blockWithBorder.border().begin();
    Point end   = blockWithBorder.border().end();


    Point totalCoord;
    Point blockCoord;
    //for(totalCoord[0]=begin[0],blockCoord[0]=0; totalCoord[0]<end[0]; ++totalCoord[0],++blockCoord[0])
    //for(totalCoord[1]=begin[1],blockCoord[1]=0; totalCoord[1]<end[1]; ++totalCoord[1],++blockCoord[1])
    //for(totalCoord[2]=begin[2],blockCoord[2]=0; totalCoord[2]<end[2]; ++totalCoord[2],++blockCoord[2]){

    for(totalCoord[2]=begin[2],blockCoord[2]=0; totalCoord[2]<end[2]; ++totalCoord[2],++blockCoord[2])
    for(totalCoord[1]=begin[1],blockCoord[1]=0; totalCoord[1]<end[1]; ++totalCoord[1],++blockCoord[1])
    for(totalCoord[0]=begin[0],blockCoord[0]=0; totalCoord[0]<end[0]; ++totalCoord[0],++blockCoord[0]){
        blockData[blockCoord]=totalData[totalCoord];
    }
}


template<class COORDINATE,class T_BLOCK, class T_TOTAL>
void writeFromBlock(
    const vigra::detail_multi_blocking::BlockWithBorder<3, COORDINATE> & blockWithBorder,
    const vigra::MultiArrayView<3,T_BLOCK> & blockData,
    vigra::MultiArrayView<3,T_TOTAL> & totalData
){
    typedef typename vigra::detail_multi_blocking::BlockWithBorder<3, COORDINATE>::Point Point;
    const Point tBegin  = blockWithBorder.core().begin();
    const Point tEnd = blockWithBorder.core().end();

    const Point bBegin = blockWithBorder.localCore().begin();
    //const Point bEnd = blockWithBorder.blockLocalCoordinates().end();


    Point tCoord;
    Point bCoord;
    for(tCoord[2]=tBegin[2],bCoord[2]=bBegin[2]; tCoord[2]<tEnd[2]; ++tCoord[2],++bCoord[2])
    for(tCoord[1]=tBegin[1],bCoord[1]=bBegin[1]; tCoord[1]<tEnd[1]; ++tCoord[1],++bCoord[1])
    for(tCoord[0]=tBegin[0],bCoord[0]=bBegin[0]; tCoord[0]<tEnd[0]; ++tCoord[0],++bCoord[0]){
        totalData[tCoord]=blockData[bCoord];
    }
}


} // end namespace skneuro

#endif /*SKNEURO_UTILITIES_BLOCKING */
