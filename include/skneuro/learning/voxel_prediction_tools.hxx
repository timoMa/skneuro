#ifndef SKNEURO_VOXEL_PREDICTION_TOOLS_HXX
#define SKNEURO_VOXEL_PREDICTION_TOOLS_HXX

#include <iostream>

#include <vigra/multi_array.hxx>
#include <vigra/multi_gridgraph.hxx>
#include <vigra/graphs.hxx>
#include <vigra/graph_generalization.hxx>
#include <vigra/multi_gridgraph.hxx>
#include <vigra/adjacency_list_graph.hxx>
#include <vigra/graph_maps.hxx>
#include <vigra/timing.hxx>
#include <vigra/accumulator.hxx>
#include <vigra/multi_convolution.hxx>

#include <omp.h>
namespace skneuro{

    template<class T, class SET_TYPE>
    bool hasLabels(
        const vigra::MultiArrayView<3, T> & labelVolume,
        const SET_TYPE & labels
    ){
        typedef typename vigra::MultiArrayView<3, T>::const_iterator Iter;
        for(Iter iter = labelVolume.begin();iter!=labelVolume.end(); ++iter){
            const T l(*iter);
            if(l!=0){
                if(labels.find(l)!=labels.end()){
                    return true;
                }
            }
        }
        return false;
    }


    template<class T, class SET_TYPE>
    vigra::UInt32 countLabels(
        const vigra::MultiArrayView<3, T> & labelVolume,
        const SET_TYPE & labels
    ){
        vigra::UInt32 c = 0;
        typedef typename vigra::MultiArrayView<3, T>::const_iterator Iter;
        for(Iter iter = labelVolume.begin();iter!=labelVolume.end(); ++iter){
            const T l(*iter);
            if(l!=0){
                if(labels.find(l)!=labels.end()){
                    ++c;
                }
            }
        }
        return c;
    }


    template<class T,class C, class SET_TYPE>
    vigra::UInt32 countLabelsAndFindRoi(
        const vigra::MultiArrayView<3, T> & labelVolume,
        const SET_TYPE & labels,
        vigra::TinyVector<C, 3 > & roiBegin,
        vigra::TinyVector<C, 3 > & roiEnd
    ){
        typedef vigra::TinyVector<C, 3 > Coord;

        roiEnd = Coord(0);
        roiBegin = labelVolume.shape();    

        vigra::UInt32 lCount = 0;
        Coord c;
        for(c[0]=0; c[0]<labelVolume.shape(0); ++c[0])
        for(c[1]=0; c[1]<labelVolume.shape(1); ++c[1])
        for(c[2]=0; c[2]<labelVolume.shape(2); ++c[2]){
            const T l(labelVolume[c]);
            if(l!=0){
                if(labels.find(l)!=labels.end()){
                    for(int d=0; d<3; ++d){
                        roiBegin[d] = std::min(roiBegin[d],c[d]);
                        roiEnd[d] = std::max(roiEnd[d],c[d]);
                    }
                    ++lCount;
                }
            }
        }
        roiEnd+=Coord(1);
        return lCount;
    }




    template<class T, class L>
    void remapLabels(
        vigra::MultiArrayView<3, T> & labelVolume,
        const vigra::MultiArrayView<1, L> & remapping
    ){
        vigra::UInt32 c = 0;
        typedef typename vigra::MultiArrayView<3, T>::iterator Iter;
        for(Iter iter = labelVolume.begin();iter!=labelVolume.end(); ++iter){
            const T l(*iter);
            if(l!=0){
                *iter = remapping[l];
            }
        }
    }



    template<class T, class L>
    void getLabelsAndLocation(
        const vigra::MultiArrayView<3, T> & labelVolume,
        const vigra::MultiArrayView<1, L> & remapping,
        // output
        vigra::MultiArrayView<1, vigra::UInt32> & actualLabels,
        vigra::MultiArrayView<1, vigra::TinyVector<vigra::UInt32,3> > & whereLabels
    ){
        typedef vigra::TinyVector<UInt32,3> Coord;
        vigra::UInt32 counter = 0;
        for(vigra::UInt32 x=0; x<labelVolume.shape(0); ++x)
        for(vigra::UInt32 y=0; y<labelVolume.shape(1); ++y)
        for(vigra::UInt32 z=0; z<labelVolume.shape(2); ++z){
            const T l(labelVolume(x,y,z));
            if(l!=0){
                const vigra::UInt32 rl = remapping[l];
                //std::cout<<"RL "<<rl<<"\n";
                if(rl!=0){
                    const vigra::UInt32 al = rl -1;
                    //std::cout<<"l "<<l<<"rl "<<rl<<"al "<<al<<"\n";
                    actualLabels[counter] = al;
                    whereLabels(counter) = Coord(x,y,z);
                    ++counter;
                    //std::cout<<"counter "<<counter<<"\n";
                }
            }
        }
    }



}

#endif //SKNEURO_VOXEL_PREDICTION_TOOLS_HXX
