#ifndef SKNEURO_LEARNING_GT_HELPER_HXX
#define SKNEURO_LEARNING_GT_HELPER_HXX


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


namespace skneuro{


    template<class L_IN, class L_OUT>
    void regionToEdgeGt(
        const vigra::MultiArrayView<3, L_IN> & regionGt,
        vigra::MultiArrayView<3, L_OUT> & pixelGt
    ){
        pixelGt = L_OUT(0);
        for(size_t z=0; z<regionGt.shape(2); ++z)
        for(size_t y=0; y<regionGt.shape(1); ++y)
        for(size_t x=0; x<regionGt.shape(0); ++x){

            if(x+1<regionGt.shape(0)){
                if(regionGt(x, y, z) != regionGt(x+1, y, z)){
                    pixelGt(x, y, z) = 2;
                    pixelGt(x+1, y, z) = 2;
                }
            }
            if(y+1<regionGt.shape(1)){
                if(regionGt(x, y, z) != regionGt(x, y+1, z)){
                    pixelGt(x, y, z) = 2;
                    pixelGt(x, y+1, z) = 2;
                }
            }
            if(z+1<regionGt.shape(2)){
                if(regionGt(x, y, z) != regionGt(x, y, z+1)){
                    pixelGt(x, y, z) = 2;
                    pixelGt(x, y, z+1) = 2;
                }
            }

            // 
            if(pixelGt(x, y, z)!=2 && regionGt(x, y, z)!=0){
                pixelGt(x, y, z)=1;
            }
        }
    }

}

#endif /*SKNEURO_LEARNING_GT_HELPER_HXX*/