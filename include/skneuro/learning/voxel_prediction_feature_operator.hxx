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

    
    
    class IlastikFeatureOperator{

    public:
        typedef vigra::TinyVector<vigra::UInt32, 3> Coord;

        IlastikFeatureOperator(

        ){

        }


        Coord margin()const{
            return Coord(20);
        }

        size_t nFeatures()const{
            return 1;
        }

        template<class T_IN, class T_OUT>
        void computeFeaturesTrain(
            const vigra::MultiArrayView<3, T_IN> & data,
            const Coord & roiBegin,
            const Coord & roiEnd,
            const vigra::MultiArrayView<1, vigra::TinyVector<vigra::UInt32, 3> > whereGt,
            vigra::MultiArrayView<2, T_OUT> & features
        )const{
            
            this->computeFeaturesTest(data, roiBegin, roiEnd)
        }

        template<class T_IN, class T_OUT>
        void computeFeaturesTest (
            const vigra::MultiArrayView<3, T_IN> & data,
            const Coord & roiBegin,
            const Coord & roiEnd,
            vigra::MultiArrayView<4, T_OUT> & features
        )const{

        }


    private:


    };

}

#endif //SKNEURO_VOXEL_PREDICTION_TOOLS_HXX
