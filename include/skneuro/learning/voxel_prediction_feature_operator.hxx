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

    
    class GaussianSmoothBank(){
        GaussianSmoothBank(
            float sigmaMin,
            float sigmaMax,
            int   totalMargin
        )
    }


    
    class IlastikFeatureOperator{

    public:
        // ScalesList = [0.3, 0.7, 1, 1.6, 3.5, 5.0, 10.0]
        typedef vigra::TinyVector<vigra::UInt32, 3> Coord;
        typedef vigra::MultiArrayView<4, float> Shape4;
        IlastikFeatureOperator(

        )
        : sigmas_(){

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
                
            // allocate buffer
            Shape4 shape4;
            shape4[0] = nFeatures();
            for(size_t d=0; d<3; ++d){
                shape4[d+1] = roiEnd[d] - roiBegin[d];
            }
            
            // outAsVolume is only the very CORE
            vigra::MultiArray<4, T_OUT> outAsVolume(shape4); 
            this->computeFeaturesTest(data, roiBegin, roiEnd,outAsVolume);

            for(size_t i=0; i<whereGt.size(); ++i){
                const Coord c = whereGt(i);
                for(size_t f=0; f<nFeatures(); ++f){
                    features(f, i) = outAsVolume(f,c[0],c[1],c[2]);
                }
            }
        }

        template<class T_IN, class T_OUT>
        void computeFeaturesTest (
            const vigra::MultiArrayView<3, T_IN> & data,
            const Coord & roiBegin,
            const Coord & roiEnd,
            vigra::MultiArrayView<4, T_OUT> & features
        )const{


            // allocate buffer
            Shape4 dataShape4;
            dataShape4[0] = nFeatures();
            for(size_t d=0; d<3; ++d){
                dataShape4[d+1] = data.shape(d);
            }
            

            ////////////////////////////
            ///  Gaussian Smoothing  ///
            ////////////////////////////
            dataShape4[0] = 
            //std::vector<>
        }


    private:
        std::vector<float> sigmas_;

    };


    
    class GaussianFeatures{

    private:
        std::vector<float> sigmas_;
        MultiArray<1, TinyVector<bool, > > featureSelection_;
    };

    /*

        0 'GaussianSmoothing' 
        1 'LaplacianOfGaussian'
        2 'GaussianGradientMagnitude'
        3 'DifferenceOfGaussians'
        4 'StructureTensorEigenvalues'
        5 'HessianOfGaussianEigenvalues'



        


        gauss,
        *gradient(gaussian), 
        -gradientMagnitude(gradient)
        -laplacianOfGaussian(gaussian),
        *hessianOfGaussian(???)
        -hogEV
        -hogDET
        -hogTRACE
        *StructureTensor
        *stEVEC
        -stEV
    */

}

#endif //SKNEURO_VOXEL_PREDICTION_TOOLS_HXX
