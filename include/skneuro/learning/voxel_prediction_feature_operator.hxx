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
#include <vigra/multi_tensorutilities.hxx>


#include <omp.h>
namespace skneuro{

    template<class T_OUT>
    struct ExtractTrain{
        typedef T_OUT value_type;

        ExtractTrain(
            const vigra::MultiArrayView<1, vigra::TinyVector<vigra::UInt32, 3> > & whereGt,
            vigra::MultiArrayView<2, T_OUT> & features
        )
        :   whereGt_(whereGt),
            features_(features),
            fIndex_(0){

        }


        void store(const vigra::MultiArrayView<3, T_OUT> & featureImg){
            for(size_t inst=0; inst<whereGt_.size(); ++inst){
                features_(fIndex_, inst) = featureImg[whereGt_[inst]];
            }
            ++fIndex_;
        }

        template<int NC>
        void store(const vigra::MultiArrayView<3, vigra::TinyVector<T_OUT, NC> > & featureImg){
            for(size_t inst=0; inst<whereGt_.size(); ++inst){
                const auto & tv = featureImg[whereGt_[inst]];
                for(size_t f=0; f<NC; ++f)
                    features_(fIndex_+f, inst) = tv[f];
            }
            ++fIndex_;
        }

        const vigra::MultiArrayView<1, vigra::TinyVector<vigra::UInt32, 3> > & whereGt_;
        vigra::MultiArrayView<2, T_OUT > & features_;
        size_t fIndex_;
    };

    template<class T_OUT>
    struct ExtractTest{
        typedef T_OUT value_type;

        ExtractTest(
            vigra::MultiArrayView<4, T_OUT> & features
        )
        :   features_(features),
            fIndex_(0){

        }


        void store(const vigra::MultiArrayView<3, T_OUT> & featureImg){
            features_.bindInner(fIndex_) = featureImg;
            ++fIndex_;
        }

        template<int NC>
        void store(const vigra::MultiArrayView<3, vigra::TinyVector<T_OUT, NC> > & featureImg){
            for(size_t f=0; f<NC; ++f)
                features_.bindInner(fIndex_) = featureImg.bindElementChannel(f);
            ++fIndex_;
        }


        vigra::MultiArrayView<4, T_OUT > & features_;
        size_t fIndex_;
    };
    

    
    class IlastikFeatureOperator{

    public:
        typedef  vigra::ConvolutionOptions<3> ConvOpts;

        static const size_t NFeatFunc = 6;
        enum class UseFeatFunc: size_t{
            GaussianSmoothing             = 0, 
            LaplacianOfGaussian           = 1, 
            GaussianGradientMagnitude     = 2, 
            DifferenceOfGaussians         = 3, 
            StructureTensorEigenvalues    = 4, 
            HessianOfGaussianEigenvalues  = 5
        };


        typedef vigra::TinyVector<vigra::UInt32, 3> Coord;
        typedef vigra::MultiArrayView<4, float> Shape4;
        IlastikFeatureOperator(

        )
        : sigmas_({0.7, 1, 1.6, 3.5, 5.0, 10.0}){

        }


        vigra::TinyVector<int,3> margin()const{
            const double s = sigmas_.back();
            int w = static_cast<int>(3.0*s + 1.5 + 0.5);
            return vigra::TinyVector<int,3>(w);

        }

        size_t nFeatures()const{
            return 6*sigmas_.size();
        }

        template<class T_IN, class T_OUT>
        void computeFeaturesTrain(
            const vigra::MultiArrayView<3, T_IN> & data,
            const Coord & roiBegin,
            const Coord & roiEnd,
            const vigra::MultiArrayView<1, vigra::TinyVector<vigra::UInt32, 3> > & whereGt,
            vigra::MultiArrayView<2, T_OUT> & features
        )const{
                
            // allocate buffer
            
            ExtractTrain<T_OUT> extractor(whereGt, features);
            this->computeFeatures(data, roiBegin, roiEnd, extractor);
        }

        template<class T_IN, class T_OUT>
        void computeFeaturesTest (
            const vigra::MultiArrayView<3, T_IN> & data,
            const Coord & roiBegin,
            const Coord & roiEnd,
            vigra::MultiArrayView<4, T_OUT> & features
        )const{
            ExtractTest<T_OUT> extractor(features);
            this->computeFeatures(data, roiBegin, roiEnd, extractor);
        }


        template<class T_IN, class EXTRACTOR>
        void computeFeatures(
            const vigra::MultiArrayView<3, T_IN> & data,
            const Coord & roiBegin,
            const Coord & roiEnd,
            EXTRACTOR & extractor
        )const{
            //feature out type
            typedef typename EXTRACTOR::value_type FType;


            // single buffer for gaussian
            float currentSigma = 0;
            vigra::MultiArray<3, FType>  gaussSmoothed(data);


            vigra::MultiArray<3, FType>  scalarCoreBuffer(roiEnd-roiBegin);
            typedef vigra::TinyVector<FType, 6> TV6;
            vigra::MultiArray<3, TV6>  core6Buffer(roiEnd-roiBegin);

            typedef vigra::TinyVector<FType, 3> TV3;
            vigra::MultiArray<3, TV3>  core3Buffer(roiEnd-roiBegin);
            for(size_t si=0; si<sigmas_.size(); ++si){

                // sigma'ing
                const float desiredSigma = sigmas_[si];
                const float sigmaPresmooth = desiredSigma - 0.25;


                //std::cout<<"\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
                //std::cout<<"si "<<si<<" "<<desiredSigma<<" dataShape"<<data.shape();
                //std::cout<<" margin"<<this->margin()<<"\n";

                //////////////////////////////
                // gaussian pre smoothing (inplace)
                ///////////////////////////////////
                //std::cout<<"presmooth\n";
                vigra::gaussianSmoothMultiArray(gaussSmoothed,gaussSmoothed,
                    ConvOpts().stdDev(sigmaPresmooth).resolutionStdDev(currentSigma));


                /////////////////////////////////
                // filters on pre-smoothed
                /////////////////////////////////
                ConvOpts  opts = ConvOpts().stdDev(desiredSigma).resolutionStdDev(sigmaPresmooth).subarray(roiBegin,roiEnd);

                // LaplacianOfGaussian
                //std::cout<<"laplacianOfGaussian\n";
                vigra::laplacianOfGaussianMultiArray(gaussSmoothed, scalarCoreBuffer, opts);
                // ==> store feature
                extractor.store(scalarCoreBuffer);

                // GaussianGradientMagnitude
                //std::cout<<"gaussianGradientMagnitude\n";
                vigra::gaussianGradientMagnitude(gaussSmoothed, scalarCoreBuffer, opts);
                // ==> store feature
                extractor.store(scalarCoreBuffer);


                

                // others
                // HessianOfGaussian
                vigra::hessianOfGaussianMultiArray(gaussSmoothed, core6Buffer, opts); 
                vigra::tensorEigenvaluesMultiArray(core6Buffer, core3Buffer);
                extractor.store(core3Buffer);


                // GaussianSmoothing
                //std::cout<<"gaussianSmoothMultiArray\n";
                vigra::gaussianSmoothMultiArray(gaussSmoothed, gaussSmoothed,
                    ConvOpts().stdDev(desiredSigma).resolutionStdDev(sigmaPresmooth)
                );
                // ==> store feature
                extractor.store(gaussSmoothed.subarray(roiBegin,roiEnd));

                // remember current sigma
                currentSigma = desiredSigma;

            }
        }



    private:
        std::vector<float> sigmas_;
        vigra::MultiArray<1, vigra::TinyVector<bool,NFeatFunc> > useFeature_;
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
