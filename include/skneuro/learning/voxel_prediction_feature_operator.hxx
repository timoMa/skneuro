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
#include <vigra/slic.hxx>
#include <vigra/accumulator.hxx>
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
            fIndex_+=NC;
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
                features_.bindInner(fIndex_+f) = featureImg.bindElementChannel(f);
            fIndex_+=NC;
        }


        vigra::MultiArrayView<4, T_OUT > & features_;
        size_t fIndex_;
    };
    

    template<class T>
    void structureTensorAndGradientMagnitude(
        vigra::MultiArrayView<3, T>                          data,
        vigra::MultiArrayView<3, T >                       & gradientMagnitude,
        vigra::MultiArrayView<3, vigra::TinyVector<T, 6> > & structureTensor,
        const vigra::ConvolutionOptions<3> & convOpts
    ){
        typedef vigra::ConvolutionOptions<3> Opts;
        typedef typename  Opts::Shape Shape;

        const auto roi = convOpts.getSubarray();
        const auto & roiBegin = roi.first;
        const auto & roiEnd = roi.second;


        // gradient 
        const auto & outerScale =  convOpts.getOuterScale();
        Shape convMargin;
        Shape gradRoiBegin = roiBegin;
        Shape gradRoiEnd = roiEnd;
        for(size_t d=0; d<3; ++d){
            convMargin = int(outerScale[d]*3.0+0.5);
        }  

        gradRoiBegin-=convMargin;
        gradRoiEnd  +=convMargin;
        for(size_t d=0; d<3; ++d){
            gradRoiBegin[d] = std::max(vigra::Int64(0),vigra::Int64(gradRoiBegin[d]));
            gradRoiEnd[d] = std::min(data.shape(d),gradRoiEnd[d]);
        } 

        Opts convOptsGrad = convOpts;
        convOptsGrad.subarray(gradRoiBegin, gradRoiEnd);
        vigra::MultiArray<3,  vigra::TinyVector<T, 3> > gradWithMargin(gradRoiEnd-gradRoiBegin);

        // call actual gradient computing function
        vigra::gaussianGradientMultiArray(data, gradWithMargin, convOptsGrad);

        // make tensor from image
        vigra::MultiArray<3,  vigra::TinyVector<T, 6> > gradientTensor(gradRoiEnd-gradRoiBegin);
        vigra::transformMultiArray(gradWithMargin,gradientTensor, 
                                  vigra::detail::StructurTensorFunctor<3, vigra::TinyVector<T, 6> >());



        const Shape newRoiBegin = roiBegin - gradRoiBegin;
        const Shape newRoiEnd  = roiEnd - gradRoiBegin;
        Opts convOptSmooth = convOpts;
        convOptSmooth.subarray(newRoiBegin, newRoiEnd);
        convOptSmooth.stdDev(convOpts.getOuterScale());
        vigra::gaussianSmoothMultiArray(gradientTensor, structureTensor, 
                                        convOptSmooth);


        // fetch the grad subarray
        auto subGrad = gradWithMargin.subarray(newRoiBegin,newRoiEnd);

        {
            using namespace vigra::multi_math;
            const auto gx = subGrad.bindElementChannel(0);
            const auto gy = subGrad.bindElementChannel(1);
            const auto gz = subGrad.bindElementChannel(2);
            gradientMagnitude = gx*gx + gy*gy + gz*gz;
        }


        //vigra::gaussianGradientMultiArray(data, gradient,  convOpts);
        //vigra::ConvolutionOptions<3> &
        //vigra::transformMultiArray(gradient,vigra::detail::StructurTensorFunctor<3, T>());
    }


    
    class IlastikFeatureOperator{

    public:
        typedef  vigra::ConvolutionOptions<3> ConvOpts;

        static const size_t NFeatFunc = 6;
        typedef vigra::TinyVector<bool,NFeatFunc> UseSigma;
        enum class FuncIndex: size_t{
            GaussianSmoothing             = 0, 
            LaplacianOfGaussian           = 1, 
            GaussianGradientMagnitude     = 2, 
            StructureTensorEigenvaluesS2  = 3, 
            StructureTensorEigenvaluesS4  = 4, 
            HessianOfGaussianEigenvalues  = 5
        };


        typedef vigra::TinyVector<vigra::UInt32, 3> Coord;
        typedef vigra::MultiArrayView<4, float> Shape4;
        IlastikFeatureOperator(
            const std::vector<float> & sigmas,
            const vigra::MultiArrayView<2, bool> & featureSelection
        )
        :   sigmas_(sigmas),
            featureSelection_(featureSelection),
            nFeatures_(0){

            nFeatures_ += featureSelection_.bindOuter(size_t(FuncIndex::GaussianSmoothing)).sum<size_t>();
            nFeatures_ += featureSelection_.bindOuter(size_t(FuncIndex::LaplacianOfGaussian)).sum<size_t>();
            nFeatures_ += featureSelection_.bindOuter(size_t(FuncIndex::GaussianGradientMagnitude)).sum<size_t>();
            nFeatures_ += 3*featureSelection_.bindOuter(size_t(FuncIndex::StructureTensorEigenvaluesS2)).sum<size_t>();
            nFeatures_ += 3*featureSelection_.bindOuter(size_t(FuncIndex::StructureTensorEigenvaluesS4)).sum<size_t>();
            nFeatures_ += 3*featureSelection_.bindOuter(size_t(FuncIndex::HessianOfGaussianEigenvalues)).sum<size_t>();

            maxSigma_ = 0.0;
            for(size_t si=0; si<sigmas.size(); ++si){
                if(featureSelection_(si,size_t(FuncIndex::GaussianSmoothing))){
                    maxSigma_ = std::max(sigmas_[si],maxSigma_);
                }
                if(featureSelection_(si,size_t(FuncIndex::LaplacianOfGaussian))){
                    maxSigma_ = std::max(sigmas_[si],maxSigma_);
                }
                if(featureSelection_(si,size_t(FuncIndex::GaussianGradientMagnitude))){
                    maxSigma_ = std::max(sigmas_[si],maxSigma_);
                }
                if(featureSelection_(si,size_t(FuncIndex::StructureTensorEigenvaluesS2))){
                    maxSigma_ = std::max(2.0f*sigmas_[si],maxSigma_);
                }
                if(featureSelection_(si,size_t(FuncIndex::StructureTensorEigenvaluesS4))){
                    maxSigma_ = std::max(4.0f*sigmas_[si],maxSigma_);
                }
                if(featureSelection_(si,size_t(FuncIndex::HessianOfGaussianEigenvalues))){
                    maxSigma_ = std::max(sigmas_[si],maxSigma_);
                }
            }

        }


        vigra::TinyVector<int,3> margin()const{
            int w = static_cast<int>(3.0*maxSigma_ + 1.5 + 0.5);
            return vigra::TinyVector<int,3>(w);

        }

        size_t nFeatures()const{
            return nFeatures_;
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


            typedef vigra::TinyVector<FType, 3> TV3;
            vigra::MultiArray<3, TV3>  gradient(data.shape());

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
                if(sigmaPresmooth>currentSigma){
                    vigra::gaussianSmoothMultiArray(gaussSmoothed,gaussSmoothed,
                        ConvOpts().stdDev(sigmaPresmooth).resolutionStdDev(currentSigma));

                    // remember current sigma
                    currentSigma = sigmaPresmooth;
                }

                /////////////////////////////////
                // filters on pre-smoothed
                /////////////////////////////////
                ConvOpts  opts = ConvOpts().stdDev(desiredSigma).resolutionStdDev(sigmaPresmooth).subarray(roiBegin,roiEnd);

                // LaplacianOfGaussian
                if(featureSelection_(si,size_t(FuncIndex::LaplacianOfGaussian))){
                    
                    vigra::laplacianOfGaussianMultiArray(gaussSmoothed, scalarCoreBuffer, opts);
                    extractor.store(scalarCoreBuffer);
                }

 
                // HessianOfGaussian
                if(featureSelection_(si,size_t(FuncIndex::HessianOfGaussianEigenvalues))){
                    vigra::hessianOfGaussianMultiArray(gaussSmoothed, core6Buffer, opts); 
                    vigra::tensorEigenvaluesMultiArray(core6Buffer, core3Buffer);
                    extractor.store(core3Buffer);
                }

                if(false){
                    // new structureTensorAndGradientMagnitude
                    structureTensorAndGradientMagnitude(gaussSmoothed, scalarCoreBuffer, core6Buffer,
                        ConvOpts().stdDev(desiredSigma).resolutionStdDev(sigmaPresmooth)
                                  .subarray(roiBegin,roiEnd).outerScale(desiredSigma*2.0)            
                    );   
                    extractor.store(scalarCoreBuffer); // squared gradient magnitude
                    vigra::tensorEigenvaluesMultiArray(core6Buffer, core3Buffer);
                    extractor.store(core3Buffer);         
                }
                if(true){
                    // GaussianGradientMagnitude
                    if(featureSelection_(si,size_t(FuncIndex::GaussianGradientMagnitude))){
                        vigra::gaussianGradientMagnitude(gaussSmoothed, scalarCoreBuffer, opts);
                        extractor.store(scalarCoreBuffer);
                    }

                    // Structure Tensor sigma*2
                    if(featureSelection_(si,size_t(FuncIndex::StructureTensorEigenvaluesS2))){
                        vigra::structureTensorMultiArray(gaussSmoothed, core6Buffer,
                            ConvOpts().stdDev(desiredSigma).resolutionStdDev(sigmaPresmooth)
                                      .subarray(roiBegin,roiEnd).outerScale(desiredSigma*2.0)
                        ); 
                        vigra::tensorEigenvaluesMultiArray(core6Buffer, core3Buffer);
                        extractor.store(core3Buffer);
                    }
                    // Structure Tensor sigma*4
                    if(featureSelection_(si,size_t(FuncIndex::StructureTensorEigenvaluesS4))){
                        vigra::structureTensorMultiArray(gaussSmoothed, core6Buffer,
                            ConvOpts().stdDev(desiredSigma).resolutionStdDev(sigmaPresmooth)
                                      .subarray(roiBegin,roiEnd).outerScale(desiredSigma*4.0)
                        ); 
                        vigra::tensorEigenvaluesMultiArray(core6Buffer, core3Buffer);
                        extractor.store(core3Buffer);
                    }
                }
                
                // GaussianSmoothing
                if(featureSelection_(si,size_t(FuncIndex::GaussianSmoothing))){
                    vigra::gaussianSmoothMultiArray(gaussSmoothed, gaussSmoothed,
                        ConvOpts().stdDev(desiredSigma).resolutionStdDev(sigmaPresmooth)
                    );
                    extractor.store(gaussSmoothed.subarray(roiBegin,roiEnd));

                    // remember current sigma
                    currentSigma = desiredSigma;
                }

            }
        }



    private:
        std::vector<float> sigmas_;
        vigra::MultiArray<2, bool> featureSelection_;
        size_t nFeatures_;
        float maxSigma_;
    };


    class SlicFeatureOp{

    public:
        typedef  vigra::ConvolutionOptions<3> ConvOpts;
        typedef vigra::TinyVector<vigra::UInt32, 3> Coord;
        typedef vigra::MultiArrayView<4, float> Shape4;
        SlicFeatureOp(
            const std::vector<unsigned int> & seedDistances = {5,10, 15},
            const std::vector<double> & intensityScalings = {10.0, 20.0, 30.0}
        )
        :    seedDistances_(seedDistances),
             intensityScalings_(intensityScalings)
        {


        }


        vigra::TinyVector<int,3> margin()const{
            return vigra::TinyVector<int,3>(seedDistances_.back()*2+3);
        }

        size_t nFeatures()const{
            return seedDistances_.size()*intensityScalings_.size();
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

            vigra::MultiArray<3, vigra::UInt32>  labelBuffer(data.shape());
            vigra::MultiArray<3, float>  valueBuffer(data.shape());
            for(size_t sdi=0; sdi<seedDistances_.size(); ++sdi){
                const unsigned int seedDist = seedDistances_[sdi];
                for(size_t isi=0; isi<intensityScalings_.size(); ++isi){
                    const double intScaling = intensityScalings_[isi];

                    unsigned int maxLabel = vigra::slicSuperpixels(data, labelBuffer, intScaling, seedDist, 
                                                                   vigra::SlicOptions().iterations(40));


                    typedef vigra::acc::Select<vigra::acc::DataArg<1>, vigra::acc::LabelArg<2>, vigra::acc::Mean> Statistics;
                    typedef vigra::acc::AccumulatorChainArray<vigra::CoupledArrays<3, T_IN, vigra::UInt32 >, Statistics> RegionFeatures;
                    RegionFeatures clusters_;

                    vigra::acc::extractFeatures(data, labelBuffer, clusters_);

                    for(size_t z=0; z<data.shape(2); ++z)
                    for(size_t y=0; y<data.shape(1); ++y)
                    for(size_t x=0; x<data.shape(0); ++x){
                        valueBuffer(x,y,z) = vigra::acc::get<vigra::acc::Mean>(clusters_,labelBuffer(x,y,z));
                    }

                    extractor.store(valueBuffer.subarray(roiBegin,roiEnd));
                }
            }
        }


    private:
        std::vector<unsigned int> seedDistances_;  
        std::vector<double> intensityScalings_;
    };

}

#endif //SKNEURO_VOXEL_PREDICTION_TOOLS_HXX
