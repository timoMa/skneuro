#ifndef SKNEURO_MINI_BATCH_KMEANS
#define SKNEURO_MINI_BATCH_KMEANS



#include <vigra/random.hxx>
#include <vigra/metrics.hxx>


#include "k_means_plus_plus.hxx"


namespace skneuro{

/*
    X = (nFeatures, nInstances)
    Centers =  (nFeatures, nInstances)
*/
template<class T, class L>
class MiniBatchKMeans{
    typedef vigra::TinyVector<int, 2> C2;
public:
    typedef vigra::MultiArrayView<2, T> FeaturesView;
    typedef double NumericType;

    typedef vigra::MultiArray<2, NumericType> Centers;
    typedef vigra::MultiArrayView<2, NumericType> NView2;
    typedef vigra::MultiArrayView<1, NumericType> NView1;
    typedef vigra::MultiArrayView<1, T> TView1;
    typedef vigra::MultiArray<1, T> TCopy1;
    struct Parameter{
        enum InitType{
            KPP
        };
        Parameter(){
            initType_ = KPP;
            maxIter_ = 100;
            batchSize_ = 1000;
        }
        size_t numberOfCluters_;
        InitType initType_;
        size_t maxIter_;
        size_t batchSize_;
    };


    MiniBatchKMeans(const Parameter & param)
    : param_(param),
      centers_(),
      randgen_(),
      metric_()
    {

    }

    void fit(const FeaturesView & X){

        std::cout<<"MINI BATCH K MEANS with #"<<param_.numberOfCluters_<<"clusters \n";
        const size_t nFeat = X.shape(0);
        const size_t nInst = X.shape(1);

        SKNEURO_CHECK_OP(nInst, >= , param_.numberOfCluters_ ,"not enough instances");
        SKNEURO_CHECK_OP(nInst, >= , param_.batchSize_ ,"not enough instances");

        // reshape centers
        centers_.reshape(vigra::TinyVector<int,2>(nFeat, param_.numberOfCluters_));


        // initialization
        if(param_.initType_ == Parameter::KPP){
            typedef KMeansPlusPlus<T, L> Kmpp;
            typedef typename Kmpp::Parameter KmppParam;

            KmppParam kmppParam;
            kmppParam.numberOfCluters_ = param_.numberOfCluters_;
            Kmpp kmpp(kmppParam);
            kmpp.fit(X);

            centers_ = kmpp.centers();
        }
        
        std::vector<size_t> cachedCenters(param_.batchSize_);
        std::vector<size_t> miniBatchExamples(nInst);
        std::vector<size_t> perCenterCount(param_.numberOfCluters_, 0);
        for(size_t iter=0; iter<param_.maxIter_; ++iter){
            std::cout<<"iter "<<iter<<"\n";
            // pick mini batch examples
            for(size_t inst=0; inst<nInst; ++inst){
                miniBatchExamples[inst] = inst;
            }
            std::random_shuffle(miniBatchExamples.begin(), miniBatchExamples.end());


            // cache the closest center for 
            // each mini batch example
            for(size_t mi=0; mi<param_.batchSize_; ++mi){
                NumericType minDist = std::numeric_limits<NumericType>::infinity();
                size_t minCenter = 0;
                const TView1 instMi = X.bindOuter(miniBatchExamples[mi]);
                for(size_t ci=0; ci<param_.numberOfCluters_; ++ci){
                    const NumericType dist = metric_(instMi, centers_.bindOuter(ci));
                    if(dist < minDist){
                        minDist = dist;
                        minCenter = ci;
                    }
                }
                cachedCenters[mi] = minCenter;
            }

            // update
            for(size_t mi=0; mi<param_.batchSize_; ++mi){

                // get cached center
                size_t c = cachedCenters[mi];

                // increment par center count
                ++perCenterCount[c];

                // per center learning rate
                const NumericType n = 1.0 / static_cast<NumericType>(perCenterCount[c]);

                // gradient step
                NView1 centerC  = centers_.bindOuter(c);
                TCopy1 instanceMi = X.bindOuter(miniBatchExamples[mi]);
                instanceMi*=n;
                centerC *= (1.0 - n);
                centerC += instanceMi;
            }
        }
    }
    
    const Centers & centers(){
        return centers_;
    }; 

private:
    Parameter param_;
    Centers centers_;
    vigra::RandomNumberGenerator<> randgen_;
    vigra::metrics::SquaredNorm<NumericType> metric_;
};


template<class T, class C, class L, class M>
void findNearest(
    vigra::MultiArrayView<2, T> features,
    vigra::MultiArrayView<2, C> centers,
    const M & metric,
    vigra::MultiArrayView<1, L> nearestCenters
){
    for(size_t i=0; i<features.shape(1);++i){
        vigra::MultiArrayView<1, T> fi = features.bindOuter(i);

        double minDist = std::numeric_limits<double>::infinity();
        size_t minK = 0;

        for(size_t k=0; k<centers.shape(1); ++k){
            const double dist = metric(fi, centers.bindOuter(k));
            if(dist<minDist){
                minDist = dist;
                minK = k;
            }
        }
        nearestCenters[i] = minK;
    }
}




}







#endif /*SKNEURO_MINI_BATCH_KMEANS*/
