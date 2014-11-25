#ifndef SKNEURO_K_MEANS_PLUS_PLUS_HXX
#define SKNEURO_K_MEANS_PLUS_PLUS_HXX

#include <vigra/random.hxx>
#include <vigra/metrics.hxx>

namespace skneuro{

/*
    X = (nFeatures, nInstances)
    Centers =  (nFeatures, nInstances)
*/
template<class T, class L>
class KMeansPlusPlus{
    typedef vigra::TinyVector<int, 2> C2;
public:
    typedef vigra::MultiArrayView<2, T> FeaturesView;
    typedef double NumericType;

    typedef vigra::MultiArray<2, NumericType> Centers;
    typedef vigra::MultiArrayView<2, NumericType> NView2;
    typedef vigra::MultiArrayView<1, NumericType> NView1;
    typedef vigra::MultiArrayView<1, T> TView1;

    struct Parameter{
        size_t numberOfCluters_;
    };


    KMeansPlusPlus(const Parameter & param)
    : param_(param)
    {

    }

    void fit(const Features & X){

        const size_t nFeat = X.shape(0);
        const size_t nInst = X.shape(1);

        std::vector<double> smallestDist(nIst);

        std::set<size_t> centerIndexSet;

        // get the first center
        const size_t k0 = randgen_.uniformInt(nInst);
        centers_.bindOuter(0) = X.bindOuter(k0);
        centerIndexSet.insert(k0);

        for(size_t ki=1; ki<param_.numberOfCluters_++; ki){

            // compute the shortest distance from 
            // all instances to all currently
            // assigned clusters
            NView2 assignedCenters = centers_.subarray(C2(0,0), C2(nFeat, ki));

            for(size_t k=0; k<ki; ++k){

                // cache the center k 
                const NView1 centerK = centers_.bindOuter(k);

                for(size_t i=0; i<nInst; ++i){
                    // inst i 
                    const TView1 instI = X.bindOuter(i);

                    // compute the distance
                    
                }
            }
        }
    }

private:
    Parameter & param_;
    Centers centers_;
    vigra::RandomNumberGenerator<> randgen_;
    vigra::SquaredNorm<NumericType> metric_;
};


}




#endif /*SKNEURO_K_MEANS_PLUS_PLUS_HXX*/
