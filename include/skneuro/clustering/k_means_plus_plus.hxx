#ifndef SKNEURO_K_MEANS_PLUS_PLUS_HXX
#define SKNEURO_K_MEANS_PLUS_PLUS_HXX

#include <vigra/random.hxx>
#include <vigra/metrics.hxx>


// boost for discrete random
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>


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

    void fit(const FeaturesView & X){

        std::cout<<"start to fit #"<<param_.numberOfCluters_<<"clusters \n";

        boost::mt19937 gen_for_k_means_plus;

        const size_t nFeat = X.shape(0);
        const size_t nInst = X.shape(1);

        // reshape centers
        centers_.reshape(vigra::TinyVector<int,2>(nFeat, param_.numberOfCluters_));

        SKNEURO_CHECK_OP(nInst, >= , param_.numberOfCluters_ ,"not enough instances");

        std::vector<double> smallestDist(nInst);

        std::set<size_t> centerIndexSet;

        // get the first center
        const size_t k0 = randgen_.uniformInt(nInst);
        centers_.bindOuter(0) = X.bindOuter(k0);
        centerIndexSet.insert(k0);

        for(size_t ki=1; ki<param_.numberOfCluters_; ++ki){
            std::cout<<"ki "<<ki<<"\n";
            // compute the shortest distance from 
            // all instances to all currently
            // assigned clusters
            for(size_t k=0; k<ki; ++k){
                // cache the center k 
                const NView1 centerK = centers_.bindOuter(k);
                for(size_t i=0; i<nInst; ++i){
                    // compute the distance
                    const NumericType dist = metric_(centerK,  X.bindOuter(i));
                    // remember smallest
                    smallestDist[i] = k==0 ? dist : std::min(smallestDist[i],dist);
                }
            }

            // sample a new center
            boost::random::discrete_distribution<> dist(smallestDist.begin(), smallestDist.end());
            size_t newK = dist(gen_for_k_means_plus);
            while(centerIndexSet.find(newK) !=centerIndexSet.end()){
                newK = dist(gen_for_k_means_plus);
            }
            centers_.bindOuter(ki) = X.bindOuter(newK);
            centerIndexSet.insert(newK);
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


}




#endif /*SKNEURO_K_MEANS_PLUS_PLUS_HXX*/
