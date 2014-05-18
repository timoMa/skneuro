
#include <vigra/metrics.hxx>


namespace skneuro{


template<class T, class METRIC>
class MiniBatchKMeans{
    typedef T value_type;   
public:

    MiniBatchKMeans(){}

    MiniBatchKMeans(const size_t nFeatures,const size_t nClusters,const size_t miniBatchSize, const size_t nIter, const METRIC & metric)
    :
    nFeatures_(),
    nClusters_(nClusters),
    miniBatchSize_(miniBatchSize),
    nIter_(nIter_),
    features_( ),
    miniBatchLabels_(nFeatures_),
    clusterCenters_( typename vigra::MultiArrayView<2,value_type>::difference_type( nFeatures,nClusters) ),
    assignmentCounter_(nClusters_)
    {
        std::fill(assignmentCounter_.begin(),assignmentCounter_.end(),0);
    }



    void assignFeatures(const vigra::MultiArrayView<2,value_type> features){
        features_ = features;
    }   


    void takeSteps(){
        for(size_t i=0; i<nIter_; ++i){
            this->getMiniBatchIndices();
            this->findNearestCenter();
            this->takeGradientStepAsub();
        }
    }


private:
    void initalizeCenters(const vigra::MultiArray<2,value_type> & centers){
        clusterCenters_=centers;
    }

    void getMiniBatchIndices(){
        // todo
    }
    

    void findNearestCenter(){
        #pragma omp parallel for
        for(size_t i=0;i<miniBatchSize_;++i){

            vigra::ArrayVector<value_type> feature(nFeatures_,&features_(miniBatchIndices_[i],0));
            value_type minDistance = std::numeric_limits<value_type>::infinity();
            size_t minDistCluster  = nClusters_ ;
            for(size_t c=0;c<nClusters_;++c){
                vigra::ArrayVectorView<value_type> clusterFeature(nFeatures_,&clusterCenters_(c,0));
                value_type distance = metric_(feature,clusterFeature);
                if(distance<minDistance){
                    minDistance=distance;
                    minDistCluster = c;
                }
            }
            ++assignmentCounter_[minDistCluster];
            miniBatchLabels_[i] = minDistCluster;
        }
    }


    void takeGradientStepA(){
        for(size_t i=0;i<miniBatchSize_;++i){

            // get cached label
            const size_t cachedLabel = miniBatchLabels_[i];

            // get learning rate
            const value_type lRate = static_cast<value_type>(1)/assignmentCounter_[cachedLabel];
            const value_type ilRate = static_cast<value_type>(1)-lRate;

            // take gradient step
            #pragma omp parallel for
            for(size_t f=0;f<nFeatures_;++f){
                clusterCenters_(f,cachedLabel)= clusterCenters_(f,cachedLabel)*ilRate + features_(f,miniBatchIndices_[i])*lRate;
            }

        }

    }



    size_t nFeatures_;
    size_t nClusters_;
    size_t miniBatchSize_;
    size_t nIter_;
    vigra::MultiArrayView<2,value_type> features_;
    vigra::ArrayVector<size_t>          miniBatchIndices_;
    vigra::ArrayVector<size_t>          miniBatchLabels_;
    vigra::MultiArray<2,value_type>     clusterCenters_;
    vigra::ArrayVector<size_t>          assignmentCounter_;
    METRIC                              metric_;
};

} // end namespace skneuro