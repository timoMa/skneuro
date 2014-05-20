#ifndef SKNEURO_MINI_BATCH_EM
#define SKNEURO_MINI_BATCH_EM


#include <skneuro/skneuro.hxx>

namespace skneuro{
namespace clustering{

template<class T>
class MiniBatchEm{
    typedef T value_type;   
public:

    MiniBatchEm(){}

    MiniBatchEm(const size_t nFeatures,const size_t nClusters,const size_t miniBatchSize, const size_t nIter)
    :
    nFeatures_(nFeatures),
    nClusters_(nClusters),
    miniBatchSize_(miniBatchSize),
    nIter_(nIter),
    features_( ),
    miniBatchIndices_(miniBatchSize),
    miniBatchLabels_(miniBatchSize),
    clusterMean_( typename vigra::MultiArrayView<2,value_type>::difference_type( nFeatures,nClusters) ),
    clusterAccVar_( typename vigra::MultiArrayView<2,value_type>::difference_type( nFeatures,nClusters) ),
    clusterVar_( typename vigra::MultiArrayView<2,value_type>::difference_type( nFeatures,nClusters) ),
    clusterVarDet_( typename vigra::MultiArrayView<1,value_type>::difference_type( nClusters) ),
    assignmentCounter_(nClusters)
    {
        std::fill(assignmentCounter_.begin(),assignmentCounter_.end(),0);
        std::fill(clusterAccVar_.begin(),clusterAccVar_.end(),0.0);
    }



    void run(const vigra::MultiArrayView<2,value_type> features){

        if(features.shape(1)<=miniBatchSize_){
            throw std::runtime_error("features must be larger than mini batch size");
        }
        features_ = features;

        for(size_t i=0; i<nIter_; ++i){
            std::cout<<"i="<<i<<"\n";
            this->varToStd();
            this->getMiniBatchIndices();
            this->findNearestCenter();
            this->takeGradientStepA();
        }

    }   

    const vigra::MultiArray<2,value_type> & clusterCenters()const{
        return clusterMean_;
    }

    size_t nClusters()const{
        return nClusters_;
    }

    void predict(
        const vigra::MultiArrayView<2,value_type>  & features,
        vigra::MultiArrayView<2,value_type>        & probabilities
    )const{
        #pragma omp parallel for
        for(size_t xi=0; xi< features.shape(1); ++xi){

            value_type psum=0.0;
            for(size_t ci=0; ci< nClusters_; ++ci){

                value_type acc=0;
                for(size_t f=0; f<nFeatures_; ++f){
                    acc += std::pow(clusterMean_(f,ci)-features(f,xi),2)/clusterVar_(f,ci);
                }
                acc*=static_cast<value_type>(-0.5);
                acc = std::exp(acc);
                acc/=std::pow(2*3.14159265359,value_type(nFeatures_)/2.0);
                acc/=std::sqrt(clusterVarDet_(ci));
                probabilities(ci,xi)=acc;
                psum+=acc;
            }
            for(size_t ci=0; ci< nClusters_; ++ci){
                probabilities(ci,xi)/=psum;
            }
        }

    }
    void initalizeCenters(const vigra::MultiArray<2,value_type> & centers){
        clusterMean_=centers;
    }
private:
    void varToStd(){
        #pragma omp parallel for
        for(size_t c=0; c<nClusters_; ++c){

            const size_t count=assignmentCounter_[c];
            if(count>=2){
                long double prod = 1.0;
                for(size_t f=0; f<nFeatures_; ++f){
                    SKNEURO_CHECK_OP( clusterAccVar_(c),>,0,"" );
                    const long double  cVar = 42.0*static_cast<long double>(clusterAccVar_(f,c)/
                        (static_cast<value_type>(count)-1));

                    clusterVar_(f,c)=cVar;
                    SKNEURO_CHECK( !std::isnan(clusterVar_(f,c)),"" );
                    prod*=cVar;
                    SKNEURO_CHECK_OP( prod,>,0,"" );
                }
                clusterVarDet_(c)=prod;
                clusterVarDet_(c) = std::max(clusterVarDet_(c),value_type(0.01));
                SKNEURO_CHECK_OP( clusterVarDet_(c),>,0,"" );
                SKNEURO_CHECK( !std::isnan(clusterVarDet_(c)),"" );
                SKNEURO_CHECK( !std::isinf(clusterVarDet_(c)),"" );
            }
            else{
                for(size_t f=0; f<nFeatures_; ++f){
                    clusterVar_(f,c)=1.0;
                }
                clusterVarDet_(c)=1.0;

                SKNEURO_CHECK( !std::isnan(clusterVarDet_(c)),"" );
            }
        }
    }




    void getMiniBatchIndices(){
        std::vector<size_t> allIncides(features_.shape(1));
        for(size_t i=0;i<features_.shape(1);++i){
            allIncides[i]=i;
        }
        std::random_shuffle(allIncides.begin(), allIncides.end());
        std::copy(allIncides.begin(), allIncides.begin()+miniBatchSize_, miniBatchIndices_.begin());
    }
    

    void findNearestCenter(){
        #pragma omp parallel for
        for(size_t mbi=0;mbi<miniBatchSize_;++mbi){
            
            value_type bestProb = -1.0;
            size_t bestCluster = 0;
            //std::cout<<"mbi "<<mbi<<"\n";

            for(size_t ci=0; ci<nClusters_; ++ci ){   
                const value_type prob=clusterProbabiliy(ci,mbi);  
                //std::cout<<"   ci "<<ci<<" "<<prob<<"\n";
              
                SKNEURO_CHECK_OP(prob,>,-0.000001,"");
                SKNEURO_CHECK_OP(prob,<, 1.000001,"");
                SKNEURO_CHECK( !std::isnan(prob),"");
                SKNEURO_CHECK( !std::isinf(prob),"");
                if(prob>bestProb){
                    bestProb=prob;
                    bestCluster=ci;
                }
            }
            assignmentCounter_[bestCluster]+=1;
            miniBatchLabels_[mbi]=bestCluster;
        }
    }



    value_type clusterProbabiliy(const size_t ci, const size_t mbi){

        const size_t xi=miniBatchIndices_[mbi];
        value_type acc=0;
        for(size_t f=0; f<nFeatures_; ++f){
            acc += std::pow(clusterMean_(f,ci)-features_(f,xi),2)/clusterVar_(f,ci);
        }
        SKNEURO_CHECK( !std::isnan(acc),"");
        SKNEURO_CHECK( !std::isinf(acc),"");
        acc=std::max(value_type(0.0000001),acc);
        SKNEURO_CHECK_OP( acc,>,0.0,"");
        acc*=static_cast<value_type>(-0.5);
        SKNEURO_CHECK_OP( acc,<,0.0,"");
        acc = std::exp(acc);


        acc/=std::pow(2*3.14159265359,value_type(nFeatures_)/2.0);

        SKNEURO_CHECK( !std::isnan(acc),"");
        SKNEURO_CHECK( !std::isinf(acc),"");

        acc/=std::sqrt(clusterVarDet_(ci));

        SKNEURO_CHECK( !std::isnan(acc),"");
        SKNEURO_CHECK( !std::isinf(acc),"");

        return std::max(acc,value_type(0.0));
    }



    void takeGradientStepA(){

        for(size_t i=0;i<miniBatchSize_;++i){
            // get cached label
            const size_t cachedLabel = miniBatchLabels_[i];

            // get learning rate
            const size_t counter = assignmentCounter_[cachedLabel];

            //std::cout<<"mbi "<<i<< " label "<<cachedLabel<<" count "<<counter<<"\n";

            SKNEURO_CHECK_OP(counter,>,0,"");
            const value_type lRate = static_cast<value_type>(1.0)/assignmentCounter_[cachedLabel];

            // take gradient step
            if(counter==1){
                for(size_t f=0;f<nFeatures_;++f){

                    const value_type xMean      = features_(f,miniBatchIndices_[i]);
                    clusterMean_(f,cachedLabel) = xMean;
                    clusterAccVar_(f,cachedLabel) = 1.0;
                }
            }
            else{
                #pragma omp parallel for
                for(size_t f=0;f<nFeatures_;++f){



                    const value_type xMean      = features_(f,miniBatchIndices_[i]);
                    const value_type oldMean    = clusterMean_(f,cachedLabel);
                    const value_type diffOld    = xMean-oldMean;
                    const value_type newMean    = oldMean + diffOld*lRate;
                    const value_type diffNew    = xMean-newMean;
                    const value_type oldAccVar  = clusterAccVar_(f,cachedLabel);


                    SKNEURO_CHECK( !std::isnan(xMean),"");
                    SKNEURO_CHECK( !std::isinf(xMean),"");

                    SKNEURO_CHECK( !std::isnan(oldMean),"");
                    SKNEURO_CHECK( !std::isinf(oldMean),"");


                    SKNEURO_CHECK( !std::isnan(newMean),"");
                    SKNEURO_CHECK( !std::isinf(newMean),"");


                    //std::cout<<"old / new "<<oldMean <<  " / " <<newMean<<"\n";
                    //std::cout<<"diff "<<std::abs(diffOld*diffNew)<<"\n";

                    clusterMean_(f,cachedLabel) = newMean;
                    clusterAccVar_(f,cachedLabel)+= std::abs(diffOld*diffNew);
                }
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
    vigra::MultiArray<2,value_type>     clusterMean_;
    vigra::MultiArray<2,value_type>     clusterAccVar_;
    vigra::MultiArray<2,value_type>     clusterVar_;
    vigra::MultiArray<1,value_type>     clusterVarDet_;
    vigra::ArrayVector<size_t>          assignmentCounter_;
};

} // end clustering
} // end namespace skneuro


#endif // SKNEURO_MINI_BATCH_EM
