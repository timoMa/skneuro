#ifndef SKNEURO_MINI_BATCH_EM
#define SKNEURO_MINI_BATCH_EM


#include <skneuro/skneuro.hxx>

#include <vigra/accumulator.hxx>

#include <vigra/linear_algebra.hxx>

#include <boost/math/distributions/normal.hpp>



namespace skneuro{
namespace clustering{

namespace vacc = vigra::acc;






template<class T>
T multivariate_normal_distribution_density2(
    vigra::MultiArrayView<1,T> mean,
    vigra::MultiArrayView<1,T> variance,
    vigra::MultiArrayView<1,T> observation
){
    const size_t dim = mean.size();

    // compute covariance det.
    long double covarDet = static_cast<long double>(1);
    for(size_t i=0; i<dim; ++i){
        covarDet *= static_cast<long double>(variance[i]);
    }

    // compute argument of exp. function
    long double expArg = static_cast<long double>(0);
    for(size_t i=0; i<dim; ++i){
        const long double m = static_cast<long double>(mean[i]);
        const long double v = static_cast<long double>(variance[i]);
        const long double o = static_cast<long double>(observation[i]);
        const long double diff = o-m;
        const long double diffdiff = diff*diff;
        expArg+=(diffdiff)/v;
    }
    expArg*=static_cast<long double>(-0.5);
    const long double twoPiPowDim = std::pow(static_cast<long double>(2.0)*static_cast<long double>(M_PI),dim);
    const long double normalization = std::sqrt(twoPiPowDim*covarDet);
    return std::exp(expArg)/normalization;
}   



template <typename T>
T normal_pdf(T x, T m, T s)
{
    static const T inv_sqrt_2pi = 0.3989422804014327;
    T a = (x - m) / s;

    return inv_sqrt_2pi / s * std::exp(-T(0.5) * a * a);
}


template <typename T>
T normal_pdf2(T x, T m, T s)
{
    boost::math::normal_distribution<long double> nd(m,s);
    return boost::math::pdf(nd,x);
}



long double g1d(const long double mean,const long double sigma, const long double x){
    return std::exp(-0.5*std::pow((x-mean)/sigma,2.0))/
            (sigma*std::sqrt(2.0*M_PI));
}

template<class T>
T multivariate_normal_distribution_density(
    vigra::MultiArrayView<1,T> mean,
    vigra::MultiArrayView<1,T> variance,
    vigra::MultiArrayView<1,T> observation
){
    long double res=1.0;
    for(size_t i=0; i<mean.size(); ++i){
        //long double tmpRes = g1d( mean[i], std::sqrt(variance[i]), observation[i] );
        long double tmpRes = normal_pdf(observation[i], mean[i], std::sqrt(variance[i]) );
        if(tmpRes>1.0){
            std::cout<<"mean "<<mean[i]<<"  variance "<<variance[i]<<" observation "<<observation[i]<<" p "<<tmpRes<<"\n";
        }

        res*=tmpRes;
    }
    return res;
}   








template<class T>
class DiagonalMultivariateGaussian{

public:
    typedef vigra::MultiArrayView<1,T> ObervationType;

    DiagonalMultivariateGaussian(const size_t nFeatures, const T minVariance = 0.2)
    :   nFeatures_(nFeatures),
        acc_(),
        nUpdates_(0),
        sqrtCovarianceDet_(),
        minVariance_(minVariance),
        piFac_(std::pow(2.0*M_PI,  static_cast<T>(nFeatures)/2))
    {
        SKNEURO_CHECK_OP(acc_.passesRequired(),==,1,"internal error");
    }

    /// get the probability density
    /// for a given observation
    T density(const ObervationType & observation)const{

        SKNEURO_CHECK_OP(nUpdates_,>=,this->minUpdates(),
            "more updates are needed to make a density estimation");
        //std::cout<<"density sqrtCovarianceDet_ "<<sqrtCovarianceDet_<<"\n\n";
        // extract mean and variance
        const vigra::MultiArrayView<1,T> mean = vacc::get< vacc::Mean>(acc_);
        const vigra::MultiArrayView<1,T> variances = vacc::get< vacc::Variance>(acc_);

        vigra::MultiArray<1,T> newVar = variances;

        for(size_t i=0; i<nFeatures_; ++i){
            const T var = newVar(i);
            if(std::isinf(var) || std::isnan(var)){
                newVar(i)=0.2;
            }
            else if(var<=0.2){
                newVar(i)=0.2;
            }
        
            newVar(i)*=10.0;

        }

        return multivariate_normal_distribution_density(mean,newVar,observation);
        /*
        T tmp = 1.0;
        for(size_t i=0; i<nFeatures_; ++i){
            T var = nUpdates_ > 10 ?  variances(i)*10.0 : 1.0;
            var = (var > minVariance_ ? var : minVariance_);
            tmp*=(var*var);
        }
        tmp=std::sqrt(tmp);


        T expArg = 0.0;
        for(size_t i=0; i<nFeatures_; ++i){
            T var = nUpdates_ > 10 ?  variances(i)*10.0 : 1.0;
            var = (var > minVariance_ ? var : minVariance_);
            const T  diff  = mean(i)-observation(i);
            const T diffdiff = diff*diff;
            if( !(diffdiff>=0) ){
                std::cout<<"mean(i) "<<mean(i)<<" observation(i) "<<observation(i)<<" dd "<<diffdiff<<"\n\n";
            }
            SKNEURO_CHECK_OP(diffdiff,>=,0.0,"");
            expArg+= ((diff*diff)/(var*var));
            //std::cout<<"var "<<var<<" diff "<<diff<<" exp "<<expArg<<"\n\n";

        }
        expArg*=static_cast<T>(-0.5);
        //std::cout<<"piFac_ "<<piFac_<<" sqrtCovarianceDet_ "<<sqrtCovarianceDet_<<" exp "<<expArg<<"\n\n";
        return std::exp(expArg)/(piFac_*tmp);
        */
    }

    /// update gaussian parameters from
    /// an instance with given weight
    void updateFrom(const ObervationType & observation, const double weight){
        SKNEURO_CHECK_OP(observation.shape(0),==,nFeatures_," ");
        //std::cout<<"updateFrom   sqrtCovarianceDet "<<sqrtCovarianceDet_<<"\n";
        // do the accumation
        acc_(observation,weight);
        ++nUpdates_;
    }

    /// minimum number of updates which is needed
    /// to make a density prediction
    size_t minUpdates()const{
        return 1;
    }

    /// reset the accumulation
    void reset(){
        acc_.reset();
        nUpdates_(0);
    }

    /// disable update modus
    void lock(){
        // extract variance
        const vigra::MultiArrayView<1,T> variances = vacc::get< vacc::Variance>(acc_);
        
        T tmp = 1.0;
        for(size_t i=0; i<nFeatures_; ++i){
            const T var = nUpdates_ > 10 ?  variances(i)*10.0 : 1.0;
            tmp*= var > minVariance_ ? var*var : minVariance_;
        }
        sqrtCovarianceDet_=std::sqrt(tmp);
        //if(sqrtCovarianceDet_<0.0001){
        //    sqrtCovarianceDet_=0.0001;
       // }
        //std::cout<<"sqrtCovarianceDet "<<sqrtCovarianceDet_<<"\n";
    }   

    /// enable update modus 
    void unlock(){

    }

private:
    typedef vacc::AccumulatorChain< ObervationType,
        vacc::Select<vacc::Mean,vacc::Variance> > AccType;


    size_t nFeatures_;
    AccType acc_;
    size_t nUpdates_;
    T sqrtCovarianceDet_;
    T minVariance_;
    T piFac_;
};


    

template<class T>
class MiniBatchEm{
    typedef T value_type;   
    typedef vigra::MultiArrayView<1,value_type> DataType;
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
    miniBatchFuzzyLabels_( typename vigra::MultiArrayView<2,value_type>::difference_type( nClusters+1,miniBatchSize) ),
    gaussians_(nClusters, DiagonalMultivariateGaussian<value_type>(nFeatures) )
    {

    }



    void run(const vigra::MultiArrayView<2,value_type> features){

        if(features.shape(1)<=miniBatchSize_){
            throw std::runtime_error("features must be larger than mini batch size");
        }
        features_ = features;

        for(size_t i=0; i<nIter_; ++i){
            std::cout<<"i="<<i<<"\n";
            std::cout<<"lock\n";
            this->lockGaussians();
            std::cout<<"get mini batch indices\n";
            this->getMiniBatchIndices();
            std::cout<<"get fuzzy labels\n";
            this->getFuzzyLabels();
            std::cout<<"take gradient step\n";
            this->takeGradientStepA();
        }

    }   

    //const vigra::MultiArray<2,value_type> & clusterCenters()const{
    //    return clusterMean_;
    //}

    size_t nClusters()const{
        return nClusters_;
    }

    void predict(
        const vigra::MultiArrayView<2,value_type>  & features,
        vigra::MultiArrayView<2,value_type>        & probabilities
    )const{
        //#pragma omp parallel for
        for(size_t xi=0; xi< features.shape(1); ++xi){
            value_type psum=0.0;
            vigra::MultiArrayView<1,value_type> observation = features. template bind<1>(xi);
            for(size_t ci=0; ci< nClusters_; ++ci){
                value_type prob = gaussians_[ci].density(observation);
                probabilities(ci,xi)=prob;
                psum+=prob;
            }
            for(size_t ci=0; ci< nClusters_; ++ci){
                probabilities(ci,xi)/=psum;
            }
        }
    }
    void initalizeCenters(const vigra::MultiArray<2,value_type> & centers){
        //clusterMean_=centers;
        for(size_t c=0; c<nClusters_; ++c){
            DataType data = centers. template bind<1>(c);
            gaussians_[c].updateFrom(data,1.0);
        }
    }
private:
    void lockGaussians(){
        //#pragma omp parallel for
        for(size_t c=0; c<nClusters_; ++c){
            gaussians_[c].lock();
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
    

    void getFuzzyLabels(){
        //#pragma omp parallel for
        for(size_t mbi=0;mbi<miniBatchSize_;++mbi){
            value_type psum=0.0;
            for(size_t ci=0; ci<nClusters_; ++ci ){   
                const value_type prob=clusterProbabiliy(ci,mbi);  
                miniBatchFuzzyLabels_(ci,mbi)=prob;
                psum+=prob;
                SKNEURO_CHECK_OP(prob,>,-0.000001,"");
                SKNEURO_CHECK_OP(prob,<, 1.000001,"");
                SKNEURO_CHECK( !std::isnan(prob),"");
                SKNEURO_CHECK( !std::isinf(prob),"");
            }

            for(size_t ci=0; ci<nClusters_; ++ci ){   
                miniBatchFuzzyLabels_(ci,mbi)/=psum;
            }
        }
    }



    value_type clusterProbabiliy(const size_t ci, const size_t mbi){
        const size_t xi=miniBatchIndices_[mbi];
        vigra::MultiArrayView<1,value_type> observation = features_. template bind<1>(xi);
        return gaussians_[ci].density(observation);
    }



    void takeGradientStepA(){

        for(size_t mbi=0; mbi<miniBatchSize_; ++mbi){

            // get observation
            const size_t xi=miniBatchIndices_[mbi];
            vigra::MultiArrayView<1,value_type> observation = features_. template bind<1>(xi);


            T maxProb = -1.0;
            size_t maxProbC  = 0;

            for(size_t c=0;c<nClusters_;++c){

                // assignment probability
                const value_type prob = miniBatchFuzzyLabels_(c,mbi);

                if(prob>maxProb){
                    maxProb=prob;
                    maxProbC=c;
                }

                // weighted accumulation
                
                //gaussians_[c].updateFrom(observation,prob);
            }
            gaussians_[maxProbC].updateFrom(observation,1.0);
        }
    }




    size_t nFeatures_;
    size_t nClusters_;
    size_t miniBatchSize_;
    size_t nIter_;
    vigra::MultiArrayView<2,value_type> features_;
    vigra::ArrayVector<size_t>          miniBatchIndices_;
    vigra::MultiArray<2,value_type>     miniBatchFuzzyLabels_;
    std::vector< DiagonalMultivariateGaussian<value_type> > gaussians_;




};

} // end clustering
} // end namespace skneuro


#endif // SKNEURO_MINI_BATCH_EM
