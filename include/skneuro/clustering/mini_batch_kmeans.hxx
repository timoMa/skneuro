#ifndef SKNEURO_MINI_BATCH_KMEANS
#define SKNEURO_MINI_BATCH_KMEANS




template<class T, class L>
class MiniBatchKMeans{

public:
    typedef vigra::MultiArrayView<2, T> Features;
    typedef double NumericType;
    typedef vigra::MultiArrayView<2, NumericType> Centers;
    struct Parameter{
        size_t numberOfCluters_;
        size_t maxIter_;
        size_t miniBatchSize_;
    };


    MiniBatchKMeans(const Parameter & param)
    : param_(param)
    {

    }

    void fit(const Features & X , const Centers & startingPoint){

    }

private:
    Parameter & param_;
};







#endif /*SKNEURO_MINI_BATCH_KMEANS*/
