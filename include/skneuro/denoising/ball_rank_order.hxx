#ifndef SKNEURO_BALL_RANK_ORDER_FILTER_HXX
#define SKNEURO_BALL_RANK_ORDER_FILTER_HXX

/*vigra*/
#include <vigra/multi_array.hxx>

namespace skneuro{


    template<class T>
    void ballRankOrderFilter(
        const vigra::MultiArrayView<3,T> & image,
        const int radius,
        const float rankOrder,
        vigra::MultiArrayView<3,T>  & out
    ){
        typedef typename vigra::MultiArrayView<3,T>::difference_type CoordType;
        const int size = 2*radius + 1;

        CoordType shape = image.shape();
        CoordType rad(radius);
        CoordType c(vigra::SkipInitialization);
        CoordType oc(vigra::SkipInitialization);

        std::vector<unsigned char> mask(std::pow(size,3),0);
        {
            size_t i=0;
            for(c[2]=-radius; c[2]<radius+1; ++c[2])
            for(c[1]=-radius; c[1]<radius+1; ++c[1])
            for(c[0]=-radius; c[0]<radius+1; ++c[0], ++i){
                const float r = vigra::norm(c);
                if(r<=radius)
                    mask[i]=1;
            }
        }   


        std::vector<T> buffer(std::pow(size,3));

        for(c[2]=0; c[2]<image.shape(2); ++c[2])
        for(c[1]=0; c[1]<image.shape(1); ++c[1])
        for(c[0]=0; c[0]<image.shape(0); ++c[0]){

            CoordType s = c - rad;
            CoordType e = c + rad+1;
            CoordType oc(vigra::SkipInitialization);
            size_t i=0; 
            size_t ii=0;
            for(oc[2]=s[2]; oc[2]<e[2]; ++oc[2])
            for(oc[1]=s[1]; oc[1]<e[1]; ++oc[1])
            for(oc[0]=s[0]; oc[0]<e[0]; ++oc[0], ++i){

                if(mask[i]==1 && 
                   oc[0] >= 0       && oc[1] >= 0       && oc[2] >= 0 &&
                   oc[0] < shape[0] && oc[1] < shape[1] && oc[2] < shape[2]){
                    // inside mask and image
                    buffer[ii] = image[oc];
                    ++ii;
                }
            }

            std::sort(buffer.begin(), buffer.begin() + ii);
            //

        }

    }

}


#endif /* SKNEURO_BALL_RANK_ORDER_FILTER_HXX */