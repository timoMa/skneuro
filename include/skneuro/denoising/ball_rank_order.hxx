#ifndef SKNEURO_BALL_RANK_ORDER_FILTER_HXX
#define SKNEURO_BALL_RANK_ORDER_FILTER_HXX

/*vigra*/
#include <vigra/multi_array.hxx>
#include <vigra/gaussians.hxx>
#include <vigra/algorithm.hxx>

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
        CoordType cc(vigra::SkipInitialization);
        CoordType oc(vigra::SkipInitialization);

        std::vector<unsigned char> mask(std::pow(size,3),0);
        {
            size_t i=0;
            for(cc[2]=-radius; cc[2]<radius+1; ++cc[2])
            for(cc[1]=-radius; cc[1]<radius+1; ++cc[1])
            for(cc[0]=-radius; cc[0]<radius+1; ++cc[0], ++i){
                const float r = vigra::norm(cc);
                if(r<=radius)
                    mask[i]=1;
            }
        }   


        
 
        #pragma omp parallel for  //private(buffer)
        for(int z=0; z<image.shape(2); ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=0; c[1]<image.shape(1); ++c[1]){
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
                
                const float size = ii;
                const float rankPos = size*rankOrder;
                const int rLow = std::floor(rankPos); 
                const int rUp = rLow+1;
                const float m = rankPos - rLow;
                out[c] = buffer[rLow]*(1.0-m) + buffer[rUp]*(m);

            }
            }
        }

    }
    /*
    template<class T>
    void gaussianBallRankOrderFilter(
        const vigra::MultiArrayView<3,T> & image,
        const int radius,
        const float rankOrder,
        const float sigma,
        vigra::MultiArrayView<3,T>  & out
    ){
        typedef typename vigra::MultiArrayView<3,T>::difference_type CoordType;
        const int size = 2*radius + 1;

        CoordType shape = image.shape();
        CoordType rad(radius);
        CoordType cc(vigra::SkipInitialization);
        CoordType oc(vigra::SkipInitialization);

        vigra::Gaussian<float> gaussian(sigma); 
        std::vector<unsigned char> mask(std::pow(size,3),0);
        std::vector<float> kernel(std::pow(size,3),0);
        {
            size_t i=0;
            float sum = 0;
            for(cc[2]=-radius; cc[2]<radius+1; ++cc[2])
            for(cc[1]=-radius; cc[1]<radius+1; ++cc[1])
            for(cc[0]=-radius; cc[0]<radius+1; ++cc[0], ++i){
                const float r = vigra::norm(cc);
                if(r<=radius){
                    mask[i]=1;
                    kernel[i] = gaussian(r);
                    sum+=kernel[i];
                }
            }
            
            for(cc[2]=-radius; cc[2]<radius+1; ++cc[2])
            for(cc[1]=-radius; cc[1]<radius+1; ++cc[1])
            for(cc[0]=-radius; cc[0]<radius+1; ++cc[0], ++i){
                const float r = vigra::norm(cc);
                if(r<=radius){
                    mask[i]=1;
                    kernel[i]/sum;
                }
            }
        }   


        
 
        #pragma omp parallel for  //private(buffer)
        for(int z=0; z<image.shape(2); ++z){

            std::vector<T> valBuffer(std::pow(size,3));
            std::vector<T> weightBuffer(std::pow(size,3));
            std::vector<size_t> indexBuffer(std::pow(size,3));

            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=0; c[1]<image.shape(1); ++c[1]){
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
                        valBuffer[ii] = image[oc];
                        weightBuffer[ii] = kernel[ii];
                        ++ii;
                    }
                }

                // normalize weights and fill indices
                float wSum  0 ;
                for (int i = 0; i < ii; ++i){
                    indexBuffer[i] = i;
                    wSum += weightBuffer[i];
                }
                for (int i = 0; i < ii; ++i){
                    indexBuffer[i] = i;
                    weightBuffer[i] /= wSum;
                }

                vigra::indexSort(valBuffer.first(), valBuffer.begin()+ii, indexBuffer.begin());

                

                const float size = ii;
                const float rankPos = size*rankOrder;
                const int rLow = std::floor(rankPos); 
                const int rUp = rLow+1;
                const float m = rankPos - rLow;
                out[c] = buffer[rLow]*(1.0-m) + buffer[rUp]*(m);


            }
            }
        }

    }
    */

}


#endif /* SKNEURO_BALL_RANK_ORDER_FILTER_HXX */
