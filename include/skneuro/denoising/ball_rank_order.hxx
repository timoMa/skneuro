#ifndef SKNEURO_BALL_RANK_ORDER_FILTER_HXX
#define SKNEURO_BALL_RANK_ORDER_FILTER_HXX

/*skneuro*/
#include "skneuro/utilities/rank_order.hxx"

/*vigra*/
#include <vigra/multi_array.hxx>
#include <vigra/gaussians.hxx>
#include <vigra/algorithm.hxx>

namespace skneuro{


    template<class S, class DP, class OFFSETS>
    void getDisc(
        const int radius,
        const S & stride,
        DP & discPoints,
        OFFSETS & offsets
    )
    {
        typedef typename vigra::MultiArrayView<3,int>::difference_type CoordType;
        CoordType cc(vigra::SkipInitialization);
        for(cc[2]=-radius; cc[2]<radius+1; ++cc[2])
        for(cc[1]=-radius; cc[1]<radius+1; ++cc[1])
        for(cc[0]=-radius; cc[0]<radius+1; ++cc[0]){
            const float r = vigra::norm(cc);
            if(r<=radius){
                discPoints.push_back(cc);
                offsets.push_back(stride[0]*cc[0]+stride[1]*cc[1]+stride[2]*cc[2]);
            }
        }
    }





    template<class T>
    void ballRankOrderFilterNew(
        const vigra::MultiArrayView<3,T> & image,
        const int radius,
        const float rankOrder,
        const float minVal,
        const float maxVal,
        const float nBins,
        vigra::MultiArrayView<3,T>  & out
    ){
        typedef SortingRank<T> RankComp;
        typedef typename vigra::MultiArrayView<3,T>::difference_type CoordType;

        const CoordType shape = image.shape();

        // get the disc points
        std::vector<vigra::TinyVector<int,3> > discPoints;
        std::vector< int > offsets;
        getDisc(radius, image.stride(), discPoints, offsets);

        std::cout<<"nBallPoints "<<discPoints.size()<<"\n";

        auto getQuantile = [&] (const CoordType c,RankComp & rankComp){
            CoordType oc(vigra::SkipInitialization);
            const T * p  = & image[c];
            for(size_t dpi=0; dpi<discPoints.size();++dpi){
                oc = c + discPoints[dpi];
                if(oc[0] >= 0       && oc[1] >= 0       && oc[2] >= 0 &&
                   oc[0] < shape[0] && oc[1] < shape[1] && oc[2] < shape[2]){
                    rankComp.insert( *(p + offsets[dpi]) );
                }
            }
            const float ranks[1] = {rankOrder};
            float outVal;
            rankComp.getRanks(ranks, ranks+1, &outVal);
            rankComp.reset();
            return outVal;
        };

        auto roiCaller = [&](CoordType b, CoordType e){
            #pragma omp parallel for  //private(buffer)
            for(int z=b[2]; z<e[2]; ++z){
                RankComp rankComp(discPoints.size(), minVal, maxVal, nBins);
                CoordType c(vigra::SkipInitialization);
                c[2]=z;
                for(c[1]=b[1]; c[1]<e[1]; ++c[1])
                for(c[0]=b[0]; c[0]<e[0]; ++c[0]){
                    out[c] = getQuantile(c,rankComp);
                }
            }
        };
        
        // left 
        roiCaller(CoordType(0,0,0), CoordType(radius,shape[1],shape[2]));
        // right
        roiCaller(CoordType(shape[0]-radius,0,0),shape);
        // up 
        roiCaller(CoordType(0,0,0),CoordType(shape[0],radius,shape[2]));
        // down 
        roiCaller(CoordType(0,shape[1]-radius,0),shape);
        // front 
        roiCaller(CoordType(0,0,0),CoordType(shape[0],shape[1],radius));
        // back 
        roiCaller(CoordType(0,0,shape[2]-radius),shape);



        auto getQuantileNoBorder = [&] (const CoordType c,RankComp & rankComp){
            const T * p = &image[c];
            for(size_t dpi=0; dpi<discPoints.size();++dpi){
                rankComp.insert( *(p + offsets[dpi]) );
            }
            const float ranks[1] = {rankOrder};
            float outVal;
            rankComp.getRanks(ranks, ranks+1, &outVal);
            rankComp.reset();
            return outVal;
        };

 
        #pragma omp parallel for  //private(buffer)
        for(int z=radius; z<shape[2]-radius; ++z){
            CoordType c(vigra::SkipInitialization);
            RankComp rankComp(discPoints.size(), minVal, maxVal, nBins);
            c[2]=z;
            for(c[1]=radius; c[1]<shape[1]-radius; ++c[1])
            for(c[0]=radius; c[0]<shape[0]-radius; ++c[0]){
                out[c] = getQuantileNoBorder(c,rankComp);
            }
        }
    }






    template<class T>
    void ballRankOrderFilter(
        const vigra::MultiArrayView<3,T> & image,
        const int radius,
        const float rankOrder,
        vigra::MultiArrayView<3,T>  & out
    ){
        typedef SortingRank<T> RankComp;
        typedef typename vigra::MultiArrayView<3,T>::difference_type CoordType;
        const int size = 2*radius + 1;

        CoordType shape = image.shape();
        CoordType stride = image.stride();
        CoordType rad(radius);
        CoordType cc(vigra::SkipInitialization);
        CoordType oc(vigra::SkipInitialization);


        std::vector<vigra::TinyVector<int,3> > discPoints;
        std::vector< int > offsets;
        std::vector<unsigned char> mask(std::pow(size,3),0);
        {
            size_t i=0;
            for(cc[2]=-radius; cc[2]<radius+1; ++cc[2])
            for(cc[1]=-radius; cc[1]<radius+1; ++cc[1])
            for(cc[0]=-radius; cc[0]<radius+1; ++cc[0], ++i){
                const float r = vigra::norm(cc);
                if(r<=radius){
                    mask[i]=1;
                    discPoints.push_back(cc);
                    offsets.push_back(stride[0]*cc[0]+stride[1]*cc[1]+stride[2]*cc[2]);
                }
            }
        }   

        std::cout<<"nBallPoints "<<discPoints.size()<<"\n";
        typedef std::vector<T> Vec;

        auto getQuantile = [&] (const CoordType c,RankComp & rankComp){
            CoordType s = c - rad;
            CoordType e = c + rad+1;
            CoordType oc(vigra::SkipInitialization);
            size_t i=0; 
            size_t ii=0;

            const T * p  = & image[c];
            for(size_t dpi=0; dpi<discPoints.size();++dpi){
                oc = c + discPoints[dpi];
                if(oc[0] >= 0       && oc[1] >= 0       && oc[2] >= 0 &&
                   oc[0] < shape[0] && oc[1] < shape[1] && oc[2] < shape[2]){
                    rankComp.insert( *(p + offsets[dpi]) );
                }
            }

            const float ranks[1] = {rankOrder};
            float outVal;
            rankComp.getRanks(ranks, ranks+1, &outVal);
            rankComp.reset();
            return outVal;
        };

        auto roiCaller = [&](CoordType b, CoordType e){
            #pragma omp parallel for  //private(buffer)
            for(int z=b[2]; z<e[2]; ++z){
                std::vector<T> buffer(std::pow(size,3));
                RankComp rankComp(discPoints.size(),0,255.0);

                CoordType c(vigra::SkipInitialization);
                c[2]=z;
                for(c[1]=b[1]; c[1]<e[1]; ++c[1])
                for(c[0]=b[0]; c[0]<e[0]; ++c[0]){
                     out[c] = getQuantile(c,rankComp);
                }
            }
        };
        
        // left 
        roiCaller(CoordType(0,0,0), CoordType(radius,shape[1],shape[2]));
        // right
        roiCaller(CoordType(shape[0]-radius,0,0),shape);
        // up 
        roiCaller(CoordType(0,0,0),CoordType(shape[0],radius,shape[2]));
        // down 
        roiCaller(CoordType(0,shape[1]-radius,0),shape);
        // front 
        roiCaller(CoordType(0,0,0),CoordType(shape[0],shape[1],radius));
        // back 
        roiCaller(CoordType(0,0,shape[2]-radius),shape);



        auto getQuantileNoBorder = [&] (const CoordType c,Vec & buffer){
                CoordType s = c - rad;
                CoordType e = c + rad+1;
                CoordType oc(vigra::SkipInitialization);
                size_t i=0; 
                size_t ii=0;

                const T * p = &image[c];
                for(size_t dpi=0; dpi<discPoints.size();++dpi){
                    //oc = c + discPoints[dpi];
                    {
                        //buffer[ii] = image[oc];
                        buffer[ii] = *(p + offsets[dpi]);
                        ++ii;
                    }
                }
                //ls.sort(buffer);
                std::sort(buffer.begin(), buffer.begin() + ii);
                //semi_static_insertion_sort<T>(buffer,ii);

                const float size = ii;
                const float rankPos = size*rankOrder;
                const int rLow = std::floor(rankPos); 
                const int rUp = rLow+1;
                const float m = rankPos - rLow;
                return buffer[rLow]*(1.0-m) + buffer[rUp]*(m);
        };

 
        #pragma omp parallel for  //private(buffer)
        for(int z=radius; z<shape[2]-radius; ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=radius; c[1]<shape[1]-radius; ++c[1])
            for(c[0]=radius; c[0]<shape[0]-radius; ++c[0]){
                out[c] = getQuantileNoBorder(c,buffer);
            }
        }
    }

    

}


#endif /* SKNEURO_BALL_RANK_ORDER_FILTER_HXX */
