#ifndef SKEURO_LEARNING_PATCH_RF_UTIL_HXX
#define SKEURO_LEARNING_PATCH_RF_UTIL_HXX


// std
#include <queue>
#include <vector>
#include <algorithm>

// vigra
#include <vigra/random.hxx>
#include <lemon/list_graph.h>
#include <vigra/algorithm.hxx>



// skeuro
#include <skneuro/clustering/mini_batch_kmeans.hxx>
#include "split_finder.hxx"


template<unsigned int DIM>
class PatchPointSampler{
public:

    typedef vigra::TinyVector<int, DIM> Point;
    typedef vigra::TinyVector<int, DIM+1> PointPlusOne;
    typedef vigra::TinyVector<int, 2*DIM> DoublePoint;
    typedef std::pair<Point, Point> PointPair;
    typedef std::vector<PointPair> PointPairVec;


    PatchPointSampler(){

    }   

    
    Point
    randPatchPoint(const double sigma, 
                   const int maxRadius
    ){
        Point res;
        for(size_t d=0; d<DIM; ++d){
            const double randCoord = randgen_.normal(0.0, sigma);
            res[d] = std::round(randCoord);
            res[d] = std::max(res[d], -1*maxRadius);
            res[d] = std::min(res[d],    maxRadius);
        }
        return res;
    }


    PointPlusOne
    randPatchPointAndFeature(const double sigma, 
                             const int maxRadius,
                             const int nChannels
    ){
        PointPlusOne res;
        for(size_t d=0; d<DIM; ++d){
            const double randCoord = randgen_.normal(0.0, sigma);
            res[d] = std::round(randCoord);
            res[d] = std::max(res[d], -1*maxRadius);
            res[d] = std::min(res[d],    maxRadius);
        }
        res[DIM-1] = randgen_.uniformInt(nChannels);
        return res;
    }
    
    PointPair
    randPatchPointPair(const double sigma1, 
                       const double sigma2,
                       const int maxRadius
    ){

        PointPair res;
        res.first  = randPatchPoint(sigma1, maxRadius);
        res.second = randPatchPoint(sigma2, maxRadius);
        while(res.first == res.second){
            res.second = randPatchPoint(sigma2, maxRadius);
        }
        return res;
    }



    
    PointPairVec
    randPatchPointPairVec(const double sigma1, 
                          const double sigma2,
                          const int maxRadius,
                          const size_t nPairs
    ){


        PointPairVec  res(nPairs);

        
        std::set<DoublePoint> used;

        for(size_t i=0; i<nPairs; ++i){
            PointPair pp = randPatchPointPair(sigma1, sigma2, maxRadius);
            // search for unique
            while(used.find(makeDoublePoint(pp)) != used.end()){
                pp = randPatchPointPair(sigma1, sigma2, maxRadius);
            }
            used.insert(makeDoublePoint(pp));
            res[i] = pp;
        }
        return res;
    }

private:
    
    DoublePoint  makeDoublePoint(
        const PointPair pp
    ){
        vigra::TinyVector<int, 2*DIM>  res;
        for(size_t i=0; i<DIM; ++i){
            res[i] = pp.first[i];
            res[i+DIM] = pp.second[i];
        }
        return res;
    }
    vigra::RandomNumberGenerator<> randgen_;
};






namespace skneuro{
    

    template<class L>
    void findValids(
        const vigra::MultiArrayView<3, L> & labels,
        std::vector< vigra::TinyVector< vigra::Int64, 3> >  & coords,
        const int r
    ){
        vigra::MultiArray<3, bool>  isValid(labels.shape(), false);
        vigra::MultiArray<3, bool>  isValid2(labels.shape(), false);

        for(int z=r+1; z<labels.shape(2)-r-1; ++z)
        for(int y=r+1; y<labels.shape(1)-r-1; ++y)
        for(int x=r+1; x<labels.shape(0)-r-1; ++x){
            isValid(x, y, z) = true;
            isValid2(x, y, z) = true;
        }


        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == false ){
                for(int xx=-1*r; xx <= r; ++xx){
                    int xxx = x+xx;
                    if(xxx>=0 && xxx<labels.shape(0)){
                        isValid2(xxx, y, z) = false;
                    }
                }
            }
        }
        isValid = isValid2;
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == false ){
                for(int yy=-1*r; yy <= r; ++yy){
                    int yyy = y+yy;
                    if(yyy>=0 && yyy<labels.shape(1)){
                        isValid2(x, yyy, z) = false;
                    }
                }
            }
        }
        isValid = isValid2;
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == false ){
                for(int zz=-1*r; zz <= r; ++zz){
                    int zzz = z+zz;
                    if(zzz>=0 && zzz<labels.shape(2)){
                        isValid2(x, y, zzz) = false;
                    }
                }
            }
        }
        isValid = isValid2;
        size_t counter = 0;
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == true ){
                ++counter;
            }
        }
        coords.resize(counter);
        counter=0;
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == true ){
                coords[counter][0] = x;
                coords[counter][1] = y;
                coords[counter][2] = z;
                ++counter;
            }
        }
    }
    


    template<class INSTANCE, class RG>
    void getBootstrap(
        const std::vector<INSTANCE> & instInAll,
        std::vector<INSTANCE> & instBootstrap,
        size_t btSize,
        RG & rg
    ){
        const size_t nTotal = instInAll.size();  
        std::vector<bool> isIncluded(nTotal, false);
        for(size_t i=0; i<btSize; ++i){
            isIncluded[rg.uniformInt(nTotal)] = true;
        }
        instBootstrap.resize(0);
        instBootstrap.reserve(btSize);
        for(size_t i=0; i<nTotal; ++i){
            if(isIncluded[i]){
                instBootstrap.push_back(instInAll[i]);
            }
        }

    }

}


#endif /*SKEURO_LEARNING_PATCH_RF_UTIL_HXX*/
