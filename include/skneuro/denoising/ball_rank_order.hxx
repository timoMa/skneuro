#ifndef SKNEURO_BALL_RANK_ORDER_FILTER_HXX
#define SKNEURO_BALL_RANK_ORDER_FILTER_HXX

/*vigra*/
#include <vigra/multi_array.hxx>
#include <vigra/gaussians.hxx>
#include <vigra/algorithm.hxx>

namespace skneuro{



    template<class T, unsigned short K>
    void static_insertion_sort(std::vector<T> & vec)
    {
        for (unsigned short i=1; i<K; ++i)
        {
            T value = vec[i];
            short  hole  = i;

            for (;  hole > 0 && value < vec[hole-1];  --hole)
                vec[hole] = vec[hole-1];
            
            vec[hole] = value;
        }
    }


    template<class T>
    void semi_static_insertion_sort(std::vector<T> & vec, const size_t size)
    {   

        switch(size){
            case 123: static_insertion_sort<T, 123>(vec);break;
            case 33: static_insertion_sort<T, 33>(vec);break;
            case 7: static_insertion_sort<T, 7>(vec);break;
            case 5*5*5: static_insertion_sort<T, 5*5*5>(vec);break;
            case 9*9*9: static_insertion_sort<T, 9*9*9>(vec);break;
            case 11*11*11: static_insertion_sort<T, 11*11*11>(vec);break;
            case 13*13*13: static_insertion_sort<T, 13*13*13>(vec);break;
            case 15*15*15: static_insertion_sort<T, 15*15*15>(vec);break;
            case 17*17*17: static_insertion_sort<T, 17*17*17>(vec);break;
            case 19*19*19: static_insertion_sort<T, 19*19*19>(vec);break;
            default:
                std::sort(vec.begin(),vec.begin()+size);
        }
    }


    template<class T>
    class LazySort{
    public:
        LazySort(const size_t size)
        : size_(size),
          indices_(size),
          buffer_(size){
            for(size_t i=0; i<size; ++i){
                indices_[i] = i;
            }
        }
        void sort(const std::vector<T> & data){
            // apply current indexing
            for(size_t i=0; i<size_; ++i){
                buffer_[i] = data[indices_[i]];
            }
            // sort 
            this->_sort_indexes();

            // apply current indexing
            for(size_t i=0; i<size_; ++i){
                buffer_[i] = data[indices_[i]];
            }
        }


        void _sort_indexes() {

            //// initialize original index locations
            ////vector<size_t> indices_(v.size());
            //for (size_t i = 0; i != indices_.size(); ++i) {
            //    indices_[i] = i;
            //}
            // sort indexes based on comparing values in v
            std::sort(indices_.begin(), indices_.end(),
                [&](size_t i1, size_t i2) {
                    return buffer_[i1] < buffer_[i2];
                }
            );
            //return indices_;
        }


    private:
        std::vector<T>  buffer_;
        std::vector<size_t> indices_;
        size_t size_;
    };

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

        auto getQuantile = [&] (const CoordType c,Vec & buffer){
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
                    buffer[ii] = *(p + offsets[dpi]);
                    ++ii;
                }
            }

            std::sort(buffer.begin(), buffer.begin() + ii);
            
            //semi_static_insertion_sort<T>(buffer,ii);

            const float size = ii;
            const float rankPos = size*rankOrder;
            const int rLow = std::floor(rankPos); 
            const int rUp = rLow+1;
            const float m = rankPos - rLow;
            return buffer[rLow]*(1.0-m) + buffer[rUp]*(m);
        };

        
        /// left 
        #pragma omp parallel for  //private(buffer)
        for(int z=0; z<shape[2]; ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=0; c[1]<shape[1]; ++c[1])
            for(c[0]=0; c[0]<radius; ++c[0]){
                out[c] = getQuantile(c,buffer);
            }
        }
        /// right
        #pragma omp parallel for  //private(buffer)
        for(int z=0; z<shape[2]; ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=0; c[1]<shape[1]; ++c[1])
            for(c[0]=shape[0]-radius; c[0]<shape[0]; ++c[0]){
                out[c] = getQuantile(c,buffer);
            }
        }

        /// up 
        #pragma omp parallel for  //private(buffer)
        for(int z=0; z<shape[2]; ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=0; c[1]<radius; ++c[1])
            for(c[0]=0; c[0]<shape[0]; ++c[0]){
                out[c] = getQuantile(c,buffer);
            }
        }
        // down
        #pragma omp parallel for  //private(buffer)
        for(int z=0; z<shape[2]; ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=shape[1]-radius; c[1]<shape[1]; ++c[1])
            for(c[0]=0; c[0]<shape[0]; ++c[0]){
                out[c] = getQuantile(c,buffer);
            }
        }
        //front
        #pragma omp parallel for  //private(buffer)
        for(int z=0; z<radius; ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=0; c[1]<shape[1]; ++c[1])
            for(c[0]=0; c[0]<shape[0]; ++c[0]){
                out[c] = getQuantile(c,buffer);
            }
        }
        // back
        #pragma omp parallel for  //private(buffer)
        for(int z=shape[2]-radius; z<shape[2]; ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=0; c[1]<shape[1]; ++c[1])
            for(c[0]=0; c[0]<shape[0]; ++c[0]){
                out[c] = getQuantile(c,buffer);
            }
        }
        /*
        for(size_t z=0;z<seg.shape(2); ++z)
        for(size_t x=0;x<seg.shape(0); ++x){
            atBorder[seg(x,0,z)] = true;
            atBorder[seg(x,seg.shape(1)-1,z)] = true;
        }

        for(size_t y=0;y<seg.shape(1); ++y)
        for(size_t x=0;x<seg.shape(0); ++x){
            atBorder[seg(x,y,0)] = true;
            atBorder[seg(x,y,seg.shape(2)-1)] = true;
        }
        */

        auto getQuantileNoBorder = [&] (const CoordType c,Vec & buffer, LazySort<T> & ls){
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
            LazySort<T> ls(discPoints.size());
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=radius; c[1]<shape[1]-radius; ++c[1])
            for(c[0]=radius; c[0]<shape[0]-radius; ++c[0]){
                out[c] = getQuantileNoBorder(c,buffer, ls);
            }
        }
    }

    
    template<class T>
    void ballRankOrderFilterHistogram(
        const vigra::MultiArrayView<3,T> & image,
        const int radius,
        const float rankOrder,
        vigra::MultiArrayView<3,T>  & out
    ){
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

        auto getQuantile = [&] (const CoordType c,Vec & buffer){
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
                    buffer[ii] = *(p + offsets[dpi]);
                    ++ii;
                }
            }

            std::sort(buffer.begin(), buffer.begin() + ii);
            
            //semi_static_insertion_sort<T>(buffer,ii);

            const float size = ii;
            const float rankPos = size*rankOrder;
            const int rLow = std::floor(rankPos); 
            const int rUp = rLow+1;
            const float m = rankPos - rLow;
            return buffer[rLow]*(1.0-m) + buffer[rUp]*(m);
        };

        
        /// left 
        #pragma omp parallel for  //private(buffer)
        for(int z=0; z<shape[2]; ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=0; c[1]<shape[1]; ++c[1])
            for(c[0]=0; c[0]<radius; ++c[0]){
                out[c] = getQuantile(c,buffer);
            }
        }
        /// right
        #pragma omp parallel for  //private(buffer)
        for(int z=0; z<shape[2]; ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=0; c[1]<shape[1]; ++c[1])
            for(c[0]=shape[0]-radius; c[0]<shape[0]; ++c[0]){
                out[c] = getQuantile(c,buffer);
            }
        }

        /// up 
        #pragma omp parallel for  //private(buffer)
        for(int z=0; z<shape[2]; ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=0; c[1]<radius; ++c[1])
            for(c[0]=0; c[0]<shape[0]; ++c[0]){
                out[c] = getQuantile(c,buffer);
            }
        }
        // down
        #pragma omp parallel for  //private(buffer)
        for(int z=0; z<shape[2]; ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=shape[1]-radius; c[1]<shape[1]; ++c[1])
            for(c[0]=0; c[0]<shape[0]; ++c[0]){
                out[c] = getQuantile(c,buffer);
            }
        }
        //front
        #pragma omp parallel for  //private(buffer)
        for(int z=0; z<radius; ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=0; c[1]<shape[1]; ++c[1])
            for(c[0]=0; c[0]<shape[0]; ++c[0]){
                out[c] = getQuantile(c,buffer);
            }
        }
        // back
        #pragma omp parallel for  //private(buffer)
        for(int z=shape[2]-radius; z<shape[2]; ++z){
            std::vector<T> buffer(std::pow(size,3));
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=0; c[1]<shape[1]; ++c[1])
            for(c[0]=0; c[0]<shape[0]; ++c[0]){
                out[c] = getQuantile(c,buffer);
            }
        }
        /*
        for(size_t z=0;z<seg.shape(2); ++z)
        for(size_t x=0;x<seg.shape(0); ++x){
            atBorder[seg(x,0,z)] = true;
            atBorder[seg(x,seg.shape(1)-1,z)] = true;
        }

        for(size_t y=0;y<seg.shape(1); ++y)
        for(size_t x=0;x<seg.shape(0); ++x){
            atBorder[seg(x,y,0)] = true;
            atBorder[seg(x,y,seg.shape(2)-1)] = true;
        }
        */

        auto getQuantileNoBorder = [&] (const CoordType c,Vec & buffer, LazySort<T> & ls){
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
            LazySort<T> ls(discPoints.size());
            CoordType c(vigra::SkipInitialization);
            c[2]=z;
            for(c[1]=radius; c[1]<shape[1]-radius; ++c[1])
            for(c[0]=radius; c[0]<shape[0]-radius; ++c[0]){
                out[c] = getQuantileNoBorder(c,buffer, ls);
            }
        }
    }
}


#endif /* SKNEURO_BALL_RANK_ORDER_FILTER_HXX */
