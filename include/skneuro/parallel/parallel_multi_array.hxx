#ifndef SKNEURO_PARALLEL_MULTI_ARRAY_HXX
#define SKNEURO_PARALLEL_MULTI_ARRAY_HXX

namespace skneuro{
    namespace parallel{

        template<unsigned int DIM,class T>
        void arrayMinMax(
            const vigra::MultiArrayView<DIM, T> & array,
            T & minVal,
            T & maxVal
        ){
            T min_val=array[0];
            T max_val=array[0];
            // if dense
            if(&array[array.size()]-&array[0] == array.size()){     
                const T * data =  array.data();
                #pragma omp parallel for reduction(min : min_val) reduction(max: max_val)
                for(size_t i=0; i<array.size(); ++i){
                    const T val = data[i];
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }   
            }
            else{
                if(DIM==1){
                    #pragma omp parallel for reduction(min : min_val) reduction(max: max_val)
                    for(size_t i=0; i<array.size(); ++i){
                        min_val = std::min(min_val, array[i]);
                        max_val = std::max(max_val, array[i]);
                    }
                }
                else{
                    typedef typename vigra::MultiArrayView<DIM, T>::const_iterator Iter;
                    #pragma omp parallel for reduction(min : min_val) reduction(max: max_val)
                    for(Iter iter = array.begin(); iter< array.end() ; ++iter ){
                        const T value = *iter;
                        min_val = std::min(min_val, value);
                        max_val = std::max(max_val, value);
                    }
                }
            }
            minVal = min_val;
            maxVal = max_val;
        }
    }
}

#endif // SKNEURO_PARALLEL_MULTI_ARRAY_HXX
