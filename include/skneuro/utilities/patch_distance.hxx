



template<class DIM, class T, class F> 
class GaussianPatchDistance{


    template<class COORD>
    F patchDistance(
        const COORD & coordA,
        const COORD & coordB,
        const COORD & patchRadius
    ){
        
    }


    std::vector<F> gaussianWeight_;
    vigra::MultiArrayView<DIM,T> image_;
};