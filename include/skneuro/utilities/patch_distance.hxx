



template<class T, class F> 
class PatchDistance{

    PatchDistance(
        vigra::MultiArrayView<3,T> image,
        
    )
    :   image_(image){
    }


    template<class COORD,class RADIUS>
    F operator()(
        const COORD & coordA,
        const COORD & coordB,
        const RADIUS & patchRadius
    ){
        vigra::TinyVector<int,3>  offset,coordAA,coordBB;

        F res=0;

        for(offset[2]=-patchRadius[2]; offset<=patchRadius[2]; ++offset[2])
        for(offset[1]=-patchRadius[1]; offset<=patchRadius[1]; ++offset[1])
        for(offset[0]=-patchRadius[0]; offset<=patchRadius[0]; ++offset[0]){
            coordAA=coordA+offset;
            coordBB=coordB+offset;
            res+=std::pow(image_(coordAA)-image_(coordBB),2);
        }
        return res;
    }

    vigra::MultiArrayView<DIM,T> image_;
};