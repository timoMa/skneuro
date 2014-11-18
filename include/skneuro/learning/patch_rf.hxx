#ifndef SKEURO_LEARNING_PATCH_RF_HXX
#define SKEURO_LEARNING_PATCH_RF_HXX

namespace skneuro{
    

    template<class L>
    void findValids(
        const vigra::MultiArrayView<3, L> & labels,
        std::vector< vigra::TinyVector< vigra::UInt16, 3> >  & coords,
        const int r
    ){
        vigra::MultiArray<3, bool>  isValid(labels.shape());
        isValid = true;
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(labels(x,y,z) == 0 ){
                for(int xx=-1*r; xx <= r; ++xx){
                    int xxx = x+xx;
                    if(xxx>=0 && xxx<labels.shape(0)){
                        isValid(xxx, y, z) = false;
                    }
                }
            }
        }
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == false ){
                for(int yy=-1*r; yy <= r; ++yy){
                    int yyy = y+yy;
                    if(yyy>=0 && yyy<labels.shape(1)){
                        isValid(x, yyy, z) = false;
                    }
                }
            }
        }
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == false ){
                for(int zz=-1*r; zz <= r; ++zz){
                    int zzz = z+zz;
                    if(zzz>=0 && zzz<labels.shape(2)){
                        isValid(x, y, zzz) = false;
                    }
                }
            }
        }

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


    struct PatchRfParam{
        PatchRfParam(){
            patchRadius = 5;
        }
        size_t patchRadius;
    };

    template<class T, class L>
    class PatchRf{
    public:
        typedef vigra::MultiArrayView<4, T>  FeatureVolume;
        typedef vigra::MultiArrayView<3, L>  LabelVolume;
        typedef PatchRfParam Param;


        PatchRf(const Param & param)
        :   param_(param){
        }


        void train(
            const FeatureVolume & features,
            const LabelVolume & labels
        ){

            std::cout<<"start training\n";
            std::cout<<"find legit training instances\n";
            std::vector< vigra::TinyVector< vigra::UInt16, 3> > coords;
            findValids(labels, coords, param_.patchRadius);
            std::cout<<"training examples "<<coords.size()<<"\n";
      
        }





        Param param_;
    };

}


#endif /*SKEURO_LEARNING_PATCH_RF_HXX*/