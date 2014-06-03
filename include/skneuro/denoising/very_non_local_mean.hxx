

/*


    - PRESELECTION INVARIANT TO NOISE :

        - compute patch distance with single pixel multiband feature:
    

            BEST IDEA: 
                pre-selection which is pixel wise:
                    - TV smoothed mean  centeral pixel distance
                    - an Step Edge orientation (and magnitude ) detector distance
                        which should be weighted with the magnitude 
                        ( if there is no magnitude orientation is bad)
                    - same but with non-step edge
                patch distance :
                    - TV smoothed mean patch distance (!!!)




        - compute patch distance on a bit smoothed  image
          but use not-smoothed image for averaging


*/


template< unsigned int DIM, class T, class PATCH_DISTANCE>
class VeryNonLocalMean{

    typedef vigra::TinyVector<T,DIM> CoordType;
    typedef PATCH_DISTANCE PatchDistance;
    typedef typename PatchDistance::WeightType WeightType;
    typedef long double EstimateType;
    typedef long double EstimateScalarType;
    struct ParameterType{
         
    };

    void run(){



        CoordType coordA;

        for(coordA[2]=0; coordA[2]<shape_[2]; ++coordA[2])
        for(coordA[1]=0; coordA[1]<shape_[1]; ++coordA[1])
        for(coordA[0]=0; coordA[0]<shape_[0]; ++coordA[0]){
            fullSmoothing(coord);
        }

    }


    void fullSmoothing(
        const CoordType & coorA
    ){

        CoordType coordB;
        for(coordB[2]=0; coordB[2]<shape_[2]; ++coordB[2])
        for(coordB[1]=0; coordB[1]<shape_[1]; ++coordB[1])
        for(coordB[0]=0; coordB[0]<shape_[0]; ++coordB[0]){
            if(coordA != corrdB){
                evaluatePair(coordA,coordB);
            }
        }

    }
    
    void evaluatePair(
        const CoordType & coordA,
        const CoordType & coordB
    ){
        std::pair<bool,WeightType> res = patchDistance_(coordA,coordB);
        if(res.first==true){
            writeInWeightedSumImage(coordA,coordB,res.second);
        }
    }


    void writeInWeightedSumImage(
        const CoordType & coordA,
        const CoordType & coordB,
        WeightType weight
    ){

    }

    void normalizeResult(

    ){
        for(coord[2]=0; coord[2]<shape_[2]; ++coord[2])
        for(coord[1]=0; coord[1]<shape_[1]; ++coord[1])
        for(coord[0]=0; coord[0]<shape_[0]; ++coord[0]){
           weightedSumImage_(coord)/=weightedSumImage_(coord); 
        }
    }

    CoordType shape_;
    PatchDistance patchDitance_;

    vigra::MultiArray<DIM,EstimateType> weightedSumImage_;
    vigra::MultiArray<DIM,WeightType>   weightSumImage_;
};