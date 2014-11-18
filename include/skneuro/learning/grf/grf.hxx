#ifndef SKNEURO_LEARNING_GRF_GRF_HXX
#define SKNEURO_LEARNING_GRF_GRF_HXX

#include "tree.hxx"

namespace skneuro{

class PatchLabel{
    public:
};


template<unsigned int DIM>
class PatchLabel{

};

template<unsigned int DIM>
class PatchFeature{

};


template< 
        class FEATUTRE_CONTAINER, 
        class LABEL_CONTAINER
>
class GeneralizedRandomForest{
public:

    typedef TreeNode<SplitNodeData> SplitNode;

    struct Parameter{
        Parameter(){

        }
    };

    GeneralizedRandomForest()
    :   tree_(NULL, SplitNodeData())
    {

    }


    void train(const)

private:

    SplitNode tree_;
};


}


#endif /*SKNEURO_LEARNING_GRF_GRF_HXX */
