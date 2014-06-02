#include <vigra/box.hxx>


template<class COORDINATE, int DIMENSION>
class Block
: public vigra::Box<COORDINATE, DIMENSION>{

public:
    typedef vigra::Box<COORDINATE, DIMENSION> BaseType;
    typedef typename BaseType::Vector CoordType;


    Block()
    :   BaseType(){
        
    }

    Block(const CoordType & a, const CoordType & b)
    :   BaseType(a,b){

    }
private:
};


template<class COORDINATE, int DIMENSION>
class Blocking{

public:
    
    typedef Block<COORDINATE, DIMENSION>  BlockType;
    typedef typename BlockType::CoordType CoordType;

    Blocking()
    :   shape_(),
        blockShape_(),
        totalBlock_(),
        blocking_(){

    }

    Blocking(const CoordType & shape, const CoordType & blockShape)
    :   shape_(shape),
        blockShape_(blockShape),
        totalBlock_(CoordType(0),shape),
        blocking_(){

        SKNEURO_CHECK_OP(DIMENSION,==,3,"currently only implemented for 3D");

        CoordType blockStart(0);
        for(blockStart[2]=0; blockStart[2]<shape[2]; blockStart[2]+=blockShape[2])
        for(blockStart[1]=0; blockStart[1]<shape[1]; blockStart[1]+=blockShape[1])
        for(blockStart[0]=0; blockStart[0]<shape[0]; blockStart[0]+=blockShape[0]){

            CoordType blockEnd = blockStart + blockShape;
            BlockType block(blockStart,blockEnd);
            // intersect
            block &= totalBlock_;
            blocking_.push_back(block);
        }   
    }

    size_t size()const{
        return blocking_.size();
    }

    const BlockType & operator[](const size_t i)const{
        SKNEURO_ASSERT_OP(i,<,blocking_.size());
        return blocking_[i];
    }

    BlockType blockWithBorder(const size_t index , const size_t width)const{
        BlockType block = blocking_[index];
        block.addBorder(width);
        // intersect
        block &=  totalBlock_;
    }   

private:
    CoordType shape_;
    CoordType blockShape_;
    BlockType totalBlock_;
    std::vector<BlockType> blocking_;
};