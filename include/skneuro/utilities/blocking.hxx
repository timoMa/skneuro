#ifndef SKNEURO_UTILITIES_BLOCKING 
#define SKNEURO_UTILITIES_BLOCKING 

#include <vigra/box.hxx>
#include <iostream>

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

    std::string str()const{
        std::stringstream ss;
        ss<<"(";
        for(int i=0; i<DIMENSION; ++i){
            ss<<this->begin()[i]<<" ";
        }
        ss<<") -- (";
        for(int i=0; i<DIMENSION; ++i){
            ss<<this->end()[i]<<" ";
        }
        return ss.str();
    }

private:
};

template<class COORDINATE, int DIMENSION>
class BlockWithBorder{
public:
    typedef Block<COORDINATE, DIMENSION>  BlockType;
    typedef typename BlockType::CoordType CoordType;

    BlockWithBorder(){  
    }

    BlockWithBorder(const BlockType & core, const BlockType & coreWithBorder)
    :   core_(core),
        coreWithBorder_(coreWithBorder){
    }

    std::string str()const{
        std::stringstream ss;
        ss<<"Core "<<core_<<" CoreWithBorder "<<coreWithBorder_;
        return ss.str();
    }

private:
    BlockType core_;
    BlockType coreWithBorder_;
};


template<class COORDINATE,int DIMENSION>
std::ostream & operator<<(std::ostream & lhs, const Block<COORDINATE,DIMENSION > & block ){
    lhs<<block.str();
    return lhs;
}

template<class COORDINATE,int DIMENSION>
std::ostream & operator<<(std::ostream & lhs, const BlockWithBorder<COORDINATE,DIMENSION > & blockWithBorder ){
    lhs<<blockWithBorder.str();
    return lhs;
}


template<class COORDINATE, int DIMENSION>
class Blocking{

public:
    
    typedef Block<COORDINATE, DIMENSION>  BlockType;
    typedef BlockWithBorder<COORDINATE, DIMENSION>  BlockWithBorderType;
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
        std::cout<<"get block cpp\n";
        return blocking_[i];
    }

    BlockWithBorderType blockWithBorder(const size_t index , const size_t width)const{
        std::cout<<"get block  with border cpp\n";
        BlockType blockWithBorder = blocking_[index];
        blockWithBorder.addBorder(width);
        std::cout<<"intersect\n";
        // intersect
        blockWithBorder &=  totalBlock_;
        std::cout<<"return \n";
        const BlockWithBorderType bwb( blocking_[index],blockWithBorder);
        std::cout<<"return2 \n";
        return bwb;
    }   

private:
    CoordType shape_;
    CoordType blockShape_;
    BlockType totalBlock_;
    std::vector<BlockType> blocking_;
};

#endif /*SKNEURO_UTILITIES_BLOCKING */