
struct LdtParam{
    LdtParam(){
        maxDistance_ = 10;
    }

    vigra::UInt8 maxDistance_;

    float lambdaUnary_;
};


class LdtOpt{
public:
    typedef vigra::UInt64 ViType;
    typedef vigra::UInt8 LabelType;

    
    float unary(const ViType vi, const LabelType l)const{
        return unaryVal(noisyDt_[vi], l);
    }

    float unaryVal(const float uval, const LabelType l)const{
        const float ll = static_cast<float>(l)
        const ld float = std::pow(uval-ll, 2);
        return param_.lambdaUnary_;
    }

private:
    LDtParam param_;
    vigra::MultiArray<float, 3>         noisyDt_;
    vigra::MultiArray<vigra::UInt8, 3>  dt_;
};
