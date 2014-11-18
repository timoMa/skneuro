#include "skneuro/utilities/blocking.hxx"
#include <vigra/multi_convolution.hxx>
#include <vigra/convolution.hxx>
#include <vigra/tinyvector.hxx>
#include "eigen_decomposition3.hxx"
#include <vigra/multi_iterator.hxx>

namespace skneuro{


struct DiffusionParam{

    enum Eigenmode{
        WeickertLine,
        WeickertPlane,
        EED,
        CED
    };


    enum OrientationEstimation{
        Dynamic, // in each iteration new estimate
        
    };




    DiffusionParam(){
        strength_ = 1.0;
        dt_= 0.2;
        maxT_= 10.0;
        sigmaStep_= 0.6;
        C_= 1.0;
        m_= 1.0;
        eps_= 1e-20;
        alpha_= 0.001;
        useSt_=true;
        sigmaTensor1_= 2.0;
        sigmaTensor2_= 2.0;
        sigmaSmooth_ = 1.0;
    }

    double  strength_;
    double  dt_;
    double  maxT_;
    double  sigmaStep_;
    double  C_;
    double  m_;
    double  eps_;
    double  alpha_;
    bool    useSt_;
    double  sigmaTensor1_;
    double  sigmaTensor2_;
    double  sigmaSmooth_;
};


template<unsigned int DIM>
struct TensorHelp;


template<>
struct TensorHelp<3>{
    enum{
        XX = 0,
        YY = 3,
        ZZ = 5,
        XY = 1,
        XZ = 4,
        YZ = 2
    };
};

template<>
struct TensorHelp<2>{
    enum{
        XX = 0,
        YY = 2,
        ZZ = -1,
        XY = 1,
        XZ = -1,
        YZ = -1
    };
};


template<class T>
void solveEigen(
    vigra::MultiArrayView<3, vigra::TinyVector<T, 6> > tensor,
    vigra::MultiArrayView<3, vigra::TinyVector<T, 3> > eigenval,
    vigra::MultiArrayView<3, vigra::TinyVector< vigra::TinyVector<T, 3>, 3> > eigenvec
){
    typedef vigra::TinyVector<int, 3> Shape;
    typedef TensorHelp<3> TH;
    Shape shape = tensor.shape();
    for(size_t z=0; z<shape[2]; ++z)
    for(size_t y=0; y<shape[1]; ++y)
    for(size_t x=0; x<shape[0]; ++x){

        const vigra::TinyVector<T, 6> & t = tensor(x, y, z);
        vigra::TinyVector< vigra::TinyVector<T, 3>, 3> & evec = eigenvec(x, y, z);
        vigra::TinyVector<T, 3> & eval = eigenval(x, y, z);
    

        double A[3][3];
        A[0][0] = static_cast<double>(t[TH::XX]); //xx
        A[1][1] = static_cast<double>(t[TH::YY]); //yy
        A[2][2] = static_cast<double>(t[TH::ZZ]); //zz
        A[0][1] = static_cast<double>(t[TH::XY]); //xy
        A[1][0] = static_cast<double>(t[TH::XY]); //xy
        A[0][2] = static_cast<double>(t[TH::XZ]); //xz
        A[2][0] = static_cast<double>(t[TH::XZ]); //xz
        A[1][2] = static_cast<double>(t[TH::YZ]); //yz
        A[2][1] = static_cast<double>(t[TH::YZ]); //yz

        double EIGENVEC[3][3];
        double EIGENVAL[3];

        // get eigenvec and eigenval
        eigen_decomposition(A, EIGENVEC, EIGENVAL);


        // write back
        for(size_t i=0; i<3; ++i)
        for(size_t j=0; j<3; ++j){
            evec[i][j] = EIGENVEC[j][i];
        }    
        for(size_t i=0; i<3; ++i){
            eval[i] = EIGENVAL[i];
        }

    }
}


template<class T>
void solveEigen(
    vigra::MultiArrayView<2, vigra::TinyVector<T, 3> > tensor,
    vigra::MultiArrayView<2, vigra::TinyVector<T, 2> > eigenval,
    vigra::MultiArrayView<2, vigra::TinyVector<vigra::TinyVector<T, 2>, 2> > eigenvec
){
    typedef vigra::TinyVector<int, 2> Shape;
    typedef TensorHelp<2> TH;

    Shape shape = tensor.shape();

    for(size_t y=0; y<shape[1]; ++y)
    for(size_t x=0; x<shape[0]; ++x){

        const vigra::TinyVector<T, 3> & t = tensor(x, y);
        vigra::TinyVector<vigra::TinyVector<T, 2>, 2> & evec = eigenvec(x, y);
        vigra::TinyVector<T, 2> & eval = eigenval(x, y);
    
        const double a=t[TH::XX];
        const double b=t[TH::XY];
        const double c=t[TH::XY];
        const double d=t[TH::YY];

        // trace
        const double D = a*d-b*c;
        const double Trace = a + d;

        double eA = Trace/2.0  + std::sqrt(Trace*Trace/4.0 -D);
        double eB = Trace/2.0  - std::sqrt(Trace*Trace/4.0 -D);

        if(std::abs(eA) < std::abs(eB))
            std::swap(eA, eB);

        eval[0] = eA;
        eval[1] = eB;
        if( std::abs(b)<=0.00000001 && std::abs(c)<=0.00000001){
            evec[0][0] = 1.0;
            evec[0][1] = 0.0;
            evec[1][0] = 0.0;
            evec[1][1] = 1.0;
        }
        else{
            evec[1][0] = eA-d;
            evec[1][1] = c;
            evec[0][0] = eB-d;
            evec[0][1] = c;

            const double s0 = std::sqrt(evec[0][0]*evec[0][0] + evec[0][1]*evec[0][1]);
            const double s1 = std::sqrt(evec[1][0]*evec[1][0] + evec[1][1]*evec[1][1]);

            evec[0][0] /= s0;
            evec[0][1] /= s0;
            evec[1][0] /= s1;
            evec[1][1] /= s1;
        }

    }
}





template<class T>
struct MyStructTensor{

    template<unsigned int DIM, class T_OUT>
    static void op(
        const vigra::MultiArrayView<DIM, T> & in,
        vigra::MultiArrayView<DIM, T_OUT> & out,
        const double sigma1,
        const double sigma2
    ){
        vigra::structureTensorMultiArray(in, out, sigma1, sigma2); 
    }
};

template<class T, int NC>
struct MyStructTensor< vigra::TinyVector<T, NC> >{

    template<unsigned int DIM, class T_OUT>
    static void op(
        const vigra::MultiArrayView<DIM, vigra::TinyVector<T, NC> > & in,
        vigra::MultiArrayView<DIM, T_OUT> & out,
        const double sigma1,
        const double sigma2
    ){
        vigra::MultiArray<DIM, T_OUT> tmpOut(out.shape());
        out = T_OUT(0.0);
        for(size_t c=0; c<NC; ++c){
            vigra::MultiArrayView<DIM, T> inC = in.bindElementChannel(c);
            vigra::structureTensorMultiArray(inC, tmpOut, sigma1, sigma2); 
            out+=tmpOut;
        }
    }
};


template<class T>
struct MyHessianOfGaussian{

    template<unsigned int DIM, class T_OUT>
    static void op(
        const vigra::MultiArrayView<DIM, T> & in,
        vigra::MultiArrayView<DIM, T_OUT> & out,
        const double sigma1
    ){
        vigra::hessianOfGaussianMultiArray(in, out, sigma1); 
    }
};

template<class T, int NC>
struct MyHessianOfGaussian< vigra::TinyVector<T, NC> >{

    template<unsigned int DIM, class T_OUT>
    static void op(
        const vigra::MultiArrayView<DIM, vigra::TinyVector<T, NC> > & in,
        vigra::MultiArrayView<DIM, T_OUT> & out,
        const double sigma1
    ){
        vigra::MultiArray<DIM, T_OUT> tmpOut(out.shape());
        out = T_OUT(0.0);
        for(size_t c=0; c<NC; ++c){
            vigra::MultiArrayView<DIM, T> inC = in.bindElementChannel(c);
            vigra::hessianOfGaussianMultiArray(inC, tmpOut, sigma1); 
            out+=tmpOut;
        }
    }
};





template<class T>
struct MyGrad{

    template<unsigned int DIM, class T_OUT>
    static void op(
        const vigra::MultiArrayView<DIM, T> & in,
        vigra::MultiArrayView<DIM, T_OUT> out,
        const double sigma1
    ){
        vigra::gaussianGradientMultiArray(in, out, sigma1); 
    }
};

template<class T, int NC>
struct MyGrad< vigra::TinyVector<T, NC> >{

    template<unsigned int DIM, class T_OUT>
    static void op(
        vigra::MultiArrayView<DIM, vigra::TinyVector<T, NC> > & in,
        vigra::MultiArrayView<DIM, T_OUT> & out,
        const double sigma1
    ){
        vigra::MultiArray<DIM, T_OUT> tmpOut(out.shape());
        out = T_OUT(0.0);
        for(size_t c=0; c<NC; ++c){
            vigra::MultiArrayView<DIM, T> inC = in.bindElementChannel(c);
            vigra::gaussianGradientMultiArray(inC, tmpOut, sigma1); 
            out+=tmpOut;
        }
    }
};



template<unsigned int DIM, class T>
struct BlockUpdate{

    typedef vigra::NumericTraits<T> ScalarValueType;



    typedef TensorHelp<DIM> TH;
    typedef vigra::MultiCoordinateIterator<DIM> CoordIter;
    typedef vigra::TinyVector<double, (DIM*DIM+DIM)/2> TensorType;
    typedef vigra::TinyVector<int, DIM> Shape;
    typedef Blocking<int , DIM> BlockingType;
    typedef typename BlockingType::BlockWithBorderType BlockWithBorder;

    typedef vigra::TinyVector<double, DIM> GradVec;

    BlockUpdate(
        const vigra::MultiArrayView<DIM,T> & img,
        const BlockWithBorder &  block,
        const DiffusionParam & param
    )
    :   img_(img),
        block_(block),
        param_(param){
        
    }

    void operator()(){

        // sub image
        vigra::MultiArrayView<DIM, T> subImg = img_.subarray(block_.blockWithBorder().begin(),
                                                               block_.blockWithBorder().end());

        
        //vigra::MultiArray<DIM, T> subImgS(subImg.shape());

        vigra::MultiArray<DIM, TensorType> structureTensor(subImg.shape());

        //vigra::gaussianSmoothMultiArray(subImg, subImgS, param_.sigmaSmooth_); 

        //std::cout<<"tensor\n";
        if(param_.useSt_){
            MyStructTensor<T>::op(subImg, structureTensor, param_.sigmaTensor1_, 
                                             param_.sigmaTensor2_);
        }
        else{
            MyHessianOfGaussian<T>::op(subImg, structureTensor, param_.sigmaTensor1_);
        }
        
        

        //get the diffusion tensor
        vigra::MultiArray<DIM, TensorType> diffTensor(subImg.shape());

        diffTensor = TensorType(1.0);

        //std::cout<<"strucuteTensorToDiffusionTensor\n";
        if(true)
            strucuteTensorToDiffusionTensor(structureTensor, diffTensor);

        //std::cout<<"do step\n";
        // do step
        doStep(subImg, diffTensor);

    }

    void doStep( vigra::MultiArrayView<DIM, T> & subImg,
                 vigra::MultiArrayView<DIM, TensorType> & diffTensor
    ){

        Shape shape = subImg.shape();


        vigra::MultiArray<DIM, GradVec > grad(shape);
        vigra::MultiArray<DIM, GradVec > flux(shape);
        vigra::MultiArray<DIM, double > divergence(shape);

        flux = GradVec(0.0);

        //std::cout<<"gradient\n";
        //vigra::gaussianGradientMultiArray(subImg , grad, param_.sigmaStep_);
        MyGrad<T>::op(subImg , grad, param_.sigmaStep_);


        //std::cout<<"gradient done\n";
        {
            CoordIter coord(shape), endCoord = coord.getEndIterator();
            for(; coord != endCoord; ++coord){
                const GradVec & gvec = grad[*coord];
                const TensorType & dten = diffTensor[*coord];
                GradVec & j = flux[*coord];

                if(DIM==2){
                    j[0] += dten[TH::XX]*gvec[0];
                    j[1] += dten[TH::XY]*gvec[0];

                    j[0] += dten[TH::XY]*gvec[1];
                    j[1] += dten[TH::YY]*gvec[1];
                }
                else if(DIM==3){
                    j[0] += dten[TH::XX]*gvec[0];
                    j[1] += dten[TH::XY]*gvec[0];
                    j[2] += dten[TH::XZ]*gvec[0];

                    j[0] += dten[TH::XY]*gvec[1];
                    j[1] += dten[TH::YY]*gvec[1];
                    j[2] += dten[TH::XZ]*gvec[1];

                    j[0] += dten[TH::YZ]*gvec[2];
                    j[1] += dten[TH::XZ]*gvec[2];
                    j[2] += dten[TH::ZZ]*gvec[2];
                }
            }
        }
        //std::cout<<"divergence\n";
        vigra::gaussianDivergenceMultiArray(flux, divergence, param_.sigmaStep_*0.5);


        //std::cout<<"get views\n";
        vigra::MultiArrayView<DIM, T> subSubImg = subImg.subarray(
                                                         block_.blockLocalCoordinates().begin(), 
                                                         block_.blockLocalCoordinates().end());

        vigra::MultiArray<DIM, double> subSubDiv = divergence.subarray(
                                                         block_.blockLocalCoordinates().begin(), 
                                                       block_.blockLocalCoordinates().end());

        //std::cout<<"update\n";
        subSubDiv*= param_.dt_;
        {
            CoordIter coord(subSubImg.shape()), endCoord = coord.getEndIterator();
            for(; coord != endCoord; ++coord){
                subSubImg[*coord] += T(subSubDiv[*coord]);
            }
        }
        //subSubImg += subSubDiv;

    }

    void strucuteTensorToDiffusionTensor( const vigra::MultiArrayView<DIM, TensorType> & structureTensor,
                                          vigra::MultiArrayView<DIM, TensorType> & diffusionTensor
    ){


        Shape shape = structureTensor.shape();

        // compute eigenvalues / eigenvectors
        vigra::MultiArray<DIM, vigra::TinyVector<double, DIM> > eigenval(structureTensor.shape());
        vigra::MultiArray<DIM, vigra::TinyVector<vigra::TinyVector<double, DIM>, DIM> > eigenvec(structureTensor.shape());

        //std::cout<<"solve eigen\n";
        solveEigen(structureTensor, eigenval, eigenvec);

        CoordIter coord(shape), endCoord = coord.getEndIterator();
        for(; coord != endCoord; ++coord){

            const vigra::TinyVector<vigra::TinyVector<double, DIM>, DIM>  & evec = eigenvec[*coord];
            const vigra::TinyVector<double, DIM> & eval = eigenval[*coord];

            //
            TensorType & dten = diffusionTensor[*coord];
            vigra::TinyVector<double, DIM> ll(0.0);            

            /* Weickert plane shaped */
            const double eps = param_.eps_;
            const double alpha = param_.alpha_;
            const double m = param_.m_;
            const double C = param_.C_;
     
            for(size_t dd=0; dd<DIM-1; ++dd){
                const double ewdiff = std::abs(eval[dd]) - std::abs(eval[DIM-1]);
                //std::cout<<dd<<" "<<ewdiff<<"\n";
                if(ewdiff<eps && ewdiff> -eps){
                    ll[dd] = alpha;
                    //std::cout<<"here\n";
                }
                else{
                    const double tmp = std::exp(-C/std::pow(ewdiff,2.0*m));
                    ll[dd] = alpha + (1.0- alpha)*tmp;
                    //std::cout<<"here!!!!!!!!!!!!!!!!\n";
                }
                //std::cout<<"ll "<<ll[dd]<<"\n";
            }
            ll[DIM-1] = alpha;
                         
            ll*=param_.strength_;
            if(DIM==2){
                dten[TH::XX] = ll[0]*evec[0][0]*evec[0][0] + ll[1]*evec[1][0]*evec[1][0] ; // xx
                dten[TH::YY] = ll[0]*evec[0][1]*evec[0][1] + ll[1]*evec[1][1]*evec[1][1] ; // yy
                dten[TH::XY] = ll[0]*evec[0][0]*evec[0][1] + ll[1]*evec[1][0]*evec[1][1] ; // xy
            }
            else if(DIM==3){
                dten[TH::XX] = ll[0]*evec[0][0]*evec[0][0] + ll[1]*evec[1][0]*evec[1][0] + ll[2]*evec[2][0]*evec[2][0]; // xx
                dten[TH::YY] = ll[0]*evec[0][1]*evec[0][1] + ll[1]*evec[1][1]*evec[1][1] + ll[2]*evec[2][1]*evec[2][1]; // yy
                dten[TH::ZZ] = ll[0]*evec[0][2]*evec[0][2] + ll[1]*evec[1][2]*evec[1][2] + ll[2]*evec[2][2]*evec[2][2]; // zz
                dten[TH::XY] = ll[0]*evec[0][0]*evec[0][1] + ll[1]*evec[1][0]*evec[1][1] + ll[2]*evec[2][0]*evec[2][1]; // xy
                dten[TH::XZ] = ll[0]*evec[0][0]*evec[0][2] + ll[1]*evec[1][0]*evec[1][2] + ll[2]*evec[2][0]*evec[2][2]; // xz
                dten[TH::YZ] = ll[0]*evec[0][1]*evec[0][2] + ll[1]*evec[1][1]*evec[1][2] + ll[2]*evec[2][1]*evec[2][2]; // yz  
            }

            //dten = TensorType(1.0);                                                                   
        }

    }



    vigra::MultiArrayView<DIM,T> img_;
    BlockWithBorder block_;
    DiffusionParam param_;
};  

template<unsigned int DIM, class T>
void blockwiseDiffusion( vigra::MultiArrayView<DIM,T> & img, const DiffusionParam & param){

    typedef vigra::TinyVector<int, DIM> Shape;
    typedef Blocking<int , DIM> BlockingType;

    Shape shape = img.shape();
    Shape blockShape(DIM==3 ? 100 : 256);
    for(size_t d=0; d<DIM; ++d){
        blockShape[d] = std::min(blockShape[d], shape[d]);
    }

    BlockingType blocking(img.shape(), blockShape);

    float t = 0.0f;
    const int margin = 30;
    while(t < param.maxT_){

        std::cout<<"t "<<t<<"\n";
        #pragma omp parallel for
        for(size_t bi=0; bi<blocking.size(); ++bi){
            BlockUpdate<DIM, T> bu(img, blocking.blockWithBorder(bi,margin), param);
            bu();
        }
        t += param.dt_;
    }
}


}
