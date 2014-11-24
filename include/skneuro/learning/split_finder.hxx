#ifndef SKEURO_LEARNING_SPLIT_FINDER_HXX
#define SKEURO_LEARNING_SPLIT_FINDER_HXX


// std
#include <queue>
#include <vector>
#include <algorithm>
#include <vigra/random.hxx>
#include <lemon/list_graph.h>
#include <vigra/algorithm.hxx>

namespace skneuro{

	template<class F, class L>
	class VarianceRediction{
	public:
		VarianceRediction(
			const vigra::MultiArray<2, L> & labels
		)
        : labels_(labels),
          indices_(labels.shape(1)),
          nInst_(labels.shape(1)),
          nDim_(labels.shape(0)),
          nA_(0),
          nB_(0),
          meanA_(vigra::TinyVector<int,1>(labels.shape(0))),
          meanB_(vigra::TinyVector<int,1>(labels.shape(0))),
          m2A_(vigra::TinyVector<int,1>(labels.shape(0))),
          m2B_(vigra::TinyVector<int,1>(labels.shape(0)))
        {
            //std::cout<<"labels shape : \n";
            //std::cout<<labels.shape(0)<<" nDim\n";
            //std::cout<<labels.shape(1)<<" nInst\n";
		}


		std::pair<F,double> bestSplit(const std::vector<F> & features){

            //std::cout<<"in split finder\n";
            //std::cout<<"labels shape : \n";
            //std::cout<<labels_.shape(0)<<" nDim\n";
            //std::cout<<labels_.shape(1)<<" nInst\n";


            SKNEURO_CHECK_OP(features.size(), ==, nInst_, "");

            // get the sorted indices according to
            // this feature
            reset();
            vigra::indexSort(features.begin(),features.end(), indices_.begin());

            //std::cout<<"add to b\n";
            // put all istances to set B
            for(size_t i=0; i<nInst_; ++i){
                addToB(labels_.bindOuter(i));
            }


            std::pair<F,double> retVal;
            retVal.first = F();
            retVal.second = std::numeric_limits<double>::infinity();

            //std::cout<<"start actual eval\n";
            for(size_t i=0; i<nInst_-1; ++i){
                SKNEURO_CHECK_OP(nA_, == ,i, "");
                SKNEURO_CHECK_OP(nB_, == ,nInst_-i, "");

                // move indices_[i] from B to A
                // from set B
                const size_t changingIndex = indices_[i]; 
                SKNEURO_CHECK_OP(changingIndex, <= ,nInst_ ,"");

                fromBtoA(labels_.bindOuter(changingIndex));

                // eval
                if(i+1< nInst_-1 && std::abs(features[changingIndex] - features[indices_[i+1]])<0.000001){
                    continue;
                }

                const double tVar = eval();
                if(tVar<retVal.second){
                    retVal.first = features[changingIndex];
                    retVal.second = tVar;
                }

                SKNEURO_CHECK_OP(nA_, == ,i+1, "");
                SKNEURO_CHECK_OP(nB_, == ,nInst_-i-1, "");
            }
            return retVal;
		}

	private:   
        double eval()const{

            SKNEURO_CHECK_OP(nA_, >=, 1, "");
            SKNEURO_CHECK_OP(nB_, >=, 1, "");

            SKNEURO_CHECK_OP(nA_, <, nInst_, "");
            SKNEURO_CHECK_OP(nB_, <, nInst_, "");

            double vA = 0;
            double vB = 0;

            for(size_t d=0; d<nDim_; ++d){
                vA  += m2A_[d]/(static_cast<double>(nA_-1));
                vB  += m2B_[d]/(static_cast<double>(nB_-1));
            }
            vA = nA_ == 1 ? 0 : vA;
            vB = nB_ == 1 ? 0 : vB;

            return (vA/nA_ + vB/nB_)/static_cast<double>(nDim_);
        }


        void fromBtoA(const vigra::MultiArrayView<1,L> & patchL){
            removeFromB(patchL);
            addToA(patchL);
        }

        void addToA(const vigra::MultiArrayView<1,L> & patchL){
            add(patchL, meanA_, m2A_, nA_);
        }
        void addToB(const vigra::MultiArrayView<1,L> & patchL){
            add(patchL, meanB_, m2B_, nB_);
        }

        void removeFromA(const vigra::MultiArrayView<1,L> & patchL){
            remove(patchL, meanA_, m2A_, nA_);
        }
        void removeFromB(const vigra::MultiArrayView<1,L> & patchL){
            remove(patchL, meanB_, m2B_, nB_);
        }

        void add(
            const vigra::MultiArrayView<1,L> & patchL,
            vigra::MultiArray<1,double> & mean,
            vigra::MultiArray<1,double> & m2,
            size_t & n
        ){
            SKNEURO_CHECK_OP(patchL.size(), ==, nDim_, "");
            ++n;
            SKNEURO_CHECK_OP(n,<=,nInst_,"");
            for(size_t d=0; d<nDim_; ++d){
                const double l = static_cast<double>(patchL[d]);
                const double delta =  l - mean[d];
                mean[d] += delta/static_cast<double>(n);
                m2[d] += delta*(l-mean[d]);
            }
        }

        void remove(
            const vigra::MultiArrayView<1,L> & patchL,
            vigra::MultiArray<1,double> & mean,
            vigra::MultiArray<1,double> & m2,
            size_t & n
        ){
            SKNEURO_CHECK_OP(patchL.size(), ==, nDim_, "");
            SKNEURO_CHECK_OP(n,>=,2,"");
            --n;
            for(size_t d=0; d<nDim_; ++d){
                const double l = static_cast<double>(patchL[d]);
                const double delta =  l - mean[d];
                mean[d] -= delta/static_cast<double>(n);
                m2[d] -= delta*(l-mean[d]);
            }
        }



        void reset(){

            for(size_t i=0; i<indices_.size(); ++i){
                indices_[i] = i;
            }
            nA_ = 0;
            nB_ = 0;
            meanA_ = 0.0;
            meanB_ = 0.0;
            m2A_ = 0.0;
            m2B_ = 0.0;
        }


		const vigra::MultiArray<2, L> & labels_;
		std::vector<size_t> indices_;
        size_t nInst_;
        size_t nDim_;
        // for incremental / decremental mean
        // and variance
        size_t nA_,nB_;
        vigra::MultiArray<1,double> meanA_,meanB_,m2A_,m2B_;



	};
}

#endif /* SKEURO_LEARNING_SPLIT_FINDER_HXX */