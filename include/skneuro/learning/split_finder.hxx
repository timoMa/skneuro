#ifndef SKEURO_LEARNING_SPLIT_FINDER_HXX
#define SKEURO_LEARNING_SPLIT_FINDER_HXX

#include <stdio.h>
#include <stdlib.h>

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


    template<class F, class L>
    class GiniImpurity{
    public:
        GiniImpurity(
            const size_t nClasses,
            const vigra::MultiArray<1, L> & labels
        )
        : nClasses_(nClasses),
          labels_(labels),
          indices_(labels.shape(0)),
          nInst_(labels.shape(0)),
          nA_(0),
          nB_(0),
          fA_(vigra::TinyVector<int,1>(nClasses)),
          fB_(vigra::TinyVector<int,1>(nClasses))
        {
            //std::cout<<"labels shape : \n";
            //std::cout<<labels.shape(0)<<" nDim\n";

            //for(size_t i=0; i< labels_.size(); ++i){
            //    std::cout<<"label "<<labels_[i]<<"\n";
            //}
        }


        std::pair<F,double> bestSplit(const std::vector<F> & features){
            SKNEURO_CHECK_OP(features.size(), ==, nInst_, "");

            // get the sorted indices according to
            // this feature
            reset();
            vigra::indexSort(features.begin(),features.end(), indices_.begin());

            //std::cout<<"add to b\n";
            // put all istances to set B
            for(size_t i=0; i<nInst_; ++i){
                addToB(labels_(i));
            }


            std::pair<F,double> retVal;
            retVal.first = F();
            retVal.second = std::numeric_limits<double>::infinity();

            size_t bestI=1000000;
            //std::cout<<"start actual eval\n";
            for(size_t i=0; i<nInst_-1; ++i){
                SKNEURO_CHECK_OP(nA_, == ,i, "");
                SKNEURO_CHECK_OP(nB_, == ,nInst_-i, "");

                // move indices_[i] from B to A
                // from set B
                const size_t changingIndex = indices_[i]; 
                SKNEURO_CHECK_OP(changingIndex, <= ,nInst_ ,"");

                fromBtoA(labels_(changingIndex));

                // eval
                if(i+1< nInst_-1 && std::abs(features[changingIndex] - features[indices_[i+1]])<0.000001){
                    continue;
                }

                const double tVar = eval();

                //std::cout<<"eval["<<i<<"] = "<<tVar<<"moving  l "<<labels_(changingIndex)<<"\n";

                if(tVar<retVal.second){
                    retVal.first = features[changingIndex];
                    retVal.second = tVar;
                    bestI = i;
                }

                SKNEURO_CHECK_OP(nA_, == ,i+1, "");
                SKNEURO_CHECK_OP(nB_, == ,nInst_-i-1, "");

                //if(i>10000){
                //    exit(1)
                //}
            }
            //std::cout<<"best i "<<bestI<<"\n";
            return retVal;
        }
    private:   
        double eval()const{

            SKNEURO_CHECK_OP(nA_, >=, 1, "");
            SKNEURO_CHECK_OP(nB_, >=, 1, "");

            SKNEURO_CHECK_OP(nA_, <, nInst_, "");
            SKNEURO_CHECK_OP(nB_, <, nInst_, "");
            
            double fSumA = 0.0;
            double fSumB = 0.0;

            for(size_t c=0; c<nClasses_; ++c){
                const double fa = static_cast<double>(fA_[c])/static_cast<double>(nA_);
                const double fb = static_cast<double>(fB_[c])/static_cast<double>(nB_);
                SKNEURO_CHECK_OP(fa,<=,1.0,"");
                SKNEURO_CHECK_OP(fb,<=,1.0,"");
                fSumA += fa*fa;
                fSumB += fb*fb;
            }

            const double nn = nA_+nB_;
            double nna = nA_;
            double nnb = nB_;

            nna /= (nn);
            nnb /= (nn);

            return ((1.0 - fSumA)*nna + (1.0 - fSumB)*nnb)/2.0;

        }


        void fromBtoA(const L label){
            removeFromB(label);
            addToA(label);
        }

        void addToA(const L label){
            add(label, fA_, nA_);
        }
        void addToB(const L label){
            add(label, fB_, nB_);
        }

        void removeFromA(const L label){
            remove(label, fA_, nA_);
        }
        void removeFromB(const L label){
            remove(label, fB_, nB_);
        }

        void add(
            const L label,
            vigra::MultiArray<1,vigra::UInt64> & f,
            size_t & n
        ){
            SKNEURO_CHECK_OP(label, < , nClasses_, "");
            ++n;
            ++f[label];
        }

        void remove(
            const L label,
            vigra::MultiArray<1,vigra::UInt64> & f,
            size_t & n
        ){
            SKNEURO_CHECK_OP(label, < , nClasses_, "");
            --n;
            --f[label];
        }



        void reset(){

            for(size_t i=0; i<indices_.size(); ++i){
                indices_[i] = i;
            }
            nA_ = 0;
            nB_ = 0;
            fA_ = 0;
            fB_ = 0;
        }

        size_t nClasses_;
        const vigra::MultiArray<1, L> & labels_;
        std::vector<size_t> indices_;
        size_t nInst_;
        size_t nA_,nB_;
        vigra::MultiArray<1,vigra::UInt64> fA_,fB_;
    };
}

#endif /* SKEURO_LEARNING_SPLIT_FINDER_HXX */