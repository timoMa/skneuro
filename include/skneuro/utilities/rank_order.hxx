#ifndef SKNEURO_UTILITIES_RANK_ORDER_FILTER_HXX
#define SKNEURO_UTILITIES_RANK_ORDER_FILTER_HXX

/*std*/
#include <vector>
#include <algorithm>

/*skneuro*/
#include "skneuro/skneuro.hxx"


namespace skneuro{


    template<class T, unsigned int N_RANKS>
    class SortingRank{
    public:
        struct Options{
            Options(
                const vigra::TinyVector<float, N_RANKS> & ranks,
                const size_t sizeHint = 0
            )
            :   ranks_(ranks),
                sizeHint_(sizeHint){
            }

            vigra::TinyVector<float, N_RANKS>  ranks_;
            size_t sizeHint_;
        };

        SortingRank(const Options & options)
        :   options_(options){
            values_.reserve(options.sizeHint_);
        }


        void reset(){
            values_.resize(0);
        }
        void insert(const T value){
            values_.push_back(value);
        }   

        template<class T_OUT>
        vigra::TinyVector<T_OUT, N_RANKS> getRanks(){

            std::sort(values_.begin(), values_.end());

            vigra::TinyVector<T_OUT, N_RANKS>  out;
            for(size_t ri=0; ri<N_RANKS; ++ri){

                const float rank =  options_.ranks_[ri];
                const float size = values_.size();
                const float rankPos = size*rank;
                const int rLow = std::floor(rankPos); 
                const int rUp = rLow+1;
                const float m = rankPos - rLow;
                out[ri] =  values_[rLow]*(1.0-m) + values_[rUp]*(m);
            }
            return out;
        } 

    private:
        Options options_;

        std::vector<T> values_;
    };

    



    template<class T, unsigned int N_RANKS>
    class HistogramRank{
    public:
        struct Options{
            Options(
                const vigra::TinyVector<float, N_RANKS> & ranks,
                const T minVal,
                const T maxVal,
                const size_t nBins
            )
            :   ranks_(ranks),
                minVal_(minVal),
                maxVal_(maxVal),
                nBins_(nBins){
            }

            vigra::TinyVector<float, N_RANKS>  ranks_;
            T minVal_;
            T maxVal_;
            size_t nBins_;
        };

        HistogramRank(const Options & options)
        :   options_(options),
            binCount_(options.nBins_){
        }


        void reset(){
            std::fill(binCount_.begin(), binCount_.end(), 0.0f);
        }
        void insert(const T value){
            // options
            const T minVal = options_.minVal_;
            const T maxVal = options_.maxVal_;
            const size_t nBins = options_.nBins_;
            // bin indices
            const T fbinIndex = ((value-minVal)/(maxVal-minVal))*nBins;
            const T fFloorBin = std::floor(fbinIndex);
            const int floorBin = static_cast<int>(fFloorBin);
            const int ceilBin = static_cast<int>(std::ceil(fbinIndex));

            if(floorBin==ceilBin){
                binCount_[floorBin] += 1.0; 
            }
            else{
                const double ceilW = (fbinIndex - fFloorBin);
                const double floorW = 1.0 - ceilW;
                binCount_[floorBin] += floorW; 
                binCount_[ceilBin] += ceilW; 
            }
        }   

        template<class T_OUT>
        T_OUT bin2Value(const size_t bi)const{

            const T minVal = options_.minVal_;
            const T maxVal = options_.maxVal_;
            const T dMiMa = maxVal - minVal;
            const size_t nBins = options_.nBins_;

            return  static_cast<T_OUT>(dMiMa)*
                    static_cast<T_OUT>(bi)/nBins + static_cast<T_OUT>(minVal);
        }

        template<class T_OUT>
        vigra::TinyVector<T_OUT, N_RANKS> getRanks(){
            // options
            const T minVal = options_.minVal_;
            const T maxVal = options_.maxVal_;
            const T dMiMa = maxVal - minVal;
            const size_t nBins = options_.nBins_;

            // normalize
            this->cumSum();

            vigra::TinyVector<T_OUT, N_RANKS>  out;


            size_t bi=0;
            for(size_t ri=0; ri<N_RANKS; ++ri){
                SKNEURO_CHECK_OP(bi,<,nBins,"");
                const float rank = options_.ranks_[ri];
                const float atBi = binCount_[bi];
                while(true){
                    if(rank<binCount_[atBi] || std::abs(rank-atBi) < 0.0000001){
                        out[ri] = this-> template bin2Value<T_OUT>(bi);
                        break;
                    }
                    else if(bi == nBins-1){
                        SKNEURO_CHECK_OP(rank,>=, binCount_[bi],"internal bug");
                        out[ri] = this-> template bin2Value<T_OUT>(bi);
                        break;
                    }
                    else{
                        if(atBi<= rank && rank <= binCount_[bi+1]){
                            const auto valLow  = this-> template bin2Value<T_OUT>(bi);
                            const auto valHigh = this-> template bin2Value<T_OUT>(bi+1);


                            auto dLow = rank -binCount_[bi];
                            auto dHigh = rank -binCount_[bi];
                            auto s= dLow + dHigh;
                            dLow  /= s;
                            dHigh /= s;


                            // hack 
                            out[ri]= valLow*(dHigh) + valHigh*(1.0-dHigh);
                            break;
                        }
                        else{
                            ++bi;
                        }
                    }
                }

            }
            return out;
        } 
    private:

        void cumSum(){
            // todo make it 2
            float sum = 0 ;
            for(size_t bi=0; bi<binCount_.size(); ++bi){
                sum += binCount_[bi];
            }
            for(size_t bi=0; bi<binCount_.size(); ++bi){
                binCount_[bi]/=sum;
            }
            for(size_t bi=1; bi<binCount_.size(); ++bi){
                const float prevVal = binCount_[bi-1];
                binCount_[bi] += prevVal;
            }
        }

        Options options_;

        std::vector<float> binCount_;
    };




    template<unsigned int N_RANKS>
    class HistogramRank<vigra::UInt8, N_RANKS>{
    public:
        typedef vigra::UInt8 T;
        struct Options{
            Options(
                const vigra::TinyVector<float, N_RANKS> & ranks,
                const T minVal,
                const T maxVal,
                const size_t nBins
            )
            :   ranks_(ranks){
            }

            vigra::TinyVector<float, N_RANKS>  ranks_;
        };

        HistogramRank(const Options & options)
        :   options_(options),
            binCount_(256){
        }


        void reset(){
            std::fill(binCount_.begin(), binCount_.end(), 0.0f);
        }
        void insert(const T value){
            binCount_[value] += 1.0;
        }   

        template<class T_OUT>
        T_OUT bin2Value(const size_t bi)const{
            return static_cast<T_OUT>(bi);
        }

        template<class T_OUT>
        vigra::TinyVector<T_OUT, N_RANKS> getRanks(){

            // normalize
            this->cumSum();

            vigra::TinyVector<T_OUT, N_RANKS>  out;


            size_t bi=0;
            for(size_t ri=0; ri<N_RANKS; ++ri){
                SKNEURO_CHECK_OP(bi,<,256,"");
                const float rank = options_.ranks_[ri];
                const float atBi = binCount_[bi];
                while(true){
                    if(rank<binCount_[atBi] || std::abs(rank-atBi) < 0.0000001){
                        out[ri] = this-> template bin2Value<T_OUT>(bi);
                        break;
                    }
                    else if(bi == 255){
                        SKNEURO_CHECK_OP(rank,>=, binCount_[bi],"internal bug");
                        out[ri] = this-> template bin2Value<T_OUT>(bi);
                        break;
                    }
                    else{
                        if(atBi<= rank && rank <= binCount_[bi+1]){
                            const auto valLow  = this-> template bin2Value<T_OUT>(bi);
                            const auto valHigh = this-> template bin2Value<T_OUT>(bi+1);
                            auto dLow = rank -binCount_[bi];
                            auto dHigh = rank -binCount_[bi];
                            auto s= dLow + dHigh;
                            dLow  /= s;
                            dHigh /= s;
                            // hack 
                            out[ri]= valLow*(dHigh) + valHigh*(1.0-dHigh);
                            break;
                        }
                        else{
                            ++bi;
                        }
                    }
                }

            }
            return out;
        } 
    private:

        void cumSum(){
            // todo make it 2
            float sum = 0 ;
            for(size_t bi=0; bi<binCount_.size(); ++bi){
                sum += binCount_[bi];
            }
            for(size_t bi=0; bi<binCount_.size(); ++bi){
                binCount_[bi]/=sum;
            }
            for(size_t bi=1; bi<binCount_.size(); ++bi){
                const float prevVal = binCount_[bi-1];
                binCount_[bi] += prevVal;
            }
        }

        Options options_;

        std::vector<float> binCount_;
    };
} // end namespace skneuro


#endif /*SKNEURO_UTILITIES_RANK_ORDER_FILTER_HXX*/
