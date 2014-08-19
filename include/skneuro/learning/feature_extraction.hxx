#ifndef SKNEURO_FEATURE_EXTRACTION_HXX
#define SKNEURO_FEATURE_EXTRACTION_HXX

#include <iostream>

#include <vigra/multi_array.hxx>
#include <vigra/multi_gridgraph.hxx>
#include <vigra/graphs.hxx>
#include <vigra/graph_generalization.hxx>
#include <vigra/multi_gridgraph.hxx>
#include <vigra/adjacency_list_graph.hxx>
#include <vigra/graph_maps.hxx>
#include <vigra/timing.hxx>
#include <vigra/accumulator.hxx>
#include <vigra/multi_convolution.hxx>

#include <omp.h>
namespace skneuro{
    

    struct AccumulatorOptions{

        AccumulatorOptions(){
            nBins=20;
            sigmaHist=1.5;
            edgeFeatures=true;
            nodeFeatures=true;
            select.push_back("Mean");
            select.push_back("Variance");
            select.push_back("UserRangeHistogram");
            histMin.reshape(vigra::MultiArray<1, double>::difference_type(1));
            histMax.reshape(vigra::MultiArray<1, double>::difference_type(2));

            histMax[0]=-1.0;
            histMin[0]=1.0;

        }


        std::vector< std::string> select;

        bool edgeFeatures;
        bool nodeFeatures;    
        

        size_t nBins;
        double sigmaHist;
        vigra::MultiArray<1, double> histMin;
        vigra::MultiArray<1, double> histMax;


        size_t featuresPerChannel()const{
            size_t nFeat = 0;
            for(size_t i=0; i<select.size(); ++i){
                if(select[i] == std::string("UserRangeHistogram"))
                    nFeat+=nBins;
                else
                    ++nFeat;
            }
            return nFeat;   
        }
    };


    template<class TAG>
    struct TagNr;

    #define TAG_NR_GEN(TAG, NR) \
    template<> \
    struct TagNr< TAG >{static const int value = NR;} \



    TAG_NR_GEN(vigra::acc::Mean,                  0);
    TAG_NR_GEN(vigra::acc::Variance,              1);
    TAG_NR_GEN(vigra::acc::Minimum,               2);
    TAG_NR_GEN(vigra::acc::Maximum,               3);
    TAG_NR_GEN(vigra::acc::UserRangeHistogram<0>, 4);

    struct MaxTag{
        static const int value = 4;
    };

    template< class USED_TAG>
    inline void setUsedTag(const std::vector<std::string > & select, USED_TAG & usedTag){

        for(size_t ti=0; ti<MaxTag::value; ++ti){
            usedTag[ti] = 0;
        }

        for(size_t si=0; si<select.size(); ++si){
            const std::string & name = select[si];
            if(name==std::string("Mean")){
                usedTag[TagNr<vigra::acc::Mean>::value] = 1;
            }
            else if(name==std::string("Variance")){
                usedTag[TagNr<vigra::acc::Variance>::value] = 1;
            }
            else if(name==std::string("Minimum")){
                usedTag[TagNr<vigra::acc::Minimum>::value] = 1;
            }
            else if(name==std::string("Maximum")){
                usedTag[TagNr<vigra::acc::Maximum>::value] = 1;
            }
            else if(name==std::string("UserRangeHistogram")){
                usedTag[TagNr<vigra::acc::UserRangeHistogram<0> >::value] = 1;
            }
        }
    }

    template<class ACC_CHAIN, class USED_TAG>
    inline void activateTags(ACC_CHAIN & accChain, USED_TAG & usedTag){
        using namespace vigra::acc;
        if(usedTag[TagNr<Mean>::value]){
            accChain.activate<Mean>();
        }
        if(usedTag[TagNr<Variance>::value]){
            accChain.activate<Variance>();
        }
        if(usedTag[TagNr<Minimum>::value]){
            accChain.activate<Minimum>();
        }
        if(usedTag[TagNr<Maximum>::value]){
            accChain.activate<Maximum>();
        }
        if(usedTag[TagNr<vigra::acc::UserRangeHistogram<0> >::value]){
            accChain.activate<vigra::acc::UserRangeHistogram<0> >();
        }
    }


    template<class ACC_CHAIN,class USED_TAG, class FEATURES>
    inline void extractFeatures(const ACC_CHAIN & accChain, const USED_TAG & usedTag, 
                                const AccumulatorOptions & options,const size_t id, FEATURES & features
    ){
        size_t fIndex=0;
        using namespace vigra::acc;
        if(usedTag[TagNr<Mean>::value]){
            features(id,fIndex++) = get<Mean>(accChain);
        }
        if(usedTag[TagNr<Variance>::value]){
            features(id,fIndex++) = get<Variance>(accChain);
        }
        if(usedTag[TagNr<Minimum>::value]){
            features(id,fIndex++) = get<Minimum>(accChain);
        }
        if(usedTag[TagNr<Maximum>::value]){
            features(id,fIndex++) = get<vigra::acc::Maximum>(accChain);
        }
        if(usedTag[TagNr<UserRangeHistogram<0> >::value]){
            vigra::MultiArray<1, double> hist = get<UserRangeHistogram<0> >(accChain);
            vigra::MultiArray<1, double> sHist(hist.shape());
            gaussianSmoothMultiArray(hist, sHist, options.sigmaHist);
            double sum = sHist.sum<double>();
            sHist/=sum;
            for(size_t bi=0; bi<options.nBins; ++bi){

                SKNEURO_CHECK_OP(bi,<,sHist.shape(0),"");
                SKNEURO_CHECK_OP(fIndex,<,features.shape(1),"");
                sum += sHist[bi];
                features(id,fIndex++) = sHist[bi];
            }
        }
    }


    template<class PIXEL_TYPE, class FEATURE_TYPE>
    void accumulateFeatures(
        const GridGraph3d &                             gridGraph,
        const Rag &                                     rag,
        const GridGraph3dLablsView &                    labels,
        const GridGraph3dAffiliatedEdges &              affiliatedEdges,
        const vigra::MultiArrayView<3, PIXEL_TYPE >  &  volume,
        const AccumulatorOptions & options,
        vigra::MultiArrayView<2, FEATURE_TYPE > &       edgeFeatures,
        vigra::MultiArrayView<2, FEATURE_TYPE > &       nodeFeatures
    ){
        // check input sanity
        SKNEURO_CHECK_OP(rag.edgeNum(), >, 0, "no edges");
        SKNEURO_CHECK_OP(rag.edgeNum(), == , rag.maxEdgeId()+1, "malformed graph");
        SKNEURO_CHECK_OP(options.select.size(),>, 0, "no selected accumulator");


        using namespace vigra::acc;

        typedef UserRangeHistogram<0> Hist;
        typedef Select< Mean, Variance , Minimum, Maximum, Hist> Selection;
        typedef DynamicAccumulatorChain<double,  Selection > AccChain;
      

        // which accumulator is used
        vigra::UInt8 usedTag[MaxTag::value];
        setUsedTag(options.select, usedTag);

        //  set the options 
        vigra::HistogramOptions histogram_opt;
        histogram_opt = histogram_opt.setBinCount(options.nBins).setMinMax(options.histMin[0], options.histMax[0]);


        if(options.edgeFeatures){
            //std::cout<<"DO EDGE FEATRURES\n";
            // allocate a vector of accumulator chains
            std::vector<AccChain> accChainVec(rag.edgeNum());   
            
            //  do the number of required passes
            const size_t nPasses = 1; // TODO FIX ME!!!!!!accChainVec.front().passesRequired();


            //std::cout<<"passes required?!?!"<<nPasses<<"\n";

            for(size_t p=0; p < nPasses; ++p){

                // loop over all rag edges in parallel
                #pragma omp parallel for
                for(size_t eid = 0; eid< rag.edgeNum(); ++eid){

                    // get reference to grid graph edges
                    // and the accumulator chain for this edge
                    const GridGraph3dEdgeVector & edgeVector = affiliatedEdges[rag.edgeFromId(eid)];
                    AccChain & accChain = accChainVec[eid];

                    // activate tags (in first pass)
                    if(p == 0){
                        activateTags(accChain, usedTag);
                        accChain.setHistogramOptions(histogram_opt);
                    }

                    // get values and accumulate them
                    for(size_t i=0; i<edgeVector.size(); ++i){
                        accChain(volume[gridGraph.u(edgeVector[i])]);
                        accChain(volume[gridGraph.v(edgeVector[i])]);
                    }

                    // extract features (in last pass)
                    if(p+1 == nPasses){
                        //std::cout<<"extract\n";
                        extractFeatures(accChain, usedTag, options, eid, edgeFeatures);
                    }
                }
            }
        }


        if(options.nodeFeatures){
            //std::cout<<"do node features\n";
            // allocate a vector of accumulator chains
            std::vector<AccChain> accChainVec(rag.maxNodeId()+1);


            omp_lock_t * nodeLocks = new omp_lock_t[rag.maxNodeId()+1];


            #pragma omp parallel for
            for(size_t nid=0; nid<=rag.maxNodeId(); ++nid){
                if(rag.nodeFromId(nid)!=lemon::INVALID){
                    omp_init_lock(&nodeLocks[nid]);
                    activateTags(accChainVec[nid], usedTag);
                    accChainVec[nid].setHistogramOptions(histogram_opt);
                }
            }

            vigra::TinyVector<UInt32, 3> shape = labels.shape();


            size_t nPasses;
            for(size_t nid=0; nid<=rag.maxNodeId(); ++nid){
                if(rag.nodeFromId(nid)!=lemon::INVALID){
                    nPasses = accChainVec[nid].passesRequired();
                    break;
                }
            }
            
            for(size_t p=0; p < nPasses; ++p){
                #pragma omp parallel for
                for(size_t z=0; z<shape[2]; ++z){
                    GridGraph3dNode node;
                    node[2]=z;
                    for(node[1]=0; node[1]<shape[1]; ++node[1])
                    for(node[0]=0; node[0]<shape[0]; ++node[0]){
                        const UInt32 label = labels[node];

                        SKNEURO_CHECK_OP( bool(rag.nodeFromId(label)==lemon::INVALID),==,false, "");
                        SKNEURO_CHECK_OP(label,<=, rag.maxNodeId(), "");
                        // lock the label
                        omp_set_lock(&nodeLocks[label]);

                        // do the actual accumulation
                        accChainVec[label](volume[node]);

                        // unlock the label
                        omp_unset_lock(&nodeLocks[label]);
                    }    
                }
            }
            delete[] nodeLocks;

            // extract features
            #pragma omp parallel for
            for(size_t nid=0; nid<=rag.maxNodeId(); ++nid){
                if(rag.nodeFromId(nid)!=lemon::INVALID){
                    extractFeatures(accChainVec[nid], usedTag, options, nid, nodeFeatures);
                }
            }
        }
    }
   
}
    




#endif /*SKNEURO_FEATURE_EXTRACTION_HXX*/
