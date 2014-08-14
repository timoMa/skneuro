#ifndef SKNEURO_FEATURE_EXTRACTION_HXX
#define SKNEURO_FEATURE_EXTRACTION_HXX

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



    template<class PIXEL_TYPE, class FEATURE_TYPE>
    void accumulateFeatures(
        const GridGraph3d &                             gridGraph,
        const Rag &                                     rag,
        const GridGraph3dAffiliatedEdges &              affiliatedEdges,
        const vigra::MultiArrayView<3, PIXEL_TYPE >  &  volume,
        const float histMin,
        const float histMax,
        const size_t nBins,
        const float histSigma,
        vigra::MultiArrayView<2, FEATURE_TYPE > &       features
    ){
        // check input sanity
        SKNEURO_CHECK_OP(rag.edgeNum(), >, 0, "no edges");
        SKNEURO_CHECK_OP(rag.edgeNum(), == , rag.maxEdgeId()+1, "malformed graph");

        using namespace vigra::acc;

        typedef UserRangeHistogram<0> Hist;
        typedef Select< Mean, Minimum, Maximum, Variance, Hist> Selection;
        typedef AccumulatorChain<double,  Selection > AccChain;
        //typedef DynamicAccumulatorChain<double,  Selection > AccChain;

        //  set the options
        PIXEL_TYPE minVal,maxVal; 
        vigra::HistogramOptions histogram_opt;
        histogram_opt = histogram_opt.setBinCount(nBins);
        histogram_opt = histogram_opt.setMinMax(histMin, histMax);

        std::vector<AccChain> accChainVec(rag.edgeNum());

        const size_t nPasses = accChainVec.front().passesRequired();
        for(size_t p=0; p < nPasses; ++p){
            #pragma omp parallel for
            for(size_t eid = 0; eid< rag.edgeNum(); ++eid){

                const GridGraph3dEdgeVector & edgeVector = affiliatedEdges[rag.edgeFromId(eid)];
                AccChain & accChain = accChainVec[eid];
                if(p == 0){
                    accChain.setHistogramOptions(histogram_opt); 
                }
                for(size_t i=0; i<edgeVector.size(); ++i){
                    //std::cout<<"   i "<<i<<"\n";
                    const GridGraph3dNode u = gridGraph.u(edgeVector[i]);
                    const GridGraph3dNode v = gridGraph.v(edgeVector[i]);
                    const PIXEL_TYPE uVal = volume[u];
                    const PIXEL_TYPE vVal = volume[v];
                    accChain(uVal);
                    accChain(vVal);
                }

                // DONE last pass 
                // extract features
                if(p+1 == nPasses){
                    size_t fIndex=0;
                    features(eid,fIndex++) = get<Mean>(accChain);
                    features(eid,fIndex++) = get<Variance>(accChain);

                    vigra::MultiArray<1, double> hist = get<Hist>(accChain);
                    vigra::MultiArray<1, double> sHist(hist.shape());

                    gaussianSmoothMultiArray(hist, sHist, histSigma);

                    for(size_t bi=0; bi<nBins; ++bi){
                        features(eid,fIndex++) = sHist[bi];
                    }
                }
            }
        }
    }
}

#endif /*SKNEURO_FEATURE_EXTRACTION_HXX*/