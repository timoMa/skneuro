#ifndef SKEURO_LEARNING_PATCH_RF_HXX
#define SKEURO_LEARNING_PATCH_RF_HXX


// std
#include <queue>
#include <vector>
#include <algorithm>

// vigra
#include <vigra/random.hxx>
#include <lemon/list_graph.h>
#include <vigra/algorithm.hxx>


// skeuro
#include "split_finder.hxx"
namespace skneuro{
    

    template<class L>
    void findValids(
        const vigra::MultiArrayView<3, L> & labels,
        std::vector< vigra::TinyVector< vigra::UInt16, 3> >  & coords,
        const int r
    ){
        vigra::MultiArray<3, bool>  isValid(labels.shape(), false);
        vigra::MultiArray<3, bool>  isValid2(labels.shape(), false);

        for(int z=r+1; z<labels.shape(2)-r-1; ++z)
        for(int y=r+1; y<labels.shape(1)-r-1; ++y)
        for(int x=r+1; x<labels.shape(0)-r-1; ++x){
            isValid(x, y, z) = true;
            isValid2(x, y, z) = true;
        }


        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == false ){
                for(int xx=-1*r; xx <= r; ++xx){
                    int xxx = x+xx;
                    if(xxx>=0 && xxx<labels.shape(0)){
                        isValid2(xxx, y, z) = false;
                    }
                }
            }
        }
        isValid = isValid2;
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == false ){
                for(int yy=-1*r; yy <= r; ++yy){
                    int yyy = y+yy;
                    if(yyy>=0 && yyy<labels.shape(1)){
                        isValid2(x, yyy, z) = false;
                    }
                }
            }
        }
        isValid = isValid2;
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == false ){
                for(int zz=-1*r; zz <= r; ++zz){
                    int zzz = z+zz;
                    if(zzz>=0 && zzz<labels.shape(2)){
                        isValid2(x, y, zzz) = false;
                    }
                }
            }
        }
        isValid = isValid2;
        size_t counter = 0;
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == true ){
                ++counter;
            }
        }
        coords.resize(counter);
        counter=0;
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == true ){
                coords[counter][0] = x;
                coords[counter][1] = y;
                coords[counter][2] = z;
                ++counter;
            }
        }
    }
    


    template<class INSTANCE, class RG>
    void getBootstrap(
        const std::vector<INSTANCE> & instInAll,
        std::vector<INSTANCE> & instBootstrap,
        size_t btSize,
        RG & rg
    ){
        const size_t nTotal = instInAll.size();  
        std::vector<bool> isIncluded(nTotal, false);
        for(size_t i=0; i<btSize; ++i){
            isIncluded[rg.uniformInt(nTotal)] = true;
        }
        instBootstrap.resize(0);
        instBootstrap.reserve(btSize);
        for(size_t i=0; i<nTotal; ++i){
            if(isIncluded[i]){
                instBootstrap.push_back(instInAll[i]);
            }
        }

    }


    struct RfTopologyParam{
        enum  GraphTopology{
            Tree,
            Dag
        };


        GraphTopology graphTopology_;
        size_t maxDepth_;
        size_t minInstancesForSplit_;
        
    };

    struct PatchRfParam{
        PatchRfParam(){
            patchRadius = 5;
        }
        size_t patchRadius;
    };

    template<class INSTANCE_DESC, class TOPO_NODE>
    struct SplitInOut{
        typedef std::vector<INSTANCE_DESC> InstanceVec;
        TOPO_NODE topoNode;
        InstanceVec  instIn;
    };



    template<class INSTANCE_ITER, class SPLIT_FINDER,
             class TOPOLOGY, class SPLIT_PARAM_MAP,
             class LEAF_NODE_MAP
             >
    void trainTree(
        INSTANCE_ITER instancesBegin, INSTANCE_ITER instancesEnd,
        SPLIT_FINDER & splitFinder, 
        TOPOLOGY & topology, SPLIT_PARAM_MAP & splitInfoMap,
        LEAF_NODE_MAP & leafNodeMap
    ){
        typedef typename SPLIT_PARAM_MAP::Value SplitInfo;
        typedef typename std::iterator_traits<INSTANCE_ITER>::value_type InstanceDesc;
        typedef TOPOLOGY Topology;
        typedef typename Topology::Node TopologyNode;
        typedef SplitInOut<InstanceDesc,TopologyNode> InOut;
        typedef typename InOut::InstanceVec InstanceVec;
        const size_t nInstances = std::distance(instancesBegin, instancesEnd);
        std::cout<<"train tree with "<<nInstances<<" instances\n";

        InOut rootInOut;
        rootInOut.instIn.assign(instancesBegin, instancesEnd);
        rootInOut.topoNode =  topology.addNode();

        std::queue<InOut> toSplitQ;
        toSplitQ.push(rootInOut);

        while(!toSplitQ.empty()){

            // get the top to split
            InOut splitInOut = toSplitQ.front();
            toSplitQ.pop();

            // get the corresponding node,
            // corresponding IN instances 
            const TopologyNode topoNode = splitInOut.topoNode;
            InstanceVec & instIn = splitInOut.instIn;

            // output
            InOut inA, inB;

            InstanceVec & outA = inA.instIn;
            InstanceVec & outB = inB.instIn;

            splitInfoMap[topoNode] = splitFinder.findSplit(instIn, outA, outB);

            std::cout<<"out sizes"<<outA.size()<<" "<<outB.size()<<"\n";
            if(outA.size()>=2 && outB.size()>=2){



                // connect topology
                inA.topoNode = topology.addNode();
                inB.topoNode = topology.addNode();
                topology.addArc(topoNode,inA.topoNode);
                topology.addArc(topoNode,inB.topoNode);

                // add to queue
                toSplitQ.push(inA);
                toSplitQ.push(inB);
            }
            else{
                if(outA.size()>0){
                    inA.topoNode = topology.addNode();
                    topology.addArc(topoNode,inA.topoNode);
                    for(size_t i=0; i<outA.size(); ++i)
                        leafNodeMap[inA.topoNode].push_back(outA[i]);
                }
                if(outB.size()>0){
                    inB.topoNode = topology.addNode();
                    topology.addArc(topoNode,inB.topoNode);
                    for(size_t i=0; i<outB.size(); ++i)
                        leafNodeMap[inB.topoNode].push_back(outB[i]);
                }
            }
        }

    }

    


    template<class T, class L>
    class PatchSplitFinder{
    public:

        typedef vigra::MultiArrayView<4, T>  FeatureVolume;
        typedef vigra::MultiArrayView<3, L>  LabelVolume;

        typedef vigra::TinyVector<int, 4> FIndex;

        typedef vigra::TinyVector<int, 3> PIndex;
        typedef std::pair<PIndex,PIndex> PIndexPair;
        typedef std::vector<PIndexPair>  EvalDimVec;


        typedef vigra::MultiArray<2, unsigned char> ExplicitLabels;

        struct Parameter{
            Parameter(){
                patchRadius_ = 7;
                mtry_ = 0;
                nEvalDims_ = 1000;
                maxWeakLearnerExamples_ = 250000;
            }
            int patchRadius_;
            size_t mtry_;
            size_t nEvalDims_;
            size_t maxWeakLearnerExamples_;
        };


        struct SplitInfo{
            FIndex splitFeature;
            T featureValThreshold;
        };

        PatchSplitFinder(
            FeatureVolume & features,
            LabelVolume & labels,
            const Parameter & param 
        )
        :   features_(features),
            labels_(labels),
            param_(param),
            randgen_(),
            evalDims_(param.nEvalDims_)
        {
        }   

        void randEvalDims(){
            for(size_t i=0; i<evalDims_.size(); ++i){
                evalDims_[i].first  = randPatchPoint();
                evalDims_[i].second = randPatchPoint();
                while( evalDims_[i].first==evalDims_[i].second){
                    evalDims_[i].second = randPatchPoint();
                }
            }
        }

        PIndex randPatchPoint()const{
            PIndex index;
            for(size_t d=0; d<3; ++d){
                index[d] = randgen_.uniformInt(param_.patchRadius_*2 +1)-param_.patchRadius_;
                SKNEURO_CHECK_OP(index[d],<=,param_.patchRadius_,"");
                SKNEURO_CHECK_OP(index[d],>=,-1*param_.patchRadius_,"");
            }
            return index;
        }
        FIndex randFeature()const{
            FIndex index;
            for(size_t d=0; d<3; ++d){
                index[d] = randgen_.uniformInt(param_.patchRadius_*2 +1)-param_.patchRadius_;
                SKNEURO_CHECK_OP(index[d],<=,param_.patchRadius_,"");
                SKNEURO_CHECK_OP(index[d],>=,-1*param_.patchRadius_,"");
            }
            index[3] = randgen_.uniformInt(features_.shape(3));
            return index;
        }
        
        template<class INSTANCE>
        void fillBuffer(
            const std::vector<INSTANCE> & instIn,
            const FIndex fIndex,
            std::vector<T> & buffer
        ){
            for(size_t i=0; i<instIn.size(); ++i){
                FIndex tmp;
                for(size_t d=0; d<3; ++d){
                    tmp[d] = instIn[i][d] + fIndex[d];
                }
                tmp[3] = fIndex[3];
                buffer[i] = features_[tmp];
                //std::cout<<"buffer "<<buffer[i]<<"\n";
            }
        }

        void resetIndices(std::vector<size_t> & indices)const{
            for(size_t i=0; i<indices.size(); ++i){
                indices[i] = i;
            }
        }

        template<class INSTANCE>
        void makeExplicitLabels(
            const std::vector<INSTANCE> & instIn
        ){
            vigra::TinyVector<int, 2> shape(param_.nEvalDims_, instIn.size());
            explicitLables_.reshape(shape);

            for(size_t i=0; i<instIn.size(); ++i){
                // loop over all dimensions
                for(size_t d=0; d<evalDims_.size(); ++d){
                    const bool inSameCluster = labels_[instIn[i] + evalDims_[d].first] == labels_[instIn[i] + evalDims_[d].second];
                    explicitLables_(d, i) = static_cast<unsigned char>(inSameCluster);
                }
            }

        }

        // find a split
        template<class INSTANCE>
        SplitInfo findSplit(
            const std::vector<INSTANCE> & instInAll,
            std::vector<INSTANCE> & instOutA,
            std::vector<INSTANCE> & instOutB
        ){  
            std::cout<<"find split:\n";
            const size_t nTotal = instInAll.size();
            SKNEURO_CHECK_OP(nTotal, >=, 2, "error, to few instances for split");

            std::vector<INSTANCE> instInBt;
            if(nTotal> param_.maxWeakLearnerExamples_){
                getBootstrap(instInAll, instInBt, param_.maxWeakLearnerExamples_, randgen_);
            }

            const std::vector<INSTANCE> & instIn = nTotal> param_.maxWeakLearnerExamples_ ? 
                                                    instInBt : instInAll;
            const size_t nInstances = instIn.size();

            //std::cout<<"  bootstrap"<<nInstances<<"  "<<nTotal<<"\n";

            SKNEURO_CHECK_OP(nInstances, >=, 2, "error, to few instances for split");

            // eval dims
            randEvalDims();

            
            // make labeling explicit
            makeExplicitLabels(instIn);



            double bestEvalVal = std::numeric_limits<double>::infinity();
            SplitInfo splitInfo;


            std::vector<float>          featureBuffer(instIn.size());
            std::vector<size_t>         sortedIndices(instIn.size());
            std::vector<vigra::UInt8>   splitOutput(instIn.size(),1);
            bool foundPerfekt = false;


            typedef VarianceRediction<T, vigra::UInt8> VarRed;
            VarRed varReducer(explicitLables_);

            for(size_t  tryNr=0; tryNr<param_.mtry_ && !foundPerfekt; ++tryNr){
                //std::cout<<"   try "<<tryNr<<"\n";
                // select a random feature index
                FIndex rFeatureIndex = randFeature();
                //std::cout<<"randFetures : ";
               // /for(size_t ff=0; ff<4; ++ff){
               // /    std::cout<<rFeatureIndex[ff]<<" ";
               // /}
               // /std::cout<<"\n";

                // fill buffer for that feature
                fillBuffer(instIn, rFeatureIndex, featureBuffer);

                
                std::pair<T, double> res = varReducer.bestSplit(featureBuffer);
                if(res.second < bestEvalVal){
                    std::cout<<" varbest "<<std::setprecision(10)<<res.second <<"\n";
                    bestEvalVal = res.second;
                    splitInfo.splitFeature = rFeatureIndex;
                    splitInfo.featureValThreshold = res.first;
                }
            }

            featureBuffer.resize(nTotal);
            
            

            // realize split
            fillBuffer(instInAll, splitInfo.splitFeature, featureBuffer);
            for(size_t i=0; i<instInAll.size(); ++i){
                if(featureBuffer[i]<=splitInfo.featureValThreshold){
                    instOutA.push_back(instInAll[i]);
                }
                else{
                    instOutB.push_back(instInAll[i]);
                }
            }
            return splitInfo;
        }



    private:

        FeatureVolume features_;
        LabelVolume labels_;
        Parameter param_;
        vigra::RandomNumberGenerator<> randgen_;
        EvalDimVec evalDims_;

        vigra::MultiArray<1,double> distA_;
        vigra::MultiArray<1,double> distB_;

        ExplicitLabels explicitLables_;
    };


    template<class T, class L>
    class PatchRf{
    public:

        typedef vigra::TinyVector< vigra::UInt16, 3> Instance;
        typedef std::vector<Instance> InstanceVector;
        typedef vigra::MultiArrayView<4, T>  FeatureVolume;
        typedef vigra::MultiArrayView<3, L>  LabelVolume;

        typedef PatchRfParam Param;
        typedef lemon::ListDigraph Topology;
        typedef PatchSplitFinder<T,L> SplitFinder;
        typedef typename SplitFinder::Parameter SplitFinderParameter;
        typedef typename SplitFinder::SplitInfo SplitInfo;

        typedef Topology:: template NodeMap<SplitInfo> SplitInfoMap;
        typedef std::map<typename Topology::Node, InstanceVector> LeafNodeMap;
        PatchRf(const Param & param)
        :   param_(param){
        }


        void train(
            FeatureVolume & features,
            LabelVolume & labels
        ){

            
            InstanceVector instances;
            findValids(labels, instances, param_.patchRadius);  

            // encode the topology of the graph
            Topology topology;
            SplitInfoMap splitInfoMap(topology);

            // result of tree
            LeafNodeMap leafNodeMap;

            SplitFinderParameter splitFinderParam;

            splitFinderParam.patchRadius_ = param_.patchRadius;
            splitFinderParam.mtry_ = static_cast<size_t>(std::sqrt(std::pow(3.0*2.0+1.0, 3)*features.shape(3))+ 0.5);

            std::cout<<"mtry "<< splitFinderParam.mtry_ <<"\n";

            SplitFinder splitFinder(features, labels, splitFinderParam);

            // get the bootstrap
            InstanceVector bootstrapInstances;
            vigra::RandomNumberGenerator<> randgen_;
            getBootstrap(instances,bootstrapInstances,instances.size(), randgen_ );

            std::cout<<"BTSIZE "<< bootstrapInstances.size()<<"  "<<instances.size()<<"\n";

            trainTree(bootstrapInstances.begin(), bootstrapInstances.end(), 
                      splitFinder, topology, splitInfoMap,
                      leafNodeMap);


        }





        Param param_;
    };





}


#endif /*SKEURO_LEARNING_PATCH_RF_HXX*/
