#ifndef SKEURO_LEARNING_PATCH_RF_HXX
#define SKEURO_LEARNING_PATCH_RF_HXX


// std
#include <queue>
#include <vector>
#include <algorithm>
#include <vigra/random.hxx>
#include <lemon/list_graph.h>

namespace skneuro{
    

    template<class L>
    void findValids(
        const vigra::MultiArrayView<3, L> & labels,
        std::vector< vigra::TinyVector< vigra::UInt16, 3> >  & coords,
        const int r
    ){
        vigra::MultiArray<3, bool>  isValid(labels.shape());
        isValid = true;
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(labels(x,y,z) == 0 ){
                for(int xx=-1*r; xx <= r; ++xx){
                    int xxx = x+xx;
                    if(xxx>=0 && xxx<labels.shape(0)){
                        isValid(xxx, y, z) = false;
                    }
                }
            }
        }
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == false ){
                for(int yy=-1*r; yy <= r; ++yy){
                    int yyy = y+yy;
                    if(yyy>=0 && yyy<labels.shape(1)){
                        isValid(x, yyy, z) = false;
                    }
                }
            }
        }
        for(int z=0; z<labels.shape(2); ++z)
        for(int y=0; y<labels.shape(1); ++y)
        for(int x=0; x<labels.shape(0); ++x){
            if(isValid(x,y,z) == false ){
                for(int zz=-1*r; zz <= r; ++zz){
                    int zzz = z+zz;
                    if(zzz>=0 && zzz<labels.shape(2)){
                        isValid(x, y, zzz) = false;
                    }
                }
            }
        }

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
        TOPOLOGY & topology, SPLIT_PARAM_MAP & splitParamMap,
        LEAF_NODE_MAP & leafNodeMap
    ){
        typedef typename SPLIT_PARAM_MAP::Value SplitParam;
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

            splitParamMap[topoNode] = splitFinder.findSplit(instIn, outA, outB);

            std::cout<<outA.size()<<" "<<outB.size()<<"\n";
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

        typedef vigra::TinyVector<int, 4> FeatureVolumeIndex;

        struct Param{
            FeatureVolumeIndex splitFeature;
            T featureVal;
        };

        PatchSplitFinder(
            FeatureVolume & features,
            LabelVolume & labels
        )
        :   features_(features),
            labels_(labels),
            randgen_()
        {
        }   

        // find a split
        template<class INSTANCE>
        Param findSplit(
            const std::vector<INSTANCE> & instIn,
            std::vector<INSTANCE> & instOutA,
            std::vector<INSTANCE> & instOutB
        ){
            // reset
            instOutA.resize(0);
            instOutB.resize(0);

            // dummy split
            Param dummySplit;
            
            //dummySplit.splitFeature[i] =0;

            INSTANCE randInst = instIn[randgen_.uniformInt(instIn.size())];
            FeatureVolumeIndex tmp;
            for(int i=0; i<3; ++i){
                tmp[i] = randInst[i];
            }
            tmp[3] = 0;
            dummySplit.featureVal = features_[tmp];
            for(size_t i=0; i < instIn.size(); ++i){
                for(size_t d=0; d<3; ++d){
                    tmp[d] = instIn[i][d];
                }
                if(features_[tmp]<dummySplit.featureVal){
                    instOutA.push_back(instIn[i]);
                }
                else{
                    instOutB.push_back(instIn[i]);
                }
            }
            return dummySplit;
        }


    private:

        FeatureVolume features_;
        LabelVolume labels_;
        vigra::RandomNumberGenerator<> randgen_;
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
        typedef typename SplitFinder::Param SplitParam;

        typedef Topology:: template NodeMap<SplitParam> SplitParamMap;
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
            SplitParamMap splitParamMap(topology);

            // result of tree
            LeafNodeMap leafNodeMap;

            SplitFinder splitFinder(features, labels);

            trainTree(instances.begin(), instances.end(), 
                      splitFinder, topology, splitParamMap,
                      leafNodeMap);


        }





        Param param_;
    };





}


#endif /*SKEURO_LEARNING_PATCH_RF_HXX*/
