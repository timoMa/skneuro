#ifndef SKEURO_LEARNING_PATCH_RF_HXX
#define SKEURO_LEARNING_PATCH_RF_HXX


// std
#include <queue>
#include <vector>
#include <algorithm>
#include <vigra/random.hxx>
#include <lemon/list_graph.h>
#include <vigra/algorithm.hxx>

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

        typedef vigra::TinyVector<int, 4> FIndex;

        typedef vigra::TinyVector<int, 3> PIndex;
        typedef std::pair<PIndex,PIndex> PIndexPair;
        typedef std::vector<PIndexPair>  EvalDimVec;


        struct Parameter{
            Parameter(){
                patchRadius_ = 3;
                mtry_ = 0;
                nEvalDims_ = 100;
            }
            size_t patchRadius_;
            size_t mtry_;
            size_t nEvalDims_;
        };


        struct SplitInfo{
            FIndex splitFeature;
            T featureVal;
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
            evalDims_(param.nEvalDims_),
            distA_(vigra::TinyVector<int,1>(param.nEvalDims_)),
            distB_(vigra::TinyVector<int,1>(param.nEvalDims_))
        {
        }   

        void randEvalDims(){
            for(size_t i=0; i<evalDims_.size(); ++i){
                evalDims_[i].first  = randPatchPoint();
                evalDims_[i].second = randPatchPoint();
                while( evalDims_[i].first==evalDims_[i].second)
                    evalDims_[i].second = randPatchPoint();
            }
        }

        PIndex randPatchPoint()const{
            PIndex index;
            for(size_t d=0; d<3; ++d){
                index[d] = randgen_.uniformInt(param_.patchRadius_*2 +1)-param_.patchRadius_;
            }
            return index;
        }
        FIndex randFeature()const{
            FIndex index;
            for(size_t d=0; d<3; ++d){
                index[d] = randgen_.uniformInt(param_.patchRadius_*2 +1)-param_.patchRadius_;
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
                buffer[i] = features_[fIndex];
            }
        }

        void resetIndices(std::vector<size_t> & indices)const{
            for(size_t i=0; i<indices.size(); ++i){
                indices[i] = i;
            }
        }


        // find a split
        template<class INSTANCE>
        SplitInfo findSplit(
            const std::vector<INSTANCE> & instIn,
            std::vector<INSTANCE> & instOutA,
            std::vector<INSTANCE> & instOutB
        ){  


            const size_t nInstances = instIn.size();

            // eval dims
            randEvalDims();

            SKNEURO_CHECK_OP(nInstances, >=, 2, "error, to few instances for split");

            double bestSplitValue = std::numeric_limits<double>::infinity();
            FIndex bestFeatureIndex = FIndex(0);
            T bestThreshold = T();


            std::vector<float>          featureBuffer(instIn.size());
            std::vector<size_t>         sortedIndices(instIn.size());
            std::vector<vigra::UInt8>   splitOutput(instIn.size(),1);

            for(size_t  tryNr=0; tryNr</*param_.mtry_*/ 1; ++tryNr){

                // select a random feature index
                FIndex rFeatureIndex = randFeature();

                // fill buffer for that feature
                fillBuffer(instIn, rFeatureIndex, featureBuffer);

                // reset indices and get sorted indices
                resetIndices(sortedIndices);
                vigra::indexSort(featureBuffer.begin(),featureBuffer.end(), sortedIndices.begin());

                // smallest goes to left (0)
                // largest goes to right (1)
                // - in between we try out 
                //   all splits
                splitOutput[sortedIndices[0]] = 0;
                splitOutput[sortedIndices[nInstances-1]] = 1;

                // try out all the splits
                for(size_t i=1; i<nInstances-1; ++i){

                    // set split output according to threshold
                    const T thisThreshold = featureBuffer[sortedIndices[i]];
                    splitOutput[i] = 0;
                    if(i%100==0)
                        evalSplit(instIn, splitOutput);
                }
            }

          
        }

        template<class INSTANCE>
        double evalSplit(
            const std::vector<INSTANCE>     & instIn,
            const std::vector<vigra::UInt8> & partition
        ){
            distA_  = 0.0;
            distB_  = 0.0;

            for(size_t i=0; i<instIn.size(); ++i){
                const vigra::UInt8 label = partition[i];
                // loop over all dimensions
                for(size_t d=0; d<evalDims_.size(); ++d){
                    const bool inSameCluster = labels_[instIn[i] + evalDims_[d].first] == labels_[instIn[i] + evalDims_[d].second];
                    if(!inSameCluster){
                        if(label==0)
                            distA_(d)+=1.0;
                        else
                            distB_(d)+=1.0;
                    }
                }
            }

            double distBetween = 0.0;
            distA_/=instIn.size();
            distB_/=instIn.size();

            // compute between distance
            for(size_t d=0; d<evalDims_.size(); ++d){
                const double dist = distA_[d]-distB_[d];
                distBetween += dist*dist;
            }
            distBetween = std::sqrt(distBetween);
       
            size_t cA = 0;
            size_t cB = 0;
            double dAT = 0.0;
            double dBT = 0.0;
            // within dist
            for(size_t i=0; i<instIn.size(); ++i){

                double dA = 0.0;
                double dB = 0.0;

                const vigra::UInt8 label = partition[i];
                // loop over all dimensions
                if(label==0){
                    ++cA;
                }
                else{
                    ++cB;
                }

                for(size_t d=0; d<evalDims_.size(); ++d){
                    const bool inSameCluster = labels_[instIn[i] + evalDims_[d].first] == labels_[instIn[i] + evalDims_[d].second];
                    const double val  = inSameCluster ? 0.0 : 1.0;

                    //std::cout<<"val "<<val<<"\n";

                    if(label == 0 ){
                        const double dist = val - distA_[d];
                        dA  += dist*dist;
                    }
                    else{
                        const double dist = val - distB_[d];
                        dB  += dist*dist;
                    }
                }
                dA = std::sqrt(dA);
                dB = std::sqrt(dB);

                //std::cout<<"DA "<<dA<<" DB "<<dB<<"\n";

                dAT+=dA;
                dBT+=dB;
            }


            const double withinDist = (dAT+dBT)/instIn.size();
            //std::cout<<"within dist "<<withinDist<<"\n";
            const double totalDist  = withinDist/distBetween;
            std::cout<<"total dist "<<totalDist<<"\n";

        }



    private:

        FeatureVolume features_;
        LabelVolume labels_;
        Parameter param_;
        vigra::RandomNumberGenerator<> randgen_;
        EvalDimVec evalDims_;

        vigra::MultiArray<1,double> distA_;
        vigra::MultiArray<1,double> distB_;
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

            trainTree(instances.begin(), instances.end(), 
                      splitFinder, topology, splitInfoMap,
                      leafNodeMap);


        }





        Param param_;
    };





}


#endif /*SKEURO_LEARNING_PATCH_RF_HXX*/
