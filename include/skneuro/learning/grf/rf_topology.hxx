#ifndef SKNEURO_LEARNING_RF_TOPOLOGY_HXX
#define SKNEURO_LEARNING_RF_TOPOLOGY_HXX

#ifndef WITH_LEMON_GRAPH
#define WITH_LEMON_GRAPH
#endif

#ifndef WITH_LEMON
#define WITH_LEMON
#endif


#include <vigra/graphs.hxx>
#include <lemon/list_graph.h>


namespace skneuro{

    class RfTopology  : lemon::ListDigraph
    {
        public:
            RfTopology()
            : lemon::ListDigraph()
            {

            }
            RfTopology(const RfTopology & other)
            : lemon::ListDigraph()
            {

            }
        private:
            //lemon::ListDigraph * digraph_;
    };


}


#endif /*SKNEURO_LEARNING_RF_TOPOLOGY_HXX */
