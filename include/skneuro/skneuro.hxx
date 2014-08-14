#ifndef SKNEURO_HXX
#define SKNEURO_HXX

#include <stdexcept>
#include <sstream>
#include <vector>

#include <vigra/multi_array.hxx>
#include <vigra/multi_gridgraph.hxx>
#include <vigra/adjacency_list_graph.hxx>



namespace skneuro{

  typedef vigra::GridGraph<3, boost::undirected_tag> GridGraph3d;
  typedef vigra::MultiArrayView<3, vigra::UInt32 > GridGraph3dLablsView;
  typedef GridGraph3d::Edge GridGraph3dEdge;
  typedef GridGraph3d::Node GridGraph3dNode;
  typedef std::vector<GridGraph3dEdge> GridGraph3dEdgeVector;
  typedef vigra::AdjacencyListGraph Rag;
  typedef Rag::EdgeMap< GridGraph3dEdgeVector > GridGraph3dAffiliatedEdges;
}

// as runtime assertion but cefined even if NDEBUG

#define SKNEURO_CHECK_OP(a,op,b,message) \
    if(!  static_cast<bool>( a op b )   ) { \
       std::stringstream s; \
       s << "skneuro Error: "<< message <<"\n";\
       s << "skneuro check :  " << #a <<#op <<#b<< "  failed:\n"; \
       s << #a " = "<<a<<"\n"; \
       s << #b " = "<<b<<"\n"; \
       s << "in file " << __FILE__ << ", line " << __LINE__ << "\n"; \
       throw std::runtime_error(s.str()); \
    }

#define SKNEURO_CHECK(expression,message) if(!(expression)) { \
   std::stringstream s; \
   s << message <<"\n";\
   s << "skneuro assertion " << #expression \
   << " failed in file " << __FILE__ \
   << ", line " << __LINE__ << std::endl; \
   throw std::runtime_error(s.str()); \
 }


/// runtime assertion
#ifdef NDEBUG
   #ifndef SKNEURO_DEBUG 
      #define SKNEURO_ASSERT_OP(a,op,b) { }
   #else
      #define SKNEURO_ASSERT_OP(a,op,b) \
      if(!  static_cast<bool>( a op b )   ) { \
         std::stringstream s; \
         s << "skneuro assertion :  " << #a <<#op <<#b<< "  failed:\n"; \
         s << #a " = "<<a<<"\n"; \
         s << #b " = "<<b<<"\n"; \
         s << "in file " << __FILE__ << ", line " << __LINE__ << "\n"; \
         throw std::runtime_error(s.str()); \
      }
   #endif
#else
   #define SKNEURO_ASSERT_OP(a,op,b) \
   if(!  static_cast<bool>( a op b )   ) { \
      std::stringstream s; \
      s << "skneuro assertion :  " << #a <<#op <<#b<< "  failed:\n"; \
      s << #a " = "<<a<<"\n"; \
      s << #b " = "<<b<<"\n"; \
      s << "in file " << __FILE__ << ", line " << __LINE__ << "\n"; \
      throw std::runtime_error(s.str()); \
   }
#endif

#ifdef NDEBUG
   #ifndef SKNEURO_DEBUG
      #define SKNEURO_ASSERT(expression) {}
   #else
      #define SKNEURO_ASSERT(expression) if(!(expression)) { \
         std::stringstream s; \
         s << "skneuro assertion " << #expression \
         << " failed in file " << __FILE__ \
         << ", line " << __LINE__ << std::endl; \
         throw std::runtime_error(s.str()); \
      }
   #endif
#else
      #define SKNEURO_ASSERT(expression) if(!(expression)) { \
         std::stringstream s; \
         s << "skneuro assertion " << #expression \
         << " failed in file " << __FILE__ \
         << ", line " << __LINE__ << std::endl; \
         throw std::runtime_error(s.str()); \
      }
#endif


#endif // SKNEURO_HXX
