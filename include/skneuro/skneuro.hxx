#ifndef SKNEURO_HXX
#define SKNEURO_HXX

#include <stdexcept>
#include <sstream>



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