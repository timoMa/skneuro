#define BOOST_TEST_MAIN
#if !defined( WIN32 )
    #define BOOST_TEST_DYN_LINK
#endif
#include <boost/test/unit_test.hpp>

#include <vigra/accumulator.hxx>

namespace vacc = vigra::acc;


// most frequently you implement test cases as a free functions with automatic registration
BOOST_AUTO_TEST_CASE( testWeightedAcc )
{

    typedef vacc::AccumulatorChain< vigra::MultiArrayView<1,float>,
        vacc::Select<vacc::Mean> > AccType;

    vigra::MultiArray<1,float> a(vigra::MultiArray<1,float>::difference_type(4));
    vigra::MultiArray<1,float> b(vigra::MultiArray<1,float>::difference_type(4));

    AccType acc;

    a[0]=2.0;
    a[1]=2.0;
    a[2]=2.0;
    a[3]=2.0;

    b[0]=4.0;
    b[1]=4.0;   
    b[2]=4.0;
    b[3]=4.0;

    acc(a, 1.0);
    acc(b, 2.0);

    vigra::MultiArray<1,float> m(vigra::MultiArray<1,float>::difference_type(4));

    m = vacc::get< vacc::Mean >(acc);

   BOOST_CHECK_CLOSE( m(0),3.0, 0.0001 );
   BOOST_CHECK_CLOSE( m(0),10.0/3, 0.0001 );
}

