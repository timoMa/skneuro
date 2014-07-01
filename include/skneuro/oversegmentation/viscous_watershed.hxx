#ifndef SKNEURO_OVERSEGMENTATION_VISCOUS_WATERSHED_HXX
#define SKNEURO_OVERSEGMENTATION_VISCOUS_WATERSHED_HXX


namespace skneuro{


struct CvWSNode
{
	struct CvWSNode* next;
	int mask_ofs;
	int img_ofs;
};

struct CvWSQueue
{
	CvWSNode* first;
	CvWSNode* last;
};


static CvWSNode*
icvAllocWSNodes( CvMemStorage* storage )
{
	CvWSNode* n = 0;	
	int i, count = (storage->block_size - sizeof(CvMemBlock))/sizeof(*n) - 1;
	n = (CvWSNode*)cvMemStorageAlloc( storage, count*sizeof(*n) );
	for( i = 0; i < count-1; i++ )
		n[i].next = n + i + 1;
	n[count-1].next = 0;
	return n;
}




template<class T>
void viscousWatershed(
	const vigra::MultiArrayView<2,T> & indicator,
	vigra::MultiArrayView<2,T> & seeds
){


}




} // end namespace skneuro

#endif /* SKNEURO_OVERSEGMENTATION_VISCOUS_WATERSHED_HXX */


