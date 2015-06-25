#ifndef SKNEURO_IIR_GAUSSIAN_HXX
#define SKNEURO_IIR_GAUSSIAN_HXX

#include <vigra/multi_convolution.hxx>
#include <vigra/convolution.hxx>
#include <vigra/tinyvector.hxx>
#include <vigra/multi_iterator.hxx>
#include <vigra/multi_blocking.hxx>
#include <vigra/multi_array.hxx>

#include <math.h>

namespace skneuro{

/**
 *  
 * gaussian iir from http://www.getreuer.info/home/gaussianiir
 *
 * \file gaussianiir3d.c
 * \brief Fast 3D Gaussian convolution IIR approximation
 * \author Pascal Getreuer <getreuer@gmail.com>
 * 
 * Copyright (c) 2011, Pascal Getreuer
 * All rights reserved.
 * 
 * This program is free software: you can redistribute it and/or modify it
 * under, at your option, the terms of the GNU General Public License as 
 * published by the Free Software Foundation, either version 3 of the 
 * License, or (at your option) any later version, or the terms of the 
 * simplified BSD license.
 *
 * You should have received a copy of these licenses along with this program.
 * If not, see <http://www.gnu.org/licenses/> and
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */
void gaussianiir3dImpl(float *volume, long width, long height, long depth, float sigma, int numsteps)
{
    const long plane = width*height;
    const long numel = plane*depth;
    double lambda, dnu;
    float nu, boundaryscale, postscale;
    float *ptr;
    long i, x, y, z;
    int step;
    
    if(sigma <= 0 || numsteps < 0)
        return;
    
    lambda = (sigma*sigma)/(2.0*numsteps);
    dnu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda))/(2.0*lambda);
    nu = (float)dnu;
    boundaryscale = (float)(1.0/(1.0 - dnu));
    postscale = (float)(pow(dnu/lambda,3*numsteps));
    
    /* Filter horizontally along each row */
    for(z = 0; z < depth; z++)
    {
        for(y = 0; y < height; y++)
        {
            for(step = 0; step < numsteps; step++)
            {
                ptr = volume + width*(y + height*z);
                ptr[0] *= boundaryscale;
                
                /* Filter rightwards */
                for(x = 1; x < width; x++)
                    ptr[x] += nu*ptr[x - 1];
                
                ptr[x = width - 1] *= boundaryscale;
                
                /* Filter leftwards */
                for(; x > 0; x--)
                    ptr[x - 1] += nu*ptr[x];
            }
        }
    }
    
    /* Filter vertically along each column */
    for(z = 0; z < depth; z++)
    {
        for(x = 0; x < width; x++)
        {
            for(step = 0; step < numsteps; step++)
            {
                ptr = volume + x + plane*z;
                ptr[0] *= boundaryscale;
                
                /* Filter downwards */
                for(i = width; i < plane; i += width)
                    ptr[i] += nu*ptr[i - width];
                
                ptr[i = plane - width] *= boundaryscale;
                
                /* Filter upwards */
                for(; i > 0; i -= width)
                    ptr[i - width] += nu*ptr[i];
            }
        }
    }
    
    /* Filter along z-dimension */
    for(y = 0; y < height; y++)
    {
        for(x = 0; x < width; x++)
        {
            for(step = 0; step < numsteps; step++)
            {
                ptr = volume + x + width*y;
                ptr[0] *= boundaryscale;
                
                for(i = plane; i < numel; i += plane)
                    ptr[i] += nu*ptr[i - plane];
                
                ptr[i = numel - plane] *= boundaryscale;
                
                for(; i > 0; i -= plane)
                    ptr[i - plane] += nu*ptr[i];
            }
        }
    }
    
    for(i = 0; i < numel; i++)
        volume[i] *= postscale;
    
    return;
}


void gaussianIIR(vigra::MultiArrayView<3, float> & data, float sigma, int numsteps){
    // assert that not strided

    std::cout<<" 1,0,0 "<< &data(1,0,0) - &data(0,0,0) <<"\n";
    std::cout<<" 0,1,0 "<< &data(0,1,0) - &data(0,0,0) <<"\n";
    std::cout<<" 0,0,1 "<< &data(0,0,1) - &data(0,0,0) <<"\n";

    //if( data.size() != 
    //    (&data(data.size()-1) - &data(0))  + 1 
    //)
    //{
    //    std::cout<<"data.size() "<<data.size()<<"\n";
    //    std::cout<<"address diff "<<(&data(data.size()-1) - &data(0))   <<"\n";
    //    throw std::runtime_error("must be dense");
    //}


    gaussianiir3dImpl(&data(0), data.shape(0), data.shape(1), data.shape(2), sigma, numsteps);
}









}


#endif // SKNEURO_IIR_GAUSSIAN_HXX