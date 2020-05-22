#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "global.h"

double delay(double f1, double f2, double dm) /* includefile */
{
  return(4148.741601*((1.0/f1/f1)-(1.0/f2/f2))*dm);
}


int rotate(float *data, int numSamples, int shift)
{
  int i=0;
  float* temp = (float*) malloc(sizeof(float)*numSamples);
  if(shift>=0)
  {
    shift = shift%numSamples;
    
    for(i=shift;i<numSamples;i++)
    {
      temp[i-shift]=data[i];
    }
    for(i=0;i<shift;i++)
    {
      temp[numSamples-shift+i] = data[i];
    }
  }
  if(shift<0)
  {
    //shift = shift%numSamples;
    
    for(i=0;i<numSamples+shift;i++)
    {
      temp[i-shift]=data[i];
    }
    for(i=numSamples+shift;i<numSamples;i++)
    {
      temp[i-numSamples-shift] = data[i];
    }
  }
  memcpy(data,temp,sizeof(float)*numSamples);
  free(temp);
  return 0;
}

int dedisperse_float(float *data, double f1, double f2, int nchans, int numSamples, double dm, double refrf, double tsamp) 
{
  int i;
  double fi;
  double df;
  int shift;
  df = (f2-f1)/((double)(nchans));
     
  fi = f1;
  for (i=0; i<nchans; i++) 
  {
    fi=f1+df*i;
    if (i==0) printf("%lf %lf %lf %lf \n",fi,f2,dm,tsamp);
    shift=(int)(delay(fi,f1,dm)/tsamp);
    rotate(&data[i*numSamples],numSamples,shift);
     
  }
  return 0;
}


