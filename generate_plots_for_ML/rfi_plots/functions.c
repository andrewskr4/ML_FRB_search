#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "global.h"

double dmdelay1(double f1, double f2, double dm) /* includefile */
{
  return(4148.741601*((1.0/f1/f1)-(1.0/f2/f2))*dm);
}

int dmshift1(double f1, double f2, int nchans, double dm, double refrf, double tsamp, int *shift) 
{
  int i;
  double fi;
  double df;
  
  df = (f2-f1)/((double)(nchans));
  //shift = (int *) malloc(nchans * sizeof(int));
  fi=f1;
  
  for (i=0; i<nchans; i++) 
  {
    if (refrf > 0.0) f1=refrf;
    fi=f1+df*i;
    shift[i]=(int)(dmdelay1(fi,f1,dm)/tsamp);
     
  }
  return 0;
}

int dedisperse1(unsigned short *finalArray, unsigned short *dArray, int *shift, int nchans, int location)
{
  int c,s,tempLocation,total;
  total = shift[nchans-1]*nchans;
  for(c=0;c<nchans;c++)
  {
    tempLocation = shift[c]*nchans+c+location*nchans;
    if(tempLocation<total)
      finalArray[c] = dArray[tempLocation];
    else
      finalArray[c] = dArray[tempLocation-total];
  }
  return 0;
} 

