/* 
   writeooty2fb - converts ORT  data into "filterbank" data 
   adapted from f77 code gmrt2fb. 
   BCJ
   15/02/2012
*/
#include "filterbank.h"
void writeooty2fb(double *spect, FILE *output,  int highchan, int lowchan) 
{
  unsigned short ur[highchan+1];
  int b,c;
  c=lowchan;
  b=0;
  for (c=lowchan;c<=highchan;c++,b++) 
    {
      ur[b]=((unsigned short) spect[c]);
      if(b < 0) ur[b]=0.0;
      if(b > highchan) ur[b]=0.0;
    }
  ur[0]=ur[1]=ur[highchan-1]=ur[highchan]=0;
  fwrite(ur,sizeof(short),highchan+1,output);
}
