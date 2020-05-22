#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <string.h>
#include "colorIndex.h"
#include <stdbool.h>
#include "filterbank.h"




int main(int argc, char* argv[])
{
  FILE *fpin, *fpout;
  unsigned char *data;
  float seconds=0.0;
  int i=0;

  fpin = fopen(argv[1],"rb");
  if(fpin==NULL)
  {
    printf("Cannot open the input file \n");
    exit(0);
  }
  
  for(i=0;i<argc;i++)
  {
    if(strcmp(argv[i],"-s")==0)
    {
      i++;
      seconds = atof(argv[i]);
    }

    if(strcmp(argv[i],"-o")==0)
    {
      i++;
      fpout = fopen(argv[i],"wb");
      if(fpout==NULL)
      {
        printf("Cannot open the output file \n");
        exit(0);
      }
    }
  
  }
  
  
  int numSamples=96*1024;
  read_header(fpin);
  data = (unsigned char*) malloc(sizeof(char)*96*1024*nchans);
  long header_size = ftell(fpin);
  fseek(fpin,0,SEEK_END);
  long data_size = ftell(fpin)-header_size;

  numSamples = (data_size/(nchans*sizeof(char)));
  if(numSamples>96*1024) numSamples=96*1024;
  fseek(fpin,0,SEEK_SET);
  read_header(fpin);
  
  if(seconds!=0)
  {
    tstart += ((seconds-5.0)/86400.0);
    int jump = (int)((seconds-5.0)/tsamp);
    fseek(fpin,sizeof(char)*nchans*jump+header_size, SEEK_SET);
  }
  
  fread(data,sizeof(char),numSamples*nchans,fpin);
  nbeams = 1;
  sumifs = 1;
  obits = 8;  
  nifs = 1;
  filterbank_header(fpout);

  fwrite(data,sizeof(char),numSamples*nchans,fpout);

  fclose(fpout);
  fclose(fpin);

  return 0;
}
