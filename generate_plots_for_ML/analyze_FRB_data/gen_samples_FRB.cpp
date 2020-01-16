//***************************************************************************
//                             FIL2SPEC.c                                  
//
// The following program takes the filterbank format file and  
// gives out the spec file which later can be plotted by foldDynamic 
//***************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fftw3.h"
extern "C" {
#include "filterbank.h"
#include "global.h"
}
#include <iostream>
#include <cstdlib>
#include <ctime>

//#include "sigproc.h"
//int read_header(FILE *inputfile) ;


double dmdelay(double f1, double f2, double dm)
{
  return(-1*4148.741601*((1.0/f1/f1)-(1.0/f2/f2))*dm);
}

int dmshift(double f1, double f2, int nchans, double dm, double refrf, double tsamp, int *shift)
{
  int i;
  double fi;
  double df;

  df = (f2-f1)/((double)(nchans));
  fi=f1;

  for (i=0; i<nchans; i++)
  {
    if (refrf > 0.0) f1=refrf;
    fi=f1+df*i;
    shift[i]=(int)(-1*dmdelay(fi,f1,dm)/tsamp);
  }
  return 0;
}

int dedisperse(float *dataOut, float *buffer, int *shift, int nchans, int num_samples)
{
  for(int sample=0;sample<num_samples;sample++)
  {
    for(int channel=0; channel<nchans; channel++)
    {
      dataOut[sample*nchans+channel] = buffer[(sample+shift[channel])*nchans+channel];
    }
  }
  return 0;
}

int normalize(float *plot)
{
  float rms=0.0, mean=0.0, sum=0.0,sq_sum=0.0;

  for(int i=0;i<256;i++)
  {
    sum=0.0;
    sq_sum=0.0;
    for(int j=0;j<256;j++)
    {
      sum+=plot[256*i+j];
      sq_sum += plot[256*i+j]*plot[256*i+j];
    }
    mean = sum/256;
    rms = sq_sum/256;
    rms = rms - (mean*mean);
    rms = sqrt(fabs(rms));

    for(int j=0;j<256;j++)
    {
      if(rms>0.0)
      {	
      plot[256*i+j]-=mean;
      plot[256*i+j]/=rms;
      }
      else plot[256*i+j] = 0.0;
    }
  }
  return 0;
}


int main(int argc, char *argv[])
{
  FILE *fpin , *fp_premask;
  long fileSize,headerSize;
  int i;
  int start_number=0;
  float dm_value=0.0;
  long sample_location=0;
  printf("opening file \n");
  fpin = fopen(argv[1],"rb");
  if(fpin==NULL)
  {
    printf("Cannot open the file. Please check the filename \n");
    exit(0);
  }

  for(i=0;i<argc;i++)
  {
    if(strcmp(argv[i],"-i")==0)
    {
      i++;
      fp_premask= fopen(argv[i],"r");
      if(fp_premask==NULL)
      {
        printf("Cannot open the file. Please check the filename \n");
        exit(0);
      }
    }

    if(strcmp(argv[i],"-s")==0)
    {
      i++;
      start_number= atoi(argv[i]);
    }

    if(strcmp(argv[i],"-d")==0)
    {
      i++;
      dm_value= atof(argv[i]);
    }

  }
  fseek(fpin,0,SEEK_END);
  fileSize = ftell(fpin);
  fseek(fpin,0,SEEK_SET);

  read_header(fpin);
  headerSize = ftell(fpin);
  fileSize -= headerSize;

  int *flags = new int[nchans];
  for(int i=0;i<nchans;i++) flags[i]=0;

  int t;
  while(!feof(fp_premask))
  {
    fscanf(fp_premask,"%d",&t);
    flags[t] = 1;
  }

  int *shift = new int[nchans];
  float *buffer = new float[nchans*4096*32];
  float *tbuffer = new float[nchans*4096*32];
  float *dataOut = new float[nchans*(8192+4096)];

  long count=0;
  dmshift(fch1, fch1+(foff*nchans), nchans, dm_value, fch1, tsamp, shift);
  int jump = fileSize - ((shift[nchans-1]+4096)*nchans*sizeof(float));
  fseek(fpin,jump+headerSize,SEEK_SET);
  sample_location = jump/(nchans*sizeof(float));
  count = sample_location;
  FILE *fpout;
  float *plot = new float[256*256*4];

  std::cout<<"started nchans "<<shift[nchans-1]<<" \n";
  int p=0;
  int total_samples = fileSize/(nchans*sizeof(float));
  srand(time(NULL));
  while(!feof(fpin)||sample_location<total_samples)
  {
    if(p==0)
    {
      int nread = fread(buffer,sizeof(float),nchans*4096*32,fpin);
      count += nread/nchans;
      printf("reading file %ld %ld %d\n",sample_location,count,nread/nchans);
    }
    else
    {
      memcpy(tbuffer,&buffer[8192*nchans],sizeof(float)*4096*30*nchans);
      memcpy(buffer,tbuffer,sizeof(float)*4096*30*nchans);
      if(!feof(fpin))
      {
        fread(&buffer[4096*30*nchans],sizeof(float),8192*nchans,fpin);
        count += 8192;
      }
      //else memset(&buffer[4096*30*nchans],0,sizeof(float)*8192*nchans);
      printf("reading file %ld\n",count);
    }
    fflush(stdout);
    dedisperse(dataOut, buffer, shift, nchans, 8192+4096);
    int num_samples = total_samples - sample_location;
    if(num_samples>64*128) num_samples=64*128;
    for(int smooth=1;smooth<17;smooth=smooth*2)
    {
      for(int iter=0; iter<num_samples/128; iter=iter+smooth)
      {
        for(int i=0; i<256*256; i++) plot[i]=0;	
        for(int sample=0;sample<256;sample++)
        {
          for(int channel=0;channel<256;channel++)
          {
            for(int c=0; c<4; c++)
            {
              for(int s=0; s<smooth ; s++)
              {
                if(flags[4*channel+c]==0) plot[256*channel+sample] += dataOut[(iter*128+smooth*sample+s)*nchans+4*channel+c];
                else plot[256*channel+sample] += 0.0;
              }
            }
          }
        }


        normalize(plot);

        char *file_name = new char[200];
        sprintf(file_name,"sample_%d.plt",start_number+p);
        fpout = fopen(file_name,"wb");

        float temp= (float)tstart;
        fwrite(&temp,sizeof(float),1,fpout);
        temp= (float)((double)(sample_location+(iter*128))*tsamp);
        fwrite(&temp,sizeof(float),1,fpout);
        temp= dm_value;
        fwrite(&temp,sizeof(float),1,fpout);
        temp= (float)smooth;
        fwrite(&temp,sizeof(float),1,fpout);
        temp= 0.0;
        fwrite(&temp,sizeof(float),1,fpout);
        temp = dm_value;
        fwrite(&temp,sizeof(float),1,fpout);
        temp = 0.0;
        fwrite(&temp,sizeof(float),1,fpout);
        fwrite(plot,sizeof(float),256*256,fpout);
        fclose(fpout);
        free(file_name);
        p++;
      }
    }
    sample_location += 8192;
  }
  return(0);
}
