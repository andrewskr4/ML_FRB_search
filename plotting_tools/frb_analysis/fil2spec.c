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
#include "filterbank.h"
#include "global.h"


int main(int argc, char *argv[])
{
  int i, j,b,bin,flag=0;
  int nacc,*shift,maxShift,location,total,nbins,*count,tnchans;
  unsigned char *ptr;
  double real_part, img_part, spectra[2048];
  float ts=10,f1,f2;
  double cumspectra[2048], tempadd;
  unsigned short *dArray;
  unsigned short *finalArray;
  long nread, fileSize,c=0,tc;
  double dm=0.0,time,temptime,period;
  float *spec;
  long double *dspec;
  FILE *fpin, *fpout;
  struct POLYCO polyco;
  char* polyco_file;
  polyco_file = (char*) malloc(sizeof(char)*200);

  if(argc<6)
  {
    printf("fil2spec <filterbank file name> -p <polyco_file> -dm <Dm of pulsar> -nacc < accumuation for the channels> \n");
    printf(" -n <num bins> -ts <sub integration time (s)> -o <outfile name (.spec)>\n");
    exit(0);
  }
  
  fpin = fopen(argv[1],"rb");
  if(fpin==NULL)
  {
    printf("Cannot open the file. Please check the filename \n");
    exit(0);
  }
  
  fseek(fpin,0,SEEK_END);
  fileSize = ftell(fpin);
  fseek(fpin,0,SEEK_SET);
  
  period =  0.5491399331;
  dm =10;
  ts = 10;
  nbins = 512;
  nacc = 1;
  for(i=0;i<argc;i++)
  {
     
    if(strcmp(argv[i],"-o")==0)
    {
      i++;
      fpout = fopen(argv[i],"wb");
    }
    if(strcmp(argv[i],"-n")==0)
    {
      i++;
      nbins = atoi(argv[i]);
      
    }
    if(strcmp(argv[i],"-ts")==0)
    {
      i++;
      ts = atof(argv[i]);
    }
    if(strcmp(argv[i],"-dm")==0)
    {
      i++;
      dm = atof(argv[i]);
    }
    if(strcmp(argv[i],"-p")==0)
    {
      i++;
      strcpy(polyco_file,argv[i]);
        
    }
    if(strcmp(argv[i],"-nacc")==0)
    {
      i++;
      nacc = atoi(argv[i]);
    }
  }
  if(fpout==NULL)
  {
    printf("Cannot open the file. Please check the space on the disk \n");
    exit(0);
  }
  
  read_header(fpin);
  get_nearest_polyco(polyco_file,tstart,&polyco);
      period=polyco_period(tstart,polyco);
      printf("tstart = %lf period = %f \n",tstart,period);

  // Fill the header and braodcast
  tnchans = nchans/nacc;
  
  printf(" %d %lf %lf %lf %d\n",nbits,fch1,foff,tstart,tnchans);
   
  nbits=16;
  data_type=-1;
  obits=-1;
  nifs = 1;
  nbeams=1;
  ibeam=0;
  sumifs=1;
   
 
  shift = (int *) malloc(nchans * sizeof(int));
  count = (int *) malloc(nbins * sizeof(int));
  memset(count,0,sizeof(int)*nbins);
  dmshift1(fch1, fch1+foff*nchans, nchans, dm, fch1, tsamp,shift);
  
  maxShift = shift[nchans-1];
  total = shift[nchans-1]*nchans;
  dArray = (unsigned short*) malloc(sizeof(unsigned short)*total);
  dspec = (long double*) malloc(sizeof(long double)*nbins*nchans);
  memset(dspec,0,sizeof(long double)*nbins*nchans);
  finalArray = (unsigned short*) malloc(nchans*sizeof(unsigned short));  
  spec = (float*) malloc(sizeof(float)*nbins*tnchans);
  memset(spec,0,sizeof(float)*nbins*tnchans);
  
  
  f1 = fch1;
  f2 = fch1+foff*nchans;
  tc = (long)(ts/tsamp);
  ts = tc*tsamp;
  printf("tc = %ld ts = %lf \n",tc,ts);
  fwrite(&f1,sizeof(float),1,fpout);
  fwrite(&f2,sizeof(float),1,fpout);
  fwrite(&tnchans,sizeof(int),1,fpout);
  fwrite(&nbins,sizeof(int),1,fpout);
  fwrite(&ts,sizeof(float),1,fpout);
  
  nread = fread(dArray,sizeof(short),total,fpin);
  if(nread!=total)
  {
  	printf("Input file is too small %ld %d\n",nread,total);
  	exit(0);
  }
  c+=maxShift;
  location =0;
 
  dedisperse1(finalArray, dArray, shift, nchans,location); 
  time =0.0;
  
  time += tsamp;
  temptime += tsamp;
  bin = ((int)((time/period)-((int) (time/period)))*nbins);
  
  for(i=0;i<nchans;i++)
  {
  	dspec[nbins*i+bin]=finalArray[i];
  }
  
   
  
  
  while(!feof(fpin))
  {
    
    nread = fread(&dArray[location*nchans],sizeof(short),nchans,fpin);
    location++;
    if(location==maxShift) location=0;
    dedisperse1(finalArray, dArray, shift, nchans,location);
    time += tsamp;
    temptime += tsamp;
    bin = (int)(((time/period)-floor(time/period))*nbins);
    c++;
    for(i=0;i<nchans;i++)
    {
  	  dspec[nbins*i+bin]+=finalArray[i];
  	}
     count[bin]++;
    //printf("%ld \n",c);  
    if(c%tc==0)
    {
      for(i=0;i<tnchans;i++)
      {
        for(b=0;b<nbins;b++)
        {
          for(j=0;j<nacc;j++)
          {
            spec[nbins*i+b] += (float)(dspec[nbins*(nacc*i+j)+b]/count[b]);
          }
          spec[nbins*i+b] /= nacc;
        }
      }
      fwrite(spec,sizeof(float),tnchans*nbins,fpout);
      memset(dspec,0,nchans*nbins*sizeof(long double));
      memset(spec,0,tnchans*nbins*sizeof(float));
      memset(count,0,nbins*sizeof(int));
      temptime = 0;
    }
    
    if(c%1000==0)
    {
      printf("%lf %lf percentage = %ld %% %ld\r",time,temptime,c*nchans*200/fileSize,c);
      fflush(stdout);
    }
     
     
  }
  
  free(shift);
  free(count);
  free(dArray);
  free(dspec);
  free(finalArray);
  free(spec);
  fclose(fpin);
  fclose(fpout);

  return(0);
}
