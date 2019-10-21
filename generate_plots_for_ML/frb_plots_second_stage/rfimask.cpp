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
		//std::cout<<"mean : "<<mean<<" rms: "<<rms<<"\n";
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
  FILE *fpin , *fp_premask, *fpdm;
  long fileSize,headerSize;
  int i;
	int sample_shift=0,sample_correction=0;  
  //printf("opening file \n");

	fpin = fopen(argv[1],"rb");
  if(fpin==NULL)
	{
    char* file_name = new char[100];
		char *cmd_out = new char[100];
		strcpy(file_name,argv[1]);
		file_name[6] = '*';
		file_name[7] = '*';
    sprintf(cmd_out,"find /data/ -name '%s'",file_name);
		printf("file_name: %s \n",file_name);
		FILE* file = popen(cmd_out,"r");
		
		fscanf(file, "%100s", cmd_out);
		pclose(file);
		printf("buffer is :%s\n", cmd_out);
    fpin = fopen(cmd_out,"rb");
	   if(fpin==NULL)
		 {	
		   //printf("Cannot open the file. Please check the filename %s\n",argv[1]);
       //exit(0);
		 }
  }
  
  for(i=0;i<argc;i++)
  {
    if(strcmp(argv[i],"-i")==0)
    {
      i++;
      fp_premask= fopen(argv[i],"r");
      if(fp_premask==NULL)
      {
        printf("Cannot open the file. Please check the filename %s\n",argv[i]);
        exit(0);
      }


      
    }
    
		if(strcmp(argv[i],"-s")==0)
    {
      i++;
      sample_shift= atoi(argv[i]);
          
    }
    
		if(strcmp(argv[i],"-c")==0)
	  {
		  i++;
		  sample_correction= atoi(argv[i]);
    }

  }

		
		
  fseek(fpin,0,SEEK_END);
  fileSize = ftell(fpin);
  fseek(fpin,0,SEEK_SET);
  
  read_header(fpin);
  headerSize = ftell(fpin);
  fileSize -= headerSize;
  
	int *event = (int*) malloc(sizeof(int)*1000);
	int *beam = (int*) malloc(sizeof(int)*1000);
	float *dm = (float*) malloc(sizeof(float)*1000);
  
	fpdm = fopen("/home/arun/final_list.txt","r");
	if(fpdm==NULL)
	{
		printf("Cannot open /home/arun/final_list.txt \n");
		exit(0);
  }
  
	int count=0;
	while(!feof(fpdm))
	{
		fscanf(fpdm,"%d %d %f \n",&event[count],&beam[count],&dm[count]);
		count++;
  }
  
	char *event_char = new char[100];
	int event_len =0;
	char character='t';
	//printf("%s %d \n",source_name,strlen(source_name));
	while(character!='_' && event_len < strlen(source_name))
	{
		character = source_name[4+event_len];
		event_len++;
  }
  memcpy(event_char,&source_name[4],event_len);
	int event_int = atoi(event_char);
	//printf("%d \n",ibeam);
	
	float dm_value =0.0;
  for(int i=0;i<count;i++)
	{
		 if(event[i]==event_int && beam[i]==ibeam) 
		 {
		   dm_value = dm[i];
			
		 }
	}
  
  if(dm_value==0.0) exit(0);


	
	
	int *flags = new int[nchans];
	for(int i=0;i<nchans;i++) flags[i]=0;
	
	int t;
	while(!feof(fp_premask))
	{
		fscanf(fp_premask,"%d",&t);
		flags[t] = 1;
  }

	int *shift = new int[nchans];
	float *buffer = new float[fileSize/sizeof(float)];
	float *dataOut = new float[fileSize/sizeof(float)];
	dmshift(fch1, fch1+(foff*nchans), nchans, dm_value, fch1, tsamp, shift);	
  
  FILE *fpout;
	float *plot = new float[256*256*4];
		
	//std::cout<<"started nchans "<<shift[nchans-1]<<" \n";	
	 
  srand(time(NULL));
	int num_use_samples = (fileSize/sizeof(float)/nchans)-shift[nchans-1];
	fread(buffer,sizeof(float),fileSize/sizeof(float),fpin);
	
  for(int dm_shift=-5;dm_shift<6;dm_shift++)
	{
	  dmshift(fch1, fch1+(foff*nchans), nchans, dm_value+dm_shift, fch1, tsamp, shift);
	  dedisperse(dataOut, buffer, shift, nchans, num_use_samples-1);
	  
	   //printf("Entered the loop \n"); 
	   for(int smooth=1;smooth<sample_shift;smooth=smooth*2)
	   {	   
			 for(int i=0; i<256*256*4; i++) plot[i]=0; 
		   for(int sample=0;sample<256;sample++)
		   {	
		     for(int channel=0;channel<256;channel++)
			   {
			     for(int c=0; c<4; c++)
				   {
					   for(int s=0; s<1; s++)
					   {
						   for(int sm=0;sm<smooth;sm++)
					     {
						     if(flags[4*channel+c]==0) plot[256*channel+sample] += dataOut[(num_use_samples-128*sample_shift-sample_correction-(smooth-1)*128)*nchans+(sample*smooth+sm)*nchans+(4*channel+c)];
						     else plot[256*channel+sample] += 0.0;

					     }
					   }
				   }
			   }
  
		  }
		  normalize(plot);  
		  char *file_name = new char[200];
		
		  sprintf(file_name,"pulse_%d_%d_%d_%d_%d.plt",event_int,ibeam,smooth,sample_shift,dm_shift+5);
		  //printf("creating the file %s",file_name);
			fpout = fopen(file_name,"wb");
		
		  float temp= (float)event_int;
		  fwrite(&temp,sizeof(float),1,fpout);
		  temp= (float)ibeam;
		  fwrite(&temp,sizeof(float),1,fpout);
      temp= dm_value;
		  fwrite(&temp,sizeof(float),1,fpout);
		  temp= (float)smooth;
		  fwrite(&temp,sizeof(float),1,fpout);
		  temp= (float)sample_shift;
			fwrite(&temp,sizeof(float),1,fpout);
			temp = float(dm_shift+5);
		  fwrite(&temp,sizeof(float),1,fpout);
		  temp = 1.0;
		  fwrite(&temp,sizeof(float),1,fpout);
		  fwrite(plot,sizeof(float),256*256,fpout);
		  fclose(fpout);
		  free(file_name);
		  		
	  }
	}
	fclose(fpin);
	fclose(fp_premask);
	return(0);
}
