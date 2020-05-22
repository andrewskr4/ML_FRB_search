<<<<<<< HEAD
//**************************************************************************
// gen_samples.cpp                                   
=======
//***************************************************************************
//                             FIL2SPEC.c                                  
>>>>>>> d4a809dcdbb1df453be43047e3d451c166aa0357
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
#include <random>
#include <cmath>

//#include "sigproc.h"
//int read_header(FILE *inputfile) ;


double dmdelay(double f1, double f2, double dm)
{
  return(-1*4148.741601*((1.0/f1/f1)-(1.0/f2/f2))*dm);
}

int dmshift(double f1, double f2, int nchans, double dm, double refrf, double tsamp, int *shift)
{
  int i;
<<<<<<< HEAD
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

int dedisperse(unsigned char *dataOut, unsigned char *buffer, int *shift, int nchans, int num_samples)
{
	for(int sample=0;sample<num_samples;sample++)
	{
	  for(int channel=0; channel<nchans; channel++)
		{
		  dataOut[sample*nchans+channel] = buffer[(sample+shift[channel])*nchans+channel];
	  }
	}
	return 0;
=======
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
>>>>>>> d4a809dcdbb1df453be43047e3d451c166aa0357
}

int normalize(float *plot)
{
<<<<<<< HEAD
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

int add_pulse(unsigned char *dataOut, int* flags, int nchans, int width)
{
	for(int sample=14*256+(512-width/2); sample<18*256-(512-width/2); sample++)
	{
		for(int channel=0;channel<nchans;channel++)
		{
			dataOut[sample*nchans+channel] += (unsigned char)((int)(rand()%6)+1);
	  }
	}
	return 0;
=======
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
>>>>>>> d4a809dcdbb1df453be43047e3d451c166aa0357
}


int main(int argc, char *argv[])
{
<<<<<<< HEAD
  for(int k=0; k<7; k++){
    std::cout<<k<<std::endl;
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

    float ddm = (rand() %3000)/100;
    float offset_dm = dm_value + ddm;

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
    unsigned char *buffer = new unsigned char[nchans*4096*32*3];
    unsigned char *tbuffer = new unsigned char[nchans*4096*32*3];
    unsigned char *dataOut = new unsigned char[nchans*(8192*3+32768*3)];
    float *avg_data = new float[nchans*40960];
    float *smooth_data = new float[nchans*256*7];
	
    dmshift(fch1, fch1+(foff*nchans), nchans, offset_dm, fch1, tsamp, shift);	
    int p=0;
    long count=0;
    srand(time(NULL));
    const float mean = 0.0;
    const float stdev = 0.01;
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, stdev);
  
    fread(buffer,sizeof(char),nchans*1024,fpin);
    sample_location +=1024;

    FILE *fpout;
    float *plot = new float[256*256*4];
 	
    std::cout<<"started nchans "<<shift[nchans-1]<<" \n";	
  
     while(!feof(fpin))
    { 
      if(p==0) 
      {
        fread(buffer,sizeof(char),nchans*4096*32*3,fpin);
        count = nchans*4096*32*3;
        printf("reading file %ld\r",(count*100)/fileSize);
      }
      else
      {
        memcpy(tbuffer,&buffer[8192*3*nchans],sizeof(char)*4096*30*3*nchans);
        memcpy(buffer,tbuffer,sizeof(char)*4096*30*3*nchans);
        fread(&buffer[4096*30*3*nchans],1,8192*3*nchans,fpin);
        count += 8192*3*nchans;
        printf("reading file %ld\r",(count*100)/fileSize);
      }
      //fflush(stdout);
      for  (int j=0; j<nchans*4096*32; j++){
        buffer[j] += dist(generator);
      }

      dedisperse(dataOut, buffer, shift, nchans, 8192*3+32768*3);
		
      for(int i=0; i<nchans*40960; i++) avg_data[i] = 0.0;
      for(int i=0; i<nchans*256*7; i++) smooth_data[i] = 0.0;

      for(int i=0;i<40960;i++)
      {
        for(int j=0; j<nchans; j++)
        {
          avg_data[i*nchans+j] = (float)(int)dataOut[3*i*nchans+j] + (float)(int)dataOut[(3*i+1)*nchans+j] + (float)(int)dataOut[(3*i+2)*nchans+j];
        }
      }
    
      for(int smooth=1;smooth<17;smooth=smooth*2)
      {
        for(int iter=0; iter<64; iter=iter+smooth)
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
                  if(flags[4*channel+c]==0) plot[256*channel+sample] += avg_data[(iter*128+smooth*sample+s)*nchans+4*channel+c];
                  else plot[256*channel+sample] += 0.0;
	        }
	      }
	    }
          }
          	    

          normalize(plot);  
		  
	  char *file_name = new char[200];
          //std::cout<<k<<std::endl;
	  sprintf(file_name,"sample_%d_%i.plt",start_number+p, k);
	  fpout = fopen(file_name,"wb");
          //if(p%100==0) std::cout<<file_name<<"\n";
      
						
	  float temp= (float)tstart;
	  fwrite(&temp,sizeof(float),1,fpout);
	  temp= (float)((double)(sample_location+(iter*128*3))*tsamp);
	  fwrite(&temp,sizeof(float),1,fpout);
	  temp= dm_value;
	  fwrite(&temp,sizeof(float),1,fpout);
	  temp= (float)smooth;
	  fwrite(&temp,sizeof(float),1,fpout);
	  temp= 0.0;
	  fwrite(&temp,sizeof(float),1,fpout);
	  temp = dm_value;
	  fwrite(&temp,sizeof(float),1,fpout);
	  temp = 0.0; //set to 0 to label as rfi
	  fwrite(&temp,sizeof(float),1,fpout);
		  
	  //printf("Finished reading %s\n",file_name);
																																										 
						
	  fwrite(plot,sizeof(float),256*256,fpout);
	  fclose(fpout);
	  free(file_name);
	  p++;
				
        }
      }
      sample_location += 8192*3;
    }
=======
  FILE *fpin , *fp_premask;
  long fileSize,headerSize;
  int i;
  int start_number=0;
  float dm_value=0.0;
  double sample_location=0;
  float toa = 0;
  double f_max = 800;
  double f_min = 400;


  
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
    if(strcmp(argv[i],"-t")==0)
    {
      i++;
      toa= atof(argv[i]);
      std::cout<<"toa "<<toa<<std::endl;
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

  float true_dm = dm_value;
  
  long count=0;
  srand(time(NULL));
  const float mean = 0.0;
  const float stdev = 0.05;
  std::default_random_engine generator;
  std::normal_distribution<double> dist(mean, stdev);
    
  for(int j =0; j<10; j++){



    dmshift(fch1, fch1+(foff*nchans), nchans, dm_value, fch1, tsamp, shift);
    
    int jump = (toa/tsamp)*nchans*sizeof(float);
    
    
    sample_location = jump/(nchans*sizeof(float));
    fseek(fpin,4096*sample_location+headerSize,SEEK_SET);
    
    count = sample_location;
    FILE *fpout;
    float *plot = new float[256*256*4];
    
    std::cout<<"started nchans "<<shift[nchans-1]<<" \n";
    
    int total_samples = fileSize/(nchans*sizeof(float));
    
    int nread = fread(buffer,sizeof(float),nchans*4096*32,fpin);
    for  (int j=0; j<nchans*4096*32; j++){
      buffer[j] += dist(generator);
    }
    count += nread/nchans;
    printf("reading file %ld %ld %d\n",sample_location,count,nread/nchans);
    
    fflush(stdout);
    
    dedisperse(dataOut, buffer, shift, nchans, 8192+4096);
    
    int num_samples = total_samples - sample_location;
    //std::cout<<"num_samples = "<<num_samples<<std::endl;
    
    if(num_samples>64*128) num_samples=64*128;
    
    for(int smooth=1;smooth<17;smooth=smooth*2) //(int smooth=1;smooth<17;smooth=smooth*2)
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
			if(flags[4*channel+c]==0) plot[256*channel+sample] += dataOut[(smooth*sample+s)*nchans+4*channel+c];
			else plot[256*channel+sample] += 0.0;
			
		      }
		  }
	      }
	  }
	
	
	normalize(plot);
	
	char *file_name = new char[200];
	sprintf(file_name,"nn_sample_%d_%i_%i_.plt",start_number+j, smooth, j);
	fpout = fopen(file_name,"wb");
	
	float temp= (float)tstart;
	fwrite(&temp,sizeof(float),1,fpout);
	temp= (float)((double)(sample_location*tsamp));
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
	//p++;
      }
    //sample_location += 8192;
>>>>>>> d4a809dcdbb1df453be43047e3d451c166aa0357
  }
  return(0);
}
