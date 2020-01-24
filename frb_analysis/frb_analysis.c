#include<stdio.h>
#include<stdlib.h>
#include "cpgplot.h"
#include <math.h>
#include <string.h>
#include "colorIndex.h"
#include <stdbool.h>
#include "filterbank.h"
//#include "filterbank_headr.h"


//settings for the PGplot window
float window_size_x = 5000.0;
float window_size_y = 5500.0;
float image_size_x = 4000.0;
float image_size_y = 4000.0;
float image_origin_x = 1000.0;
float image_origin_y = 1500.0;

int save =0;

int dedisperse_float(float *data, double f1, double f2, int nchans, int numSamples, double dm, double refrf, double tsamp);

float cal_mean(float *data, int N)
{
  double sum=0.0;
  int i=0;

  for(i=0;i<N;i++) sum += data[i];

  return (float)(sum/N);
}

float median(int n, float *x) 
{
    
    float temp;
    int i, j;
    // the following two loops sort the array x in ascending order
    for(i=0; i<n-1; i++) {
        for(j=i+1; j<n; j++) {
            if(x[j] < x[i]) {
                // swap elements
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }
    }

    if(n%2==0) {
        // if there is an even number of elements, return mean of the two elements in the middle
        temp = ((x[n/2] + x[n/2 - 1]) / 2.0);
        
    } else {
        // else return the element in the middle
        temp = x[n/2];
    }

    return temp;
}

int readHeader(FILE *fp,float *f1, float *f2, int *nchan, int *nbins, float *ts)
{
  fread(f1,sizeof(float),1,fp);
  fread(f2,sizeof(float),1,fp);
  fread(nchan,sizeof(int),1,fp);
  fread(nbins,sizeof(int),1,fp);
  fread(ts,sizeof(float),1,fp);
  return 0;
}

int selection_on_plot(float *t_XREF, float *t_YREF, float *t_X, float *t_Y, float x_origin, float y_origin, float x_size, float y_size)
{
  char character;
  float XREF = t_XREF[0], YREF = t_YREF[0], Y = t_Y[0], X = t_X[0], tX,tY;
  cpgband(2,0,XREF,YREF,&X,&Y,&character);
  printf("the world corrdinates are %f , %f , %c \n",X,Y,character);
  XREF = X;
  YREF = Y;
  cpgband(2,0,XREF,YREF,&X,&Y,&character);
  cpgsfs(4);
  cpgrect(XREF,X,YREF,Y);
  printf("the world corrdinates are %f , %f , %c \n",X,Y,character);

  
  

  if(XREF>X)
  {
    tX = X;
    X  =XREF;
    XREF =tX;
  }
  if(YREF>Y)
  {
    tY = Y;
    Y =YREF;
    YREF =tY;
  }
  if(XREF < x_origin)
  {
    XREF = x_origin;
  }
  if(X > x_origin+x_size)
  {
    X = x_size + x_origin;
  }
  if(YREF < y_origin)
  {
    YREF = y_origin;
  }
  if(Y > y_origin+y_size)
  {
  Y = y_size+y_origin;
  }
  
 
  t_XREF[0] = XREF-x_origin;
  t_YREF[0] = YREF-y_origin;
  t_X[0] = X-x_origin;
  t_Y[0] = Y-y_origin;

  printf("%f %f %f %f \n",XREF,YREF,X,Y);
  return 0;
}
 


int plotDynamicSpectra(float *data, float f1, float f2, int plot_b1, float ts, int numSamples, int nchans,float mean,float std, float th)
{
  
  float tr[] = { image_origin_x-1.0*image_size_x/(float)(2*numSamples), (float)(image_size_x/(float)numSamples), 0, image_origin_y-1.0*image_size_y/(float)(2*nchans), 0, (float)(image_size_y/(float)nchans) };
  if(save==1) cpgopen("dynamicSpectra.ps/CPS");
  else cpgbeg(0,"/xs",1,1);
  cpgenv(0,window_size_x,0,window_size_y,0,-2);
  char palette[10];
  strcpy(palette,"CUBE");
  colorIndex(palette);
  
  printf("Plotting the image \n");
  cpgimag(data, numSamples, nchans, 1, numSamples, 1,nchans, mean-th*std,mean+th*std , tr);
  printf("Plotting \n");

  //nornamlized axis of profile and band
  cpgslw(4);
  cpgsch(1.5);
  
  cpgaxis("N", image_origin_x, 0.0, image_size_x+image_origin_x, 0.0, ts*plot_b1, ts*(numSamples+plot_b1), 0, 2, 0.5, 0.0, 0.25, 0.25, 0);
  cpgaxis(" ", image_origin_x, image_origin_y, image_size_x+image_origin_x, image_origin_y, ts*plot_b1, ts*(numSamples+plot_b1), 0, 2, 0.5, 0.0, 0.25, 0.25, 0);
  cpgaxis(" ", image_origin_x, image_origin_y+image_size_y, image_origin_x+image_size_x, image_origin_y+image_size_y, ts*plot_b1, ts*(numSamples+plot_b1), 0, 2, 0.0, 0.5, 0.25, 0.5, 0);
  
  cpgaxis("N", 0.0, image_origin_y, 0.0, image_size_y+image_origin_y, f1, f2, 0, 2, 0.0, 0.5, 0.25, -0.5, 0);
  cpgaxis(" ", image_origin_x, image_origin_y, image_origin_x, image_size_y+image_origin_y, f1, f2, 0, 2, 0.0, 0.5, 0.25, -0.5, 0);
  cpgaxis(" ", image_size_x+image_origin_x, image_origin_y, image_size_x+image_origin_x, image_size_y+image_origin_y, f1, f2, 0, 2, 0.5, 0.0, 0.25, 0.5, 0);
  
  cpgaxis("N", image_origin_x, 0.0, image_origin_x, image_origin_y, -0.2, 1.2, 0.5, 2, 0.5, 0.25, 0.25, -0.5, 0);
  cpgaxis(" ", image_origin_x+image_size_x, 0.0, image_origin_x+image_size_x, image_origin_y, -0.2, 1.2, 0.5, 2, 0.5, 0.25, 0.25, -0.5, 0);
  
  cpgaxis("N", 0.0, image_origin_y, image_origin_x, image_origin_y, -0.2, 1.2, 0.5, 2, 0.5, 0.25, 0.25, 0.5, 0);
  cpgaxis(" ", 0.0, image_origin_y+image_size_y, image_origin_x, image_origin_y+image_size_y, -0.2, 1.2, 0.5, 2, 0.5, 0.25, 0.0, 0.5, 0);

  float *prof = (float*) malloc(sizeof(float)*numSamples);
  float *xval = (float*) malloc(sizeof(float)*numSamples);
  memset(prof,0,sizeof(float)*numSamples);

  int i,j;
  float sum=0,m,max=-1e10;
   
  
  for(i=0;i<nchans; i++)
  {
    for(j=0; j<numSamples; j++)
    {
      prof[j] += data[i*numSamples+j];
    }
  }
  
  for(j=0; j<numSamples; j++) prof[j] /= nchans;
  
  m = cal_mean(prof,numSamples);
  max = prof[0]-m;
  for(j=0; j<numSamples; j++) 
  {
    prof[j] -= m;
    if(max<prof[j]) max = prof[j];
  }
  
  for(j=0; j<numSamples; j++) 
  {
    xval[j] = image_origin_x+j*(image_size_x/(float)numSamples);
    prof[j] = ((prof[j]/(1.4*max))+0.2)*(image_origin_y);

  }

  cpgline(numSamples,xval,prof);

  free(xval);
  free(prof);
  
  float *band = (float*) malloc(sizeof(float)*nchans);
  float *yval = (float*) malloc(sizeof(float)*nchans);
  
  memset(band,0,sizeof(float)*nchans);

  
  for(i=0;i<nchans; i++)
  {
    for(j=0; j<numSamples; j++)
    {
      band[i] += data[i*numSamples+j];
    }


  }

  for(i=0; i<nchans; i++) band[i] /= numSamples;
  
  m = cal_mean(band,nchans);
  max = band[0]-m;
  for(i=0; i<nchans; i++)
  {
    band[i] -= m;
    if(max<band[i]) max = band[i];
  }

  for(i=0; i<nchans; i++)
  {
    yval[i] = image_origin_y+i*(image_size_y/(float)nchans);
    band[i] = ((band[i]/(1.4*max))+0.2)*(image_origin_x);

  }

  cpgline(nchans,band,yval);

  free(yval);
  free(band);
  
  cpgslw(4);
  //cpglab(" ", " ", "DYNAMIC SPECTRA");
  cpgmtxt("L",1.8,0.5,.5,"Frequency (MHz)");
  cpgmtxt("B",2.5,0.5,.5,"Time (s)");
  //cpgend();
  return 0;
}


int adjust_spectra(float *temp, float* data, char* weights, char *flags, float mean, int plot_c1, int plot_c2, int plot_b1, int plot_b2, int plot_numSamples, int numSamples, int ncrunch, int tcrunch)
{
  int i,j,ii,jj;
  int f=0,w=0;
  for(i=plot_c1;i<plot_c2;i++)
  {
    for(j=plot_b1;j<plot_b2;j++)
    {
      f=0;
      for(ii=0;ii<ncrunch;ii++)
      { 
        for(jj=0;jj<tcrunch;jj++)
        {
          f += (int)flags[(ncrunch*i+ii)*numSamples+tcrunch*j+jj];
        }
      }
      float sum=0.0;
      if(f==0) 
      {
        w=0;
        for(ii=0;ii<ncrunch;ii++)
        {
          for(jj=0;jj<tcrunch;jj++)
          {
            sum += data[(ncrunch*i+ii)*numSamples+(tcrunch*j+jj)];
            w += (int)weights[(ncrunch*i+ii)*numSamples+(tcrunch*j+jj)];
          }
        }
        if(w==0) w++;
        temp[(i-plot_c1)*plot_numSamples+j-plot_b1] = sum/((float)(w));
      }
      else temp[(i-plot_c1)*plot_numSamples+j-plot_b1] = mean;
    }
  }
  
  //scale the data (x-mean)/std

  float sum=0.0,sqsum=0.0;
  for(i=0;i<plot_c2-plot_c1;i++)
  { 
    sum=0.0;
    sqsum = 0.0;
    for(j=0;j<plot_b2-plot_b1;j++)
    {
      sum += temp[i*plot_numSamples+j];
      sqsum += pow(temp[i*plot_numSamples+j],2.0);
    }
    float std = sqrt(fabs(pow(sum/plot_numSamples,2)-(sqsum/plot_numSamples)));
    for(j=0;j<plot_b2-plot_b1;j++)
    {
      temp[i*plot_numSamples+j] -= sum/plot_numSamples;
      if(std!=0) temp[i*plot_numSamples+j] /= std;
    }
  }  
  return 0;
}





int main(int argc, char* argv[])
{
  int i,j,ii,jj,ncrunch=1,tcrunch=1;
  FILE *fpin;
  float *data,*temp, dm;
  char *flags, *weights, *flagfile;
  float seconds=0.0;
  double mjd=0.0;
  int jump=0;
  char *file_data_char;
  unsigned short *file_data_short; 
  float* file_data_float;
  int *file_flags;
  
  fpin = fopen(argv[1],"rb");
  if(fpin==NULL)
  {
    printf("Cannot open the input file \n");
    exit(0);
  }
  
  flagfile = (char*) malloc(sizeof(char)*200);
  
  read_header(fpin);
  file_flags = (int*) malloc(sizeof(int)*nchans);

  for(i=0;i<argc;i++)
  {
    if(strcmp(argv[i],"-s")==0)
    {
      i++;
      seconds = atof(argv[i]);
      seconds -= 20;
      jump = (int)(seconds/tsamp);
    }
    if(strcmp(argv[i],"-mjd")==0)
    {
      i++;
      mjd = atof(argv[i]);
      jump = (int)(((mjd-tstart)*86400.0)/tsamp);
    }
    if(strcmp(argv[i],"-i")==0)
    {
      i++;
      strcpy(flagfile,argv[i]); 
      FILE *fp;
      fp = fopen(flagfile,"r");
      int k,channel;
      for(k=0;k<nchans;k++) file_flags[k] = 0;
      while(!feof(fp)) 
      {
        fscanf(fp,"%d",&channel);
        printf("%d \n",channel);
        file_flags[channel] = 1;
      }
      fclose(fp);   
    }

  }
  
  int numSamples=16000;

  
  long header_size = ftell(fpin);
  fseek(fpin,0,SEEK_END);
  long data_size = ftell(fpin)-header_size;
  printf("%d \n",nbits);
  if(nbits==8) numSamples = (data_size/(nchans*sizeof(char)));
  else if(nbits==16) numSamples = (data_size/(nchans*sizeof(char)));
  else if(nbits==32) numSamples = (data_size/(nchans*sizeof(float)));
  else
  {
    printf("Does not support the format \n");
    exit(0);
  }

  if(numSamples>128*1024) numSamples=128*1024;
  fseek(fpin,0,SEEK_SET);
  read_header(fpin);
  
  fch1 = 800;
  float f1=fch1,f2=fch1-400;
  printf("nchans %d and numSamples %d \n",nchans,numSamples);
  
  if(nbits==8)
  {
    file_data_char = (char*) malloc(sizeof(char)*nchans*numSamples);
  }
  else if(nbits==16) 
  {
    file_data_short = (unsigned short*) malloc(sizeof(unsigned short)*nchans*numSamples);
  }
  else if(nbits==32)
  {
    file_data_float = (float*) malloc(sizeof(float)*nchans*numSamples);
  }
  else 
  {
    printf("Does not support the format \n");
    exit(0);
  }
  
  printf("done allocation \n");
  data = (float*) malloc(sizeof(float)*nchans*numSamples);
  temp = (float*) malloc(sizeof(float)*nchans*numSamples);
  flags = (char*) malloc(sizeof(char)*nchans*numSamples);
  weights = (char*) malloc(sizeof(char)*nchans*numSamples);

  memset(flags,0,sizeof(char)*nchans*numSamples);
  
  for(i=0;i<nchans;i++)
  {
    for(j=0;j<numSamples;j++)
    {
      flags[i*numSamples+j] = file_flags[i];
    }
  }
   
  if(jump!=0)
  {
    if(nbits==8) fseek(fpin,sizeof(char)*nchans*jump+header_size, SEEK_SET);
    else if(nbits==16) fseek(fpin,sizeof(unsigned short)*nchans*jump+header_size, SEEK_SET);
    else fseek(fpin,sizeof(float)*nchans*jump+header_size, SEEK_SET);
  }
  
  if(nbits==8) fread(file_data_char,sizeof(char),numSamples*nchans,fpin);
  else if(nbits==16) fread(file_data_short,sizeof(unsigned short),numSamples*nchans,fpin);
  else fread(file_data_float,sizeof(float),numSamples*nchans,fpin);

  for(i=0;i<nchans;i++) 
  {
    for(j=0;j<numSamples;j++) 
    {
      if(nbits==8) data[i*numSamples+j] = (float)file_data_char[j*nchans+i];
      else if(nbits==16) data[i*numSamples+j] = (float)file_data_short[j*nchans+i];
      else data[i*numSamples+j] = file_data_float[j*nchans+i];
     
      if(data[i*numSamples+j]>0.0 && file_flags[i]==0) weights[i*numSamples+j]=1;
      else weights[i*numSamples+j]=0;
    }
  }
  double sum=0.0,sqsum=0.0,total_sum=0.0, total_sqsum;
  float med=0;
  //float* temp_median = (float*) malloc(sizeof(float)*numSamples*nchans);
  //memcpy(temp_median,data,sizeof(float)*numSamples*nchans);
  int total_w=0;
  for(i=0;i<nchans;i++)
  {
    sum=0.0;
    sqsum = 0.0;
    for(j=0;j<numSamples;j++)
    {
      sum+= data[i*numSamples+j];
      sqsum += pow(data[i*numSamples+j],2);
    }
    float std = sqrt(fabs((float)(sqsum/numSamples)-pow((float)(sum/numSamples),2)));
    printf("std of the plot %d %f \n",i,std);      
    for(j=0;j<numSamples;j++) 
    { 
      data[i*numSamples+j] -= sum/numSamples;
      if(std!=0.0) data[i*numSamples+j] /= std; 
    }
    sum=0.0;
    sqsum = 0.0;
    for(j=0;j<numSamples;j++)
    {
      total_sum+= data[i*numSamples+j];
      total_sqsum += pow(data[i*numSamples+j],2);
      total_w += weights[i*numSamples+j];
    }
  }
  //free(temp_median);
    
  float mean = total_sum/(total_w);
  float std = sqrt(total_sqsum/(total_w));
  float th=3.0;
  printf("numSample %d nchans %d mean = %f, std= %0.10f \n",numSamples,nchans,mean,std);
  
  char character = 'i';
  float XREF, YREF, X, Y, tX, tY;
  
  int c1,c2,b1,b2,plot_nchans,plot_numSamples,plot_c1,plot_c2,plot_b1,plot_b2;
  float plot_dm=0.0;
  
  plot_nchans = nchans;
  plot_numSamples = numSamples;
  plot_c1 = 0;
  plot_c2 = nchans;
  plot_b1 = 0;
  plot_b2 = numSamples;
  float foff = (f2-f1)/nchans;
  //dedisperse_float(data, f1, f2, nchans, numSamples, 1000 , 400.0, 1.0e-3);
  //dedisperse_float(data, f1, f2, nchans, numSamples, 207.4, 400.0, tsamp);
  memcpy(temp,data,sizeof(float)*numSamples*nchans);
  plotDynamicSpectra(temp, f1, f2, plot_b1, tsamp, plot_numSamples, plot_nchans,mean,std,th);

  while(character != 'x')
  {
    XREF = 0.0;
    YREF = 0.0;
    cpgcurs(&X,&Y,&character);
    
    if(character == 'z')
    {
      selection_on_plot(&XREF,&YREF,&X,&Y,image_origin_x,image_origin_y,image_size_x,image_size_y);
      
      b1 = (int)((XREF/image_size_x)*plot_numSamples);
      b2 = (int)((X/image_size_x)*plot_numSamples);
      c1 = (int)((YREF/image_size_y)*plot_nchans);
      c2 = (int)((Y/image_size_y)*plot_nchans);
      
      plot_nchans = c2-c1;
      plot_numSamples = b2-b1;
      plot_c1 += c1;
      plot_c2 = plot_c1+plot_nchans;
      plot_b1 += b1;
      plot_b2 = plot_b1+plot_numSamples;
      
      adjust_spectra(temp, data, weights, flags, mean, plot_c1, plot_c2, plot_b1, plot_b2, plot_numSamples, numSamples, ncrunch, tcrunch);
      plotDynamicSpectra(temp, f1+ foff*plot_c1*ncrunch, f1+ foff*plot_c2*ncrunch, plot_b1, tsamp*tcrunch, plot_numSamples, plot_nchans,mean,std,th);

      
    }

    

    if(character == 'b')
    {
      c1 = 0;
      c2 = nchans/ncrunch;
      b1 = 0;
      b2 = numSamples/tcrunch;
      plot_nchans = nchans/ncrunch;
      plot_numSamples = numSamples/tcrunch;
      plot_c1 = 0;
      plot_c2 = nchans/ncrunch;
      plot_b1 = 0;
      plot_b2 = numSamples/tcrunch;
       
      adjust_spectra(temp, data, weights, flags, mean, plot_c1, plot_c2, plot_b1, plot_b2, plot_numSamples, numSamples, ncrunch, tcrunch);
      plotDynamicSpectra(temp, f1+ foff*plot_c1*ncrunch, f1+ foff*plot_c2*ncrunch, plot_b1, tsamp*tcrunch, plot_numSamples, plot_nchans,mean,std,th);
     
    }
    
    

    if(character == 'f')
    {
      selection_on_plot(&XREF,&YREF,&X,&Y,image_origin_x,image_origin_y,image_size_x,image_size_y);



      b1 = (int)((XREF/image_size_x)*plot_numSamples);
      b2 = (int)((X/image_size_x)*plot_numSamples);
      c1 = (int)((YREF/image_size_y)*plot_nchans);
      c2 = (int)((Y/image_size_y)*plot_nchans);

      for(i=plot_c1+c1;i<plot_c1+c2;i++)
      {
        for(j=0;j<numSamples;j++)
        {
          for(ii=0;ii<ncrunch;ii++)
          {
            for(jj=0;jj<tcrunch;jj++)
            {
              flags[(i*ncrunch+ii)*numSamples+j*tcrunch+jj]=1;
            }
          }
        }
      }
      adjust_spectra(temp, data, weights, flags, mean, plot_c1, plot_c2, plot_b1, plot_b2, plot_numSamples, numSamples, ncrunch, tcrunch);
      plotDynamicSpectra(temp, f1+ foff*plot_c1*ncrunch, f1+ foff*plot_c2*ncrunch, plot_b1, tsamp*tcrunch, plot_numSamples, plot_nchans,mean,std,th);
    }
    
    if(character == 't')
    { 
      selection_on_plot(&XREF,&YREF,&X,&Y,image_origin_x,image_origin_y,image_size_x,image_size_y);


      
      b1 = (int)((XREF/image_size_x)*plot_numSamples);
      b2 = (int)((X/image_size_x)*plot_numSamples);
      c1 = (int)((YREF/image_size_y)*plot_nchans);
      c2 = (int)((Y/image_size_y)*plot_nchans);
      
      for(i=0;i<nchans;i++)
      { 
        for(j=plot_b1+b1;j<plot_b1+b2;j++)
        { 
          for(ii=0;ii<ncrunch;ii++)
          { 
            for(jj=0;jj<tcrunch;jj++)
            { 
              flags[(i*ncrunch+ii)*numSamples+j*tcrunch+jj]=1;
            }
          }
        }
      }
      adjust_spectra(temp, data, weights, flags, mean, plot_c1, plot_c2, plot_b1, plot_b2, plot_numSamples, numSamples, ncrunch, tcrunch);
      plotDynamicSpectra(temp, f1+ foff*plot_c1*ncrunch, f1+ foff*plot_c2*ncrunch, plot_b1, tsamp*tcrunch, plot_numSamples, plot_nchans,mean,std,th);
    }


    if(character == 'a')
    {
      plot_c1 *= ncrunch;
      plot_c2 *= ncrunch;
      plot_nchans *= ncrunch;
      plot_numSamples *= tcrunch;
      plot_b1 *= tcrunch;
      plot_b2 *= tcrunch;
      
      
      printf("Enter the value of ncrunch: ");
      scanf("%d",&ncrunch);
      printf("value entered is %d \n",ncrunch);
      printf("Enter the value of tcrunch: ");
      scanf("%d",&tcrunch);
      printf("value entered is %d \n",tcrunch);
      //ncrunch = 4;
      //tcrunch = 16;
      plot_c1 /= ncrunch;
      plot_c2 /= ncrunch;
      plot_nchans /= ncrunch;
      plot_numSamples /= tcrunch;
      plot_b1 /= tcrunch;
      plot_b2 /= tcrunch;
      adjust_spectra(temp, data, weights, flags, mean, plot_c1, plot_c2, plot_b1, plot_b2, plot_numSamples, numSamples, ncrunch, tcrunch);
      plotDynamicSpectra(temp, f1+ foff*plot_c1*ncrunch, f1+ foff*plot_c2*ncrunch, plot_b1, tsamp*tcrunch, plot_numSamples, plot_nchans,mean,std,th);
    }
    
    


    if(character == 'd')
    {
      printf("Enter the value of dm: ");
      scanf("%f",&dm);
      printf("value entered is %f and tsamp is %f and plot_dm %f\n",dm,tsamp,plot_dm);
      
      dedisperse_float(data, f1, f2, nchans, numSamples, (double)(dm-plot_dm), 400.0, (double)tsamp);
      plot_dm = dm;
      adjust_spectra(temp, data, weights, flags, mean, plot_c1, plot_c2, plot_b1, plot_b2, plot_numSamples, numSamples, ncrunch, tcrunch);
      plotDynamicSpectra(temp, f1+ foff*plot_c1*ncrunch, f1+ foff*plot_c2*ncrunch, plot_b1, tsamp*tcrunch, plot_numSamples, plot_nchans,mean,std,th);
    }

    if(character == 's')
    {
      save =1;
      plotDynamicSpectra(temp, f1+ foff*plot_c1*ncrunch, f1+ foff*plot_c2*ncrunch, plot_b1, tsamp*tcrunch, plot_numSamples, plot_nchans,mean,std,th);
      save=0;
      plotDynamicSpectra(temp, f1+ foff*plot_c1*ncrunch, f1+ foff*plot_c2*ncrunch, plot_b1, tsamp*tcrunch, plot_numSamples, plot_nchans,mean,std,th);
    }
    
    if(character == 'c')
    {
      while(character != 'd')
      {
        cpgcurs(&X,&Y,&character);
        if(character=='A') th += 0.25;
        else if(character=='X') th -= 0.25;
        else character = 'd';
        plotDynamicSpectra(temp, f1+ foff*plot_c1*ncrunch, f1+ foff*plot_c2*ncrunch, plot_b1, tsamp*tcrunch, plot_numSamples, plot_nchans,mean,std,th);
      }
    }
    
    if(character == 'l')
    {
      int shift=(int)(((200.0e-3)/tsamp));
      
      if(plot_b1-shift>=0) 
      {
        plot_b1 -= shift;
        plot_b2 -= shift;
      }
      else
      {
        plot_b1 = 0;
        plot_b2 = plot_numSamples;
      }
      adjust_spectra(temp, data, weights, flags, mean, plot_c1, plot_c2, plot_b1, plot_b2, plot_numSamples, numSamples, ncrunch, tcrunch);
      plotDynamicSpectra(temp, f1+ foff*plot_c1*ncrunch, f1+ foff*plot_c2*ncrunch, plot_b1, tsamp*tcrunch, plot_numSamples, plot_nchans,mean,std,th);
    }
     
   if(character == 'r')
    {
      int shift=(int)(((100.0e-3)/tsamp));

      if(plot_b2+shift<=numSamples/tcrunch)
      {
        plot_b1 += shift;
        plot_b2 += shift;
      }
      else
      {
        plot_b2 = numSamples/tcrunch;
        plot_b1 = plot_b2-plot_numSamples;
      }
      adjust_spectra(temp, data, weights, flags, mean, plot_c1, plot_c2, plot_b1, plot_b2, plot_numSamples, numSamples, ncrunch, tcrunch);
      plotDynamicSpectra(temp, f1+ foff*plot_c1*ncrunch, f1+ foff*plot_c2*ncrunch, plot_b1, tsamp*tcrunch, plot_numSamples, plot_nchans,mean,std,th);
    }
    
    if(character == 'p')
    {
      float step = 0.001;
      printf("Enter the value of dm step: ");
      scanf("%f",&step);
      
      while(character != 'd')
      {
        cpgcurs(&X,&Y,&character);
        if(character=='A') dm = plot_dm+step;
        else if(character=='X') dm = plot_dm-step;
        else character = 'd';
        printf("DM = %f \n",dm);
        dedisperse_float(data, f1, f2, nchans, numSamples, (-1.0*plot_dm), 400.0, (double)tsamp);
        dedisperse_float(data, f1, f2, nchans, numSamples, dm, 400.0, (double)tsamp);
        plot_dm = dm;
        adjust_spectra(temp, data, weights, flags, mean, plot_c1, plot_c2, plot_b1, plot_b2, plot_numSamples, numSamples, ncrunch, tcrunch);
        plotDynamicSpectra(temp, f1+ foff*plot_c1*ncrunch, f1+ foff*plot_c2*ncrunch, plot_b1, tsamp*tcrunch, plot_numSamples, plot_nchans,mean,std,th);
      }
    }
    
  }

  //fclose(fpin0);
  //fclose(fpin1);
    
  
  
  
  return 0;
}
