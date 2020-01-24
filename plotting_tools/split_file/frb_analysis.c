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
  
  return 0;
}





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
