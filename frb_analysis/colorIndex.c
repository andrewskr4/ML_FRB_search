#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cpgplot.h>
#include <string.h>
/*
C=====================================================================72
C Calculates a "cube helix" colour table. The colours are a tapered
C helix around the diagonal of the [R,G,B] colour cube, from black
C [0,0,0] to white [1,1,1] Deviations away from the diagonal vary
C quadratically, increasing from zero at black, to a maximum, then
C decreasing to zero at white, all the time rotating in colour.
C
C The input parameters controlling the colour helix are:
C
C    START colour (1=red, 2=green, 3=blue; e.g. 0.5=purple);
C    ROTS  rotations in colour (typically -1.5 to 1.5, e.g. -1.0
C          is one blue->green->red cycle);
C    HUE   for hue intensity scaling (in the range 0.0 (B+W) to 1.0
C          to be strictly correct, larger values may be OK with
C          particular start/end colours);
C    GAMMA set the gamma correction for intensity.
C
C The routine returns a colour table NLEV elements long in RED, GRN
C and BLU (each element in the range 0.0 to 1.0), and the numbers,
C NLO and NHI, of red, green or blue values that had to be clipped
C because they were too low or too high.
C---------------------------------------------------------------------72
C Dave Green --- MRAO --- 2011 June 13th
C---------------------------------------------------------------------72
C See:
C   Green, D. A., 2011, Bulletin of the Astronomical Society of India,
C      Vol.39, p.289
C---------------------------------------------------------------------72
*/


int cubhlx(float start, float rots,float hue,float gamma,int nlev,float *red,float *grn,float *blu,int *nl,int *nh)
{
  int i;
  float pi,fract,angle,amp;
  
  pi = 3.14159265;
  int nlo = 0;
  int nhi = 0;
  
  for(i=0;i<nlev;i++)
  {
    fract = (float)(i-1)/(float)(nlev-1);
    angle = 2*pi*(start/3.0+1.0+rots*fract);
    fract = pow(fract,2);
    amp = hue*fract*(1-fract)/2.0;
    red[i] = fract+amp*(-0.14861*cos(angle)+1.78277*sin(angle));
    grn[i] = fract+amp*(-0.29227*cos(angle)-0.90649*sin(angle));
    blu[i] = fract+amp*(1.97294*cos(angle));
    
    if(red[i]<0.0)
    {
      red[i] = 0.0;
      nlo = nlo+1;
    }
    if(grn[i]<0.0)
    {
      grn[i] = 0.0;
      nlo = nlo+1;
    }
    if(blu[i]<0.0)
    {
      blu[i] = 0.0;
      nlo = nlo+1;
    }
    if(red[i]>1.0)
    {
      red[i] = 1.0;
      nhi = nhi+1;
    }
    if(grn[i]>1.0)
    {
      grn[i] = 1.0;
      nhi = nhi+1;
    }
    if(blu[i]>1.0)
    {
      blu[i] = 1.0;
      nhi = nhi+1;
    }
  }
  nl[0] = nlo;
  nh[0] = nhi;
  
  return 0;
}  

int setCudeHelix(void)
{
  int nlev = 64;
  float start= 3.0;
  float rots = 0.7;
  float hue  = 1.0;
  float gamma = 1.0;
  float red[64], grn[64], blu[64],cl[64];
  int nlo,nhi;
  float atemp;
  
  float contra = 1.0;
  float bright = 0.5;
  int i;      
  cubhlx(start, rots, hue, gamma, nlev, red, grn, blu, &nlo, &nhi);
  atemp = 1.0/(float)nlev;
  for(i=0;i<nlev;i++)
  {
    cl[i] = (i-1)*atemp + atemp/2.0;
  }
  
  cpgctab(cl,red,grn,blu,nlev,contra,bright);
    
  return 0;
}
int setGray(void)
{
  int i=0,count=1;
  float csr,csg,csb;
  //cpgqcol(idx,jj)
  
  for(i=0;i<128;i++)
  {
    csr = 0.0 + (0.5/128.0)*i;
    csg = 0.0 + (0.5/128.0)*i;
    csb = 0.0 + (0.5/128.0)*i;
    cpgscr(count,csr,csg,csb);
    count++;
  }
  
  for(i=0;i<128;i++)
  {
    csr = 0.5 + (0.5/127.0)*i;
    csg = 0.5 + (0.5/127.0)*i;
    csb = 0.5 + (0.5/127.0)*i;
    cpgscr(count,csr,csg,csb);
    count++;
  }
   
  return 0;
}

int setCmap(void)
{
  int i=0,count=1;
  float csr,csg,csb;
  //cpgqcol(idx,jj)
  
  for(i=0;i<20;i++)
  {
    csr = 0.0;
    csg = 0.0;
    csb = 0.0 + (1.0/20.0)*i;
    cpgscr(count,csr,csg,csb);
    count++;
  }
  
  for(i=0;i<50;i++)
  {
    csr = 0.0;
    csg = 0.0 + (1.0/50.0)*i;
    csb = 1.0;
    cpgscr(count,csr,csg,csb);
    count++;
  }  
  
  for(i=0;i<60;i++)
  {
    csr = 0.0;
    csg = 1.0;
    csb = 1.0 - (1.0/60.0)*i;
    cpgscr(count,csr,csg,csb);
    count++;
  }
  
  for(i=0;i<60;i++)
  {
    csr = 0.0 + (1.0/60.0)*i;
    csg = 1.0;
    csb = 0.0;
    cpgscr(count,csr,csg,csb);
    count++;
  }
  
  for(i=0;i<65;i++)
  {
    csr = 1.0;
    csg = 1.0 -(1.0/64.0)*i;
    csb = 0.0;
    cpgscr(count,csr,csg,csb);
    count++;
  }
  //cpgscr(255,1.0,0.0,0.0);
  return 0;
}
int colorIndex(char *palette)
{
  if(strcmp(palette,"CUBE")==0)
  {
  	setCudeHelix();
  }
  if(strcmp(palette,"GRAY")==0)
  {
  	setGray();
  }
  
  if(strcmp(palette,"CMAP")==0)
  {
  	setCmap();
  }
}  
  
