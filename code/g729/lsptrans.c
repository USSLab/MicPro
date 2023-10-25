#include <stdio.h>
#include <math.h>
#include "typedef.h"
#include "ld8k.h"

double trans_func(double x, double alpha, double beta);  /* transform function of lsf*/
void bubble_sort(double num[],int n);


void lsp_trans(
    Word16 lsp[]   /* (i/o) Q15 : line spectral frequencies            */
)
{
    int i,j;
    double lsf[10];
    double a[3];
    double k;
    FILE *fp = NULL;
    fp = fopen("param_moo.txt", "r");
    for(i=0;i<3;i++)
    {
        fscanf(fp, "%lf", &k);
        a[i] = k;
    }
    fclose(fp);
    for(i=0; i<M; i++)
    {   
        lsf[i] = lsp[i]/32768.0;
    }


    double xi1;
    xi1 = a[0];
    for(i=0; i<M; i++)
    {   
        lsf[i] = lsf[i] + lsf[i]*(xi1-1)*(1-lsf[i]);
    }


    double xi2;
    xi2 = a[1];
    for(i=0; i<M; i++)
    {
        lsf[i] = lsf[i] + (xi2-1)*sin(2*3.1415926*lsf[i])/2/3.1415926;
    }


    double xi3;
    xi3 = a[2];
    double delta[11] = {0};
    double delta_new[11] = {0};
    double lsf_new[10] = {0};
    for(i=1;i<10;i++){
        delta[i] = lsf[i] - lsf[i-1];
    }
    delta[0] = lsf[0];
    delta[10] = 1 - lsf[9];
    for(i=0;i<11;i++){
        delta_new[i] = delta[i] + (xi3-1)*(1/11.0-delta[i]);
    }
    for(i=0;i<10;i++){
        for(j=0;j<i+1;j++){
            lsf_new[i] += delta_new[j];
        }
    }
    for(i=0; i<M; i++)
    {   
        lsf[i] = lsf_new[i];
    }


    for(i=0; i<M; i++)
    {   
        // lsf[i] = lsf_new[i];
        lsp[i] = lsf[i]*32768;
    }
}

double trans_func(double x, double alpha, double beta)
{
    double y;
    if(x<alpha)
        y = beta/alpha*x+0.1;
    else
        y = (1-beta)/(1-alpha)*(x-alpha)+beta+0.1;
    return y;
}

void bubble_sort(double num[],int n) 
{
	int i,j; 
    float t;
	for(i=0;i<n-1;i++) 
	{
		for(j=0;j<n-1-i;j++) 
		{
			if(num[j]>num[j+1]) 
			{
				t=num[j+1];
				num[j+1]=num[j];
				num[j]=t;
			}
		}
	} 
}
