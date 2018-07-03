#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#define pi 3.14159265359


#if __CUDA_ARCH__ < 600
__device__ double katomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do { assumed = old; old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val
					+ __longlong_as_double(assumed)));
	}while (assumed != old);


	return __longlong_as_double(old);
}

#endif


//--------------Determinant and Inverse Code Start HERE-----------------------


__host__ __device__ void findcofact(double*A, double *cofact ,int row ,int col,int dim )
{
	int kl =0;
	for(int pp =0;pp<dim*dim;pp++)       //iterating over all the elements
	{
	if(((pp%dim)!=col)&&(pp/dim!=row) )  //pick the ones who are not in same row and columns as A[i] is
	{
		cofact[kl] = A[pp];            //just copying from main matrix to cofactor matrix
		kl = kl+1;
	}
	}//end of for loop
}


/*-------To find determinant of matrix--------*/
__host__ __device__ double determinant(double *A,int dim)
 { 	 double *cofact = new double [dim*dim];
 	 int sign =  1;
 	 double det = 0;

	 if (dim==1)
	 {det = A[0];
	 }//end of if

 	 else
 	 {
 		for (int i = 0 ; i<dim;i++)  // we use first row
 		{
 			findcofact(A,cofact,0 , i ,dim);
 			if (((i/dim)+(i%dim))%2!=0)
 			{	sign = -1;
 			}
 			else
 			{
 				sign = 1;
 			} //end of else
 			det = det + sign*determinant(cofact,dim-1)*A[i];
 		 }//end of for
     }//end of else
	 delete cofact;
	 return det;
 }

/*-------To find adjoint of matrix--------*/
__host__ __device__ void findAdjoint(double *A ,double *adjoint ,int dim )
{   int index = 0; int sign =0 ;
	for (int i =0 ;i <dim*dim;i++)   // find cofactor of all elements and place it on adjoint matrix with sign
	{   if (((i/dim)+(i%dim))%2!=0)
		{
		sign = -1;
		}//end of if
		else
		{
		sign = 1;
		}// end of else
	 	double *cofact = new double [dim*dim];
	 	findcofact(A,cofact,i/dim,i%dim,dim);
		adjoint [index++] = sign*determinant (cofact,dim-1);
		delete cofact;
	 }//end of for
	/*----for transposing ---*/
	for (int jj =0 ;jj<dim;jj++)
	{
		for (int ii=jj;ii<dim;ii++)
		{
			double temp = adjoint[jj*dim+ii];
			adjoint[jj*dim+ii] = adjoint[ii*dim+jj];
			adjoint[ii*dim+jj] = temp;
		}
	}//end of for

}

/*-------To find inverse of matrix--------*/
__host__ __device__ double getInverse(double *A,int dim, double *invA)
{
	double* adjoint = new double [dim*dim];
	double x = determinant(A,dim);
	findAdjoint(A,adjoint,dim);
	//for (int i =0 ;i <d*d;i++)
	//{
	//	printf("%f ",adjoint[i]);
	//}
	for (int i =0 ;i <dim*dim;i++)
	{	invA[i] = adjoint[i]/x;
		if(invA[i]== -0)
		invA[i] = 0;
	}
	delete adjoint;
	return x;
}//end of function

//--------------Determinant and Inverse Code End Here-------------------------

__device__ double pdf(double* point, double *mean, double *Invcov,double det, int dim){

	double lo = pow(2*pi,dim/(2.0))*sqrt(det);
	double* res = new double[dim];
	for(int i=0;i<dim;i++){
		res[i] = point[i] - mean[i];
	}
	double* val = new double[dim];

	for(int i=0;i<dim;i++){
		val[i] = 0;
		for(int j=0;j<dim;j++){
			val[i] = val[i] + res[j]*Invcov[j*dim + i];
		}
	}

	double up = 0;

	for(int i=0;i<dim;i++){
		up = up + val[i]*res[i];
	}

	up = (-0.5)*up;

	up = exp(up);
	delete val;
	delete res;
	return (up/lo);

}
__global__ void likeli(double *d_point,int n,int dim,int kl ,double *d_mixK,double *d_mean,double *d_cov,double* det,double* invCov,double* d_like)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if(id<n){

		double sum = 0;
		for(int i=0;i<kl;i++){
			sum = sum + d_mixK[i]*pdf(&d_point[id*dim],&d_mean[i*dim],&invCov[i*dim*dim],det[i],dim);
			}
		d_like[id]= sum;
	}
}// end of kernel

int main (int argc, char ** argv)
{  freopen(argv[3],"w",stdout);  // output willl be saved in thi s file
	FILE *file = fopen(argv[1], "r");
	int kl;int dim ;
	int i =0 ;
	fscanf(file,"%d",&kl);
	fscanf(file,"%d",&dim);
	double *h_mixK = (double *)malloc(kl*sizeof(double));
	double *h_mean =(double *)malloc(kl*dim*sizeof(double));
	double *h_cov=(double *)malloc(kl*dim*dim*sizeof(double));
	double *det = (double*) calloc(kl,sizeof(double));
	double *h_invMat = (double*) calloc(kl*dim*dim,sizeof(double));
	
	int j = 0;

	for(j = 0 ;j<kl;j++){
	fscanf(file,"%lf",&h_mixK[j]);
	}
	for(j = 0 ;j<kl*dim;j++){
	fscanf(file,"%lf",&h_mean[j]);
	}
	for(j = 0 ;j<kl*dim*dim;j++){
	fscanf(file,"%lf",&h_cov[j]);
	}

	fclose(file);
	file = fopen(argv[2], "r"); //this file will contain testing data
	 int n ;
	fscanf(file, "%d", &n);  //to find the number of points in d file
	double *h_points = (double *)malloc(n*dim*sizeof(double));
    	for(int ii =0 ;ii<n*dim ;ii++){
    	fscanf(file,"%lf",&h_points[ii]);
   }
    
    int threads =0; int blocks =0 ;
    if(n<=1024)
    {
      	threads = n;
      	blocks = 1;
    }
    else
    {
      threads = 1024;
      blocks = ceil(n/(1024.0));
    }
    double *h_like = (double*) calloc(n,sizeof(double));

    double *d_point, *d_mixK, *d_mean, *d_cov,*d_det,*d_invMat,*d_like;
    cudaMalloc(&d_det,kl*sizeof(double));
    cudaMalloc(&d_point,n*dim*sizeof(double));
    cudaMalloc(&d_mixK ,kl*sizeof(double));
    cudaMalloc(&d_mean ,kl*dim*sizeof(double));
    cudaMalloc(&d_cov  ,kl*dim*dim*sizeof(double));
    cudaMalloc(&d_det  ,kl*sizeof(double));
    cudaMalloc(&d_invMat,kl*dim*dim*sizeof(double));
    cudaMalloc(&d_like,n*sizeof(double));
    cudaMemcpy(d_point,h_points,n*dim*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_mixK,h_mixK,kl*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean,h_mean,kl*dim*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_cov,h_cov,kl*dim*dim*sizeof(double),cudaMemcpyHostToDevice);
    for(i=0;i<kl;i++){

    	det[i] = determinant(&h_cov[i*dim*dim],dim);

    	if(det[i]==0.0){
    		for (int l =0 ;l<dim;l++){

    			h_cov[i*dim*dim + dim*l + l] = h_cov[i*dim*dim + dim*l + l]+1;   //adding 1 to diagonal elements
    			}
    	}
    	det[i] = getInverse(&h_cov[i*dim*dim],dim,&h_invMat[i*dim*dim]);
    }

    cudaMemcpy(d_invMat,h_invMat,kl*dim*dim*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_det,det,kl*sizeof(double),cudaMemcpyHostToDevice);

    likeli<<<blocks,threads>>>(d_point,n,dim,kl,d_mixK,d_mean,d_cov,d_det,d_invMat,d_like);
    cudaDeviceSynchronize();

    cudaMemcpy(h_like,d_like,n*sizeof(double),cudaMemcpyDeviceToHost);
    printf("%d\n",n);// to print no. of points
    for(int i = 0; i <n;i++){

       printf("%f\n",h_like[i]);
    }
	return 0 ;

}
