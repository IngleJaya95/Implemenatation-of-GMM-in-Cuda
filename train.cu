#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
//#define n 10
//#define d 2
//#define k 3
#define pi 3.14159265359
#define GMMITR 5

// Return Distance between two points

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
	 	// for testing ------
	 	//printf("\nfor A[%d]",i);
	 	//for (int i =0 ;i <(d-1)*(d-1);i++)
	 	//{
	 		//printf("%f " ,cofact[i]);
	 	//}
	 	//printf("\n");
		adjoint [index++] = sign*determinant (cofact,dim-1);
		//printf("\n adjoint for A[%d] is %f",i,adjoint[index-1]);
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

__host__ __device__ double distBW(double* a,double* b,int dim){
	int i =0;
	double dist = 0; // Initializing Distance Between two points
	for(i=0;i<dim;i++){
		double t = a[i]-b[i];
		dist = dist + (t*t);
	}

	dist = sqrt(dist);

	return dist;
}

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

/*
 * Kernel call for the mean.
 * Take Input : Data_Points, No. of points, No. of dimensions, Old mean, new Mean,
 * 				d_dist(contain the distance of point from it's cluster)
 * 				d_noInK(no of points in cluster K)
 */

__global__ void kernKMean(double* point,int np,int dim,int kl,double* old_mean,double* new_mean,int* d_index,double* d_noInK,double* d_Dist){
	int id = (blockDim.x * blockIdx.x) + threadIdx.x;
	if(id<np){

		double minDistance = -1;
		double distance = 0;
		int minIndex;

		/*
		 * Calculating distance from all of the mean
		 * and Storing the minimum one.
		 */

		for(int kk=0;kk<kl;kk++){
			distance = distBW(&point[id*dim],&old_mean[kk*dim],dim);
			if(minDistance == (-1)){
				minDistance = distance;
				minIndex = kk;
			}

			//printf("%f\n",distance);
			if(distance < minDistance){
				minDistance = distance;
				minIndex = kk;
			}
		}
		//printf("%d\n",minIndex);

		//Storing the min Identity
		d_index[id] = minIndex;

		//Add the val no of point in Cluster
		katomicAdd(&d_noInK[minIndex],1); // Checked Working fine

		//Store the value of point to Indexed Mean it assigned

		for(int ij=0;ij<dim;ij++){
			double aa = point[(id*dim)+ij]; // Taking the value of point out
			katomicAdd(&new_mean[minIndex*dim + ij],aa);
		}

		//Store the minDistance to the convergence parameter
		katomicAdd(d_Dist,minDistance);
		//Divide New Mean by no of point in corresponding cluster

		// Check for the convergence
	}
}


__global__ void kernCov(double* point,int np,int dim,int kl,double* d_covMat,int* d_index,double* d_noInK,double* mean){

	// All input variable checked working fine

	int id = threadIdx.x + (blockIdx.x*blockDim.x);
	if(id<np){
		int identity = d_index[id];
		double *pmd = new double[dim];

		for(int i=0;i<dim;i++){
			pmd[i] = point[id*dim + i] - mean[identity*dim + i];
			//printf("%f\n",pmd[i]);
		}


		double val = 0;
		int ref = 0;
		double iden = d_noInK[identity];
		//Constructing Covariance Matrix
		for(int i=0;i<dim;i++){
			for(int j=0;j<dim;j++){
				val = (pmd[i]*pmd[j])/iden;
				ref = identity*dim*dim + i*dim + j;
				katomicAdd(&d_covMat[ref],val);
			}
		}

		delete pmd;
	}



	//Testing covMat
	//Testing noInK
	//Testing oldMean
}


__global__ void reskern(double* point,double* mean,int np,int dim,int kl,double* res,double* mixC,double* det,double* invCov,double* lnSum,double* noInK){

	/*
	 * 	Each thread will calculate responsibility of each cluster with respect to
	 *	single point.
	 *
	 *	Then it will store ln of sum of mixing_Coefficient(kth_cluster) * pdf(with_kth_cluster_parameter)
	 *
	 */

	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if(id<np){

		//Initializing Sum
		double sum = 0;
		double* tempRes = new double[kl];

		for(int i=0;i<kl;i++){

			tempRes[i] = mixC[i]*pdf(&point[id*dim],&mean[i*dim],&invCov[i*dim*dim],det[i],dim);
			sum = sum + tempRes[i];
		}

		for(int i=0;i<kl;i++){
			 double tres = tempRes[i]/sum;
			 res[id*kl + i] = tres;
			 katomicAdd(&noInK[i],tres);
		}

		sum = log(sum);

		katomicAdd(lnSum,sum);
		delete tempRes;
	}

}


__global__ void kern_newMean(double* point,double* newMean,int np,int dim,int kl,double* res,double* noInK){
	// res --> n*k per point per cluster
	// newMean --> k*d per cluster per dimension
	// point --> n*d per dimension

	int id = threadIdx.x + blockDim.x*blockIdx.x;

	if(id < np){

		for(int i=0;i<kl;i++){

			double r = res[id*kl+i];
			double nk = noInK[i];

			for(int j=0;j<dim;j++){

				double pVal = (r*point[id*dim+j])/nk;
				katomicAdd(&newMean[i*dim+j],pVal);

			}

		}

	}
}


__global__ void kernGMCov(double* point,double* mean,int np,int dim,int kl,double* res,double* noInK,double* covMat){

	int id = threadIdx.x + blockDim.x*blockIdx.x;

	if(id<np){

		//Selecting cluster
		for(int i=0;i<kl;i++){
			// i indicate cluster no

			// Taking out no of point in that cluster
			double nk = noInK[i];

			// Taking out mean with respect to that cluster
			double* mn = &mean[i*dim];

			// Taking out responsibility  of the point proportional to cluster
			double rs = res[id*kl+i];

			// creating dimension size array to store difference
			double* pd = new double[dim];

			// Calculating difference of the point and mean corresponding to there index
			for(int j=0;j<dim;j++){
				pd[j] = point[id*dim+j] - mn[j];
			}

			// Multiplying each pd with each pd multiplying that with responsibility
			// and Dividing by the no. of points.
			for(int p=0;p<dim;p++){
				for(int q=0;q<dim;q++){
					double covEnt = rs*((pd[p]*pd[q])/nk);
					// i*dim*dim select the cov matrix
					// p*dim select row in the matrix
					// q select the column
					// This is the right position of cov matrix
					katomicAdd(&covMat[i*dim*dim + p*dim + q],covEnt);
				}
			}

			delete pd;
		}


	}

}


int main(int argc ,char** argv){
int n,d,k;

scanf("%d",&n);
scanf("%d",&d);
scanf("%d",&k);

// K-mean Start
double *point = (double*)malloc(n*d*sizeof(double));

for(int fi=0;fi<n;fi++){
	for(int fj=0;fj<d;fj++){
		scanf("%lf",&point[fi*d+fj]);
	}
}


double* mean = (double*)malloc(k*d*sizeof(double));
// initialization of K means
double diff = (n/(k*1.0));
double prevDist = 0.0;
// The Needs of kernel
int* d_index; //Store Identity of all point n*1
double* d_oldMean;
double* d_newMean;
double* d_point;
double* d_noInK;
double* d_Dist;

// Threads and Blocks
int threads=0,blocks=0;

//Iterators
int i=0,j=0;

//Taking Initial Means for the K-means Algorithms
//#pragma omp parallel for
for(i=0;i<k;i++){
	//Taking out the point to be assign as mean.
	int ind = floor(diff*i);
	for(j=0;j<d;j++){
		mean[i*d+j] = point[ind*d+j]; // Copying the point at ind Index to the point at ith mean
	}
}

/* Moving Data on the device for computation
	- Assigning Memory
	- Moving Data form Host to the Device
*/

// Assigning Memory
cudaMalloc(&d_point,n*d*sizeof(double));
cudaMalloc(&d_index,n*1*sizeof(int));
cudaMalloc(&d_newMean,k*d*sizeof(double));
cudaMalloc(&d_oldMean,k*d*sizeof(double));
cudaMalloc(&d_noInK,k*1*sizeof(double));
cudaMalloc(&d_Dist,1*sizeof(double));


// Copying from host to Device
cudaMemcpy(d_point,point,n*d*sizeof(double),cudaMemcpyHostToDevice);

/* Defining no of threads required to launch the kernel
	- If n <= 1024 then single block and 1024 threads will be fine
	- but if n> 1024 the block required will be more
*/


if(n<=1024){
	threads = n;
	blocks = 1;
}
else{
	threads = 1024;
	blocks = ceil(n/(1024.0));
}
int itr = 0; //for iteration

double *h_newMean = (double*)malloc(k*d*sizeof(double));
double *h_noInK = (double*)malloc(k*sizeof(double));
double *h_Dist = (double*)malloc(1*sizeof(double));

while(true){
	//Initializing d_newMean = 0;

	//#pragma omp parallel for
	for(int jj=0;jj<k*d;jj++){
		h_newMean[jj] = 0;
	}

	//Initializing noInK
	//#pragma omp parallel for
	for(j=0;j<k;j++){
		h_noInK[j] = 0;
	}

	//Initializing d_Dist

	h_Dist[0] = 0;
	cudaMemcpy(d_Dist,h_Dist,sizeof(double),cudaMemcpyHostToDevice);

	cudaMemcpy(d_oldMean,mean,k*d*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_newMean,h_newMean,k*d*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_noInK,h_noInK,k*1*sizeof(double),cudaMemcpyHostToDevice);

	//Kernel_For_Kmean_launch
	kernKMean<<<blocks,threads>>>(d_point,n,d,k,d_oldMean,d_newMean,d_index,d_noInK,d_Dist);
	cudaDeviceSynchronize();

	//int * ind = (int*)malloc(n*1*sizeof(int)); //Testing Indicator

	//Copying noInK and newMean to the host
	cudaMemcpy(h_newMean,d_newMean,k*d*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_noInK,d_noInK,k*1*sizeof(double),cudaMemcpyDeviceToHost);

	//#pragma omp parallel for
	for(j=0;j<k;j++){
		for(int pp=0;pp<d;pp++){
			mean[(j*d)+pp] = h_newMean[(j*d)+pp]/h_noInK[j];
		}
		//printf("%f %f\n",mean[j*d],mean[j*d+1]);

	}

	//printf("\n\n");
	cudaMemcpy(h_Dist,d_Dist,sizeof(double),cudaMemcpyDeviceToHost);
	if(itr == 0){
		prevDist = h_Dist[0];
	}
	else{
		if(abs(prevDist-h_Dist[0])<0.001){
		break;
		}
		prevDist = h_Dist[0];
	}
itr++;
printf("%d\n",itr);
}


//---------------------K Mean Finished --------------------------------

//---------- Covariance Matrix in parallel Code ----------------------------

/*
 * Input -- Identity of points
 * 			Data Points
 * 			Mean of points
 * Output -- Covariance Matrix d*d
 */

double* h_covMat = (double*) calloc(k*d*d,sizeof(double));
double* d_covMat;
// Initializing Covariance Matrix by 0's

for(j=0;j<k*d*d;j++){
	h_covMat[j] = 0;
}

// Assigning Memory on the device
cudaMalloc(&d_covMat,k*d*d*sizeof(double));
// Copying Initial CovMat to device
cudaMemcpy(d_covMat,h_covMat,k*d*d*sizeof(double),cudaMemcpyHostToDevice);
//Copying mean at d_oldMean
cudaMemcpy(d_oldMean,mean,k*d*sizeof(double),cudaMemcpyHostToDevice);

/*
 * Launching Kernel
 */

kernCov<<<blocks,threads>>>(d_point,n,d,k,d_covMat,d_index,d_noInK,d_oldMean);
cudaDeviceSynchronize();

cudaMemcpy(h_covMat,d_covMat,k*d*d*sizeof(double),cudaMemcpyDeviceToHost);


cudaFree(d_index);

/*
 *
 * Initialization data for GMM finished
 * NOW GMM start
 *
 */

/*
 *
 * GMM - Consist of two steps that are done iteratively
 * 		1. Expectation - In E Steps we find responsiblity of points
 * 		2. Maximization - In Maximization step we find new mean, Covariance and mixing-factor
 *
 *
 */

// Esteps needs Covariance Matrix, Mixing Coefficient, mean -- output mixing coefficient

// finding Mixing Coefficient



double* h_mixC = (double*)malloc(k*sizeof(double));
double* d_mixC;
double* h_res = (double*)calloc(n*k,sizeof(double));
double* d_res;
cudaMalloc(&d_res,n*k*sizeof(double));
cudaMalloc(&d_mixC,k*sizeof(double));
double* det = (double*) calloc(k,sizeof(double));
double* h_invMat = (double*) calloc(k*d*d,sizeof(double));
double* d_det;
cudaMalloc(&d_det,k*sizeof(double));
double* d_invMat;
cudaMalloc(&d_invMat,k*d*d*sizeof(double));
double* d_lnSum;
cudaMalloc(&d_lnSum,sizeof(double));
double* h_lnSum = (double*)calloc(1,sizeof(double));


//-------- Mixing Coefficient-----------------
//#pragma omp parallel for
for(i=0;i<k;i++){
	h_mixC[i] = (h_noInK[i]*1.0)/n;
	//printf("%f \n",h_mixC[i]);
}


//------------------- Reapeatative Tasks --------------------------------------
//-------------------GGGGGGGG    MMM    MMM  MMM    MMM------------------------------
//------------------G        G   M  M  M  M  M  M  M  M
//----------------- G            M   MM   M  M   MM   M
//                  G	GGGGG	 M        M  M        M
//					G		G    M		  M	 M	      M
//					 GGGGGGG     M		  M	 M	      M


double prelnSum = 0;


for(int gmItr=0;gmItr<GMMITR;gmItr++){

	/*
	 * --- CheckList ----
	 * 1. Determinant -----------------Done
	 * 2. Mixing Coefficient-----------
	 * 3. Inverse Matrix----------Done
	 * 4.
	 */

// Taking Inverse and determinant of covariance matrix
//#pragma omp parallel for
for(i=0;i<k;i++){

	det[i] = determinant(&h_covMat[i*d*d],d);

	if(det[i]==0.0){
		for (int l =0 ;l<d;l++)
					{
						h_covMat[i*d*d + d*l + l] = h_covMat[i*d*d + d*l + l]+1;   //adding 1 to diagonal elements
					}
	}

	det[i] = getInverse(&h_covMat[i*d*d],d,&h_invMat[i*d*d]);
}



//------------------------------------------CODE E-------------------------------------------

//Creating Space for mixing coefficient on the device

cudaMemcpy(d_mixC,h_mixC,k*sizeof(double),cudaMemcpyHostToDevice); // Copying From host to the device
cudaMemcpy(d_invMat,h_invMat,k*d*d*sizeof(double),cudaMemcpyHostToDevice);
cudaMemcpy(d_det,det,k*sizeof(double),cudaMemcpyHostToDevice);

//Responsibility size n*k


//lnSum initialization Remained
h_lnSum[0] = 0;
cudaMemcpy(d_lnSum,h_lnSum,sizeof(double),cudaMemcpyHostToDevice);
//Either initialize or comment I

//#pragma omp parallel for
for(j=0;j<k;j++){
		h_noInK[j] = 0;
	}

cudaMemcpy(d_noInK,h_noInK,1*k*sizeof(double),cudaMemcpyHostToDevice);

// output d_res, d_lnSum, d_noInK
reskern<<<blocks,threads>>>(d_point,d_oldMean,n,d,k,d_res,d_mixC,d_det,d_invMat,d_lnSum,d_noInK);
cudaDeviceSynchronize();


cudaMemcpy(h_res,d_res,n*k*sizeof(double),cudaMemcpyDeviceToHost);

cudaMemcpy(h_noInK,d_noInK,1*k*sizeof(double),cudaMemcpyDeviceToHost);



// lnSum --> for convergence


//Finding New Mean --- || -----------------------
// Requirement Responsibility, Points, Nk, newMean

//Initializing newMean on the kernel
//#pragma omp parallel for
for(i=0;i<k;i++){
	for(j=0;j<d;j++){
		h_newMean[i*d + j] = 0;
	}
}
// Initializing newMean From 0 on device
cudaMemcpy(d_newMean,h_newMean,k*d*sizeof(double),cudaMemcpyHostToDevice);

kern_newMean<<<blocks,threads>>>(d_point,d_newMean,n,d,k,d_res,d_noInK);
cudaDeviceSynchronize();
//printf("\n -------New Mean------ \n");
cudaMemcpy(h_newMean,d_newMean,k*d*sizeof(double),cudaMemcpyDeviceToHost);

//Call the Kernel for the newMean

cudaMemcpy(mean,d_newMean,k*d*sizeof(double),cudaMemcpyDeviceToHost);
cudaMemcpy(d_oldMean,mean,k*d*sizeof(double),cudaMemcpyHostToDevice);

/*
// Copy newMean to the oldMean
// Move it on the device

// Covariance Matrix for all the cluster

 *
 *  Input : responsibility, noInK,d_oldMean, output d_cov
 *
// Preparing covaraince matrix
*/
//#pragma omp parallel for
for(i=0;i<k*d*d;i++){
	h_covMat[i] = 0;  //Initializing with 0
}

//Moving Data to kernel
cudaMemcpy(d_covMat,h_covMat,k*d*d*sizeof(double),cudaMemcpyHostToDevice);

kernGMCov<<<blocks,threads>>>(d_point,d_oldMean,n,d,k,d_res,d_noInK,d_covMat);
cudaDeviceSynchronize();

cudaMemcpy(h_covMat,d_covMat,k*d*d*sizeof(double),cudaMemcpyDeviceToHost);

//#pragma omp parallel for
for(i=0;i<k;i++){
	h_mixC[i] = (h_noInK[i]*1.0)/n;
	//printf("%f \n",h_mixC[i]);
}

cudaMemcpy(d_mixC,h_mixC,k*sizeof(double),cudaMemcpyHostToDevice);
cudaMemcpy(h_lnSum,d_lnSum,sizeof(double),cudaMemcpyDeviceToHost);


printf("ITR-----  %f----------  %f------------------------------------------\n",prelnSum,h_lnSum[0]);
if(gmItr==0){
	prelnSum = h_lnSum[0];
}
else if (abs(prelnSum-h_lnSum[0])<0.1 ){
	printf("Breaking the loop at gmItr--%d  ",gmItr);
	break;
}
else{
	prelnSum = h_lnSum[0];
}


}//end of main gmItr loop

printf("K ---Means ----------\n");
for(i=0;i<k;i++){
	for(j=0;j<d;j++){
		printf("%f ",mean[i*d+j]);
	}
	printf("\n");
}

printf("\n===Mixing_Coefficient-----\n");

for(int i=0;i<k;i++){
	printf("%f ",h_mixC[i]);
}

printf("\n===covariance -----\n");

for(i=0;i<k;i++){
	printf("for kl %d\n",i);
	for(j=0;j<d;j++){
		for(int pp=0;pp<d;pp++){
			printf("%f ",h_covMat[i*d*d+j*d+pp]);
		}
		printf("\n");
	}
	printf("\n\n");
}


FILE *fp;
fp = fopen(argv[1], "w");// "w" means that we are going to write on this file
fprintf(fp, "%d %d\n",k,d);
for(i=0;i<k;i++){
	fprintf(fp,"%f ",h_mixC[i]);
}
fprintf(fp, "\n");

for(i=0;i<k*d;i++){
	fprintf(fp,"%f ",mean[i]);
}
fprintf(fp, "\n");
for(i=0;i<k*d*d;i++){
	fprintf(fp,"%f ",h_covMat[i]);
}

return 0;
}
