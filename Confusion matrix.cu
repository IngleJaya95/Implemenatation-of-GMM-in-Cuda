#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <bits/stdc++.h>

using namespace std;



int main(int argc,char* argv[]){

// class1M1 class1M2 class2M1 class2M2
	if(argc!=5){
		cout<<"\nThere should be 4 file class1TestWM1 - class1TestWM2 - class2TestWM1 - class2TestWM2\n";
		exit(23);
	}

	// Retriving file Name
	char* class1M1 = argv[1];
	char* class1M2 = argv[2];
	char* class2M1 = argv[3];
	char* class2M2 = argv[4];

	
	// Declaring object for the file for input
	FILE* c1m1 = fopen(class1M1, "r");
	FILE* c1m2 = fopen(class1M2, "r");
	FILE* c2m1 = fopen(class2M1, "r");
	FILE* c2m2 = fopen(class2M2, "r");

	// For reading no of line in each files
	int n1,n2,n3,n4;

	// Scaning no of lines in each stream
	fscanf(c1m1,"%d",&n1);
	fscanf(c1m2,"%d",&n2);
	fscanf(c2m1,"%d",&n3);
	fscanf(c2m2,"%d",&n4);


	int i=0,j=0;

	// Declaring memory
	double *sc1m1 = (double*)malloc(n1*sizeof(double));
	double *sc1m2 = (double*)malloc(n2*sizeof(double));
	double *sc2m1 = (double*)malloc(n3*sizeof(double));
	double *sc2m2 = (double*)malloc(n4*sizeof(double));

	for(i=0;i<n1;i++){
		fscanf(c1m1,"%lf",&sc1m1[i]);
	}

	for(i=0;i<n2;i++){
		fscanf(c1m2,"%lf",&sc1m2[i]);
		}
	for(i=0;i<n3;i++){
		fscanf(c2m1,"%lf",&sc2m1[i]);
		}
	for(i=0;i<n4;i++){
		fscanf(c2m2,"%lf",&sc2m2[i]);
		}

	int* confusion = (int*) calloc(4,sizeof(int));

	for(i=0;i<n1;i++){
		if(sc1m1[i]>sc1m2[i]){
			confusion[0] = confusion[0] + 1;
		}
		else{
			confusion[1] = confusion[1] + 1;
		}
	}

	for(i=0;i<n3;i++){
			if(sc2m1[i]>sc2m2[i]){
				confusion[2] = confusion[2] + 1;
			}
			else{
				confusion[3] = confusion[3] + 1;
			}
		}


	for(i=0;i<2;i++){
		for(j=0;j<2;j++){
			printf("%d ",confusion[i*2+j]);
		}
		printf("\n");
	}

	fclose(c1m1);
	fclose(c1m2);
	fclose(c2m1);
	fclose(c2m2);
	return 0;
}
